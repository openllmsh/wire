/**
 * Context-skip gate — the per-hop context decision, shared verbatim by
 * the cloud dispatch chain (BYOK) and the daemon walker (device) so the
 * two paths cannot drift.
 *
 * The context "ladder" is deliberately thin (reference-proxy parity —
 * ref/CLIProxyAPI does no gateway-side rewriting either):
 *
 *   Plan A — serve correct per-model context metadata (and pass through
 *            Codex's own `/responses/compact`) so the CLIENT compacts
 *            itself with a STABLE prefix — the only compaction that
 *            preserves prompt-cache affinity;
 *   gate   — THIS module: skip a hop only when the estimate CLEARLY
 *            exceeds its window (saves the doomed round trip + prefill
 *            wait);
 *   walk   — the pre-commit first-event peek
 *            (`lib/streaming/peek.ts`): the real upstream tokenizer gets
 *            the final word, and a pre-output rejection walks the chain
 *            with zero double-spend.
 *
 * Gateway-side tool-output rewriting (the former "Plan B") was removed
 * on purpose: estimator-driven per-request rewrites destabilise the
 * conversation prefix (prompt-cache misses → quota drain) and silently
 * degrade requests the vendor would have accepted.
 */

/**
 * Confidence multiplier before a hop is abandoned over context. The
 * routing estimator is intentionally conservative (it counts every string
 * value), so an estimate above the catalogued input budget is enough to skip
 * a non-final hop: that model has already advertised it cannot accept the
 * request. Final and unknown-limit hops still reach the real tokenizer.
 *
 * A factor above 1 let known-over-budget requests through to an inevitable
 * upstream context rejection. Keep this at one unless the estimator contract
 * itself changes.
 */
export const CONTEXT_SKIP_CONFIDENCE_FACTOR = 1;

/**
 * Conservative ruler→vendor-token calibration for last-resort compaction. The
 * local `estimateBodyTokens` ruler can count fewer tokens than a vendor's own
 * tokenizer, so target every provider below its advertised window until the
 * vendor-grounded overflow path corrects from a real rejection.
 */
// One shared constant across providers (collapsed from the old per-provider
// table). 1.2 is a moderate calibration: it does not over-compact grok/openai
// (higher values shrink the target and dull long contexts), and the
// vendor-grounded overflow path corrects the rare Claude first-pass rejection.
export const TOKEN_ESTIMATE_FACTOR = 1.2;

/**
 * The compaction target for a hop: its input window shrunk by the shared
 * calibration factor (so a body compacted to this size fits the vendor's real
 * tokenizer, not just our ruler) and by the shared context-gate safety factor.
 * Single source for BOTH seams (cloud dispatch chain + daemon walker) so the
 * target can't drift.
 */
export const compactionTargetFor = (
  _provider: string,
  window: number,
): number =>
  Math.floor((window / TOKEN_ESTIMATE_FACTOR) * CONTEXT_SKIP_CONFIDENCE_FACTOR);

/**
 * Single safety margin applied to the vendor-grounded compaction target: aim
 * the shrunk body at `window × SAFETY` implied vendor tokens, leaving a small
 * headroom band so the retry clears the real tokenizer even if the ratio drifts
 * slightly between passes. ONE knob for every provider — the calibration itself
 * is self-derived from the vendor's own numbers, so there is no per-model
 * branching here.
 */
export const COMPACTION_OVERFLOW_SAFETY = 0.92;

/**
 * A sane lower bound for any computed compaction target. A pathological vendor
 * ratio (or a tiny window) must never drive the target to zero / negative — the
 * cut would then strip the live turn too. 512 tokens comfortably holds the
 * system prompt + the final user turn the compactor always preserves.
 */
export const COMPACTION_TARGET_FLOOR = 512;

/**
 * Vendor-grounded, self-calibrating compaction target. When the upstream
 * rejected a body on size it reports two ground-truth numbers: `requiredTokens`
 * (how many tokens ITS tokenizer counted for the body we sent) and, implicitly,
 * the `window` it must fit. Our local ruler measured that same body at
 * `localEstimate`, so the observed ratio `r = requiredTokens / localEstimate`
 * is exact calibration for THIS conversation's content — no static per-provider
 * factor needed.
 *
 * We return a target in LOCAL ruler space: shrinking the body to
 * `floor(window / r × SAFETY)` local tokens implies a vendor count of
 * `<= window × SAFETY`, i.e. inside the window with headroom. The result is
 * clamped to be no looser than `compactionTargetFor(provider, window)` (the
 * static target is an upper bound — vendor grounding may only tighten) and never
 * below `COMPACTION_TARGET_FLOOR`. When the vendor gave no usable count we fall
 * back to the static target. Generic across grok / chatgpt / anthropic / kimi —
 * whatever numbers the vendor surfaces drive the cut.
 */
export const compactionTargetFromOverflow = (params: {
  readonly requiredTokens: number | null;
  readonly window: number;
  readonly localEstimate: number;
  readonly provider: string;
}): number => {
  const staticTarget = compactionTargetFor(params.provider, params.window);
  if (
    params.requiredTokens === null ||
    params.requiredTokens <= 0 ||
    params.localEstimate <= 0
  ) {
    return staticTarget;
  }
  const ratio = params.requiredTokens / params.localEstimate;
  const vendorGrounded = Math.floor(
    (params.window / ratio) * COMPACTION_OVERFLOW_SAFETY,
  );
  // Vendor-grounded may only TIGHTEN the static target. The floor is normally
  // COMPACTION_TARGET_FLOOR, but a small live `context_window` can make the
  // static target itself dip below that floor — clamping up to 512 would then
  // return a target LOOSER than the window allows. Cap the floor at the static
  // target so the result is always `<= staticTarget`.
  const floor = Math.min(COMPACTION_TARGET_FLOOR, staticTarget);
  return Math.max(floor, Math.min(staticTarget, vendorGrounded));
};

/**
 * Hard cap on last-resort compaction rounds (daemon walker + cloud dispatch
 * chain). The vendor-grounded ratio converges fast (usually one tighter round
 * after a static undercorrect), so three rounds is ample headroom; the cap's
 * real job is to make an infinite compaction loop structurally impossible when
 * an upstream keeps rejecting. Single source of truth — both seams import this.
 */
export const MAX_LAST_RESORT_COMPACTION_ROUNDS = 3;

/**
 * A confirmed overflow always forces a real shrink. If the calibrated target
 * is already below the current local estimate, use it; otherwise tighten below
 * the current body (`localEstimate × SAFETY`, floored) so the cut is never a
 * no-op re-send of the same bytes. Shared by the daemon walker and the cloud
 * dispatch chain so the force-shrink math cannot drift between them.
 */
export const forcedCompactionTarget = (
  calibrated: number,
  localEstimate: number,
): number =>
  calibrated < localEstimate
    ? calibrated
    : Math.max(
        COMPACTION_TARGET_FLOOR,
        Math.floor(localEstimate * COMPACTION_OVERFLOW_SAFETY),
      );

/**
 * Should this hop be skipped for context? True only when a later hop
 * remains (the FINAL hop always serves — the real tokenizer must get the
 * last word, never the heuristic estimator), the model's input budget is
 * known, and the conservative routing estimate exceeds it. An unknown
 * limit always serves.
 */
export const shouldSkipHopForContext = (params: {
  readonly estimatedTokens: number;
  readonly inputTokenLimit: number | null;
  readonly finalHop: boolean;
}): boolean =>
  !params.finalHop &&
  params.inputTokenLimit !== null &&
  params.estimatedTokens >
    params.inputTokenLimit * CONTEXT_SKIP_CONFIDENCE_FACTOR;
