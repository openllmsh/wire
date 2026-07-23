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
