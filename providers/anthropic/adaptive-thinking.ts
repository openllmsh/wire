/**
 * Adaptive-thinking normalisation for Anthropic Messages requests.
 *
 * Anthropic's "adaptive thinking" (`thinking: { type: "adaptive" }` +
 * `output_config.effort`) is only honoured by Claude 4.6+ models. Every
 * other model 400s with
 *   "adaptive thinking is not supported on this model"
 * the moment any of the adaptive knobs appear in the body — even though
 * those same models support extended thinking via the older
 * `thinking: { type: "enabled", budget_tokens: N }` shape.
 *
 * This module translates the adaptive knobs into the supported shape so
 * a model like haiku-4-5 still gets thinking (we don't strip it). It is
 * the single source of truth for that translation; the pass-through
 * runner and the count_tokens handler both flow through here.
 *
 * Mirror of LiteLLM's `AnthropicConfig._map_reasoning_effort` +
 * `_translate_reasoning_effort_to_anthropic` (see
 * `ref/.litellm/litellm/litellm/llms/anthropic/chat/transformation.py`
 * and `.../experimental_pass_through/messages/transformation.py`).
 * Numeric budgets match LiteLLM's `DEFAULT_REASONING_EFFORT_*_THINKING_BUDGET`
 * constants in `litellm/constants.py` — keep them in sync if those move.
 */

/**
 * Effort → thinking budget. Same numbers LiteLLM uses; single source of
 * truth — do NOT inline these values anywhere else.
 */
export const EFFORT_TO_BUDGET_TOKENS = {
  minimal: 128,
  low: 1024,
  medium: 2048,
  high: 4096,
  xhigh: 8192,
  max: 16384,
} as const;
export type TReasoningEffort = keyof typeof EFFORT_TO_BUDGET_TOKENS;

/**
 * OpenAI-style `reasoning_effort` → Anthropic `output_config.effort`
 * map. Mirrors LiteLLM's `REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT` —
 * `minimal` folds into `low` because Anthropic's effort enum starts at
 * `low`; the rest is identity.
 */
export const REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT = {
  minimal: "low",
  low: "low",
  medium: "medium",
  high: "high",
  xhigh: "xhigh",
  max: "max",
} as const;

/** Anthropic's hard floor on `thinking.budget_tokens`. Below this, 400. */
export const ANTHROPIC_MIN_THINKING_BUDGET = 1024;

/**
 * Models that do NOT accept the adaptive-thinking knobs. Deny-list (not
 * allow-list) so an unknown future opus/sonnet works without a code
 * change; the set of non-supporters is small and stable.
 */
const NO_ADAPTIVE_THINKING = /haiku|claude-3|claude-instant/i;

export const supportsAdaptiveThinking = (model: unknown): boolean => {
  if (typeof model !== "string" || model.length === 0) return true;
  return !NO_ADAPTIVE_THINKING.test(model);
};

const readEffort = (v: unknown): TReasoningEffort | undefined => {
  if (typeof v !== "string") return undefined;
  return v in EFFORT_TO_BUDGET_TOKENS ? (v as TReasoningEffort) : undefined;
};

const isObject = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === "object";

/**
 * Does the body's `tool_choice` force tool use? Anthropic accepts
 * `thinking: { type: "adaptive" }` paired with a forced tool_choice (the
 * model decides whether to think before invoking), but REJECTS
 *   `thinking: { type: "enabled", budget_tokens: N }` + forced
 * with: "Thinking may not be enabled when tool_choice forces tool use."
 * So when we are about to downgrade adaptive → enabled on a non-
 * adaptive model (haiku, claude-3.*, claude-instant.*), we must also
 * check this and skip the enable. Forced shapes are `{type:"tool",
 * name}` and `{type:"any"}`; `{type:"auto"}` and `{type:"none"}` are
 * not forcing.
 */
const toolChoiceForcesTool = (tc: unknown): boolean => {
  if (!isObject(tc)) return false;
  return tc.type === "tool" || tc.type === "any";
};

/**
 * Filter out `clear_thinking_*` strategies from
 * `context_management.edits`. Used only on the edge path where thinking
 * has to be dropped entirely — leaving the strategy in 400s with
 *   "clear_thinking_20251015 strategy requires `thinking` to be
 *    enabled or adaptive"
 * Other strategies (`clear_tool_uses_*`, …) stay. If filtering empties
 * the edits array, the whole `context_management` field is dropped.
 */
/**
 * Translate an OpenAI-style `reasoning_effort` to the Anthropic
 * extended-thinking shape for the canonical (`/v1/chat/completions` →
 * Anthropic) path. Returns the two fields the caller should spread onto
 * the outbound `AnthropicRequest`, or `null` to mean "no thinking" (the
 * `reasoning_effort: "none"` case, or the model doesn't have room).
 *
 * Mirrors LiteLLM's `_map_reasoning_effort` +
 * `_translate_reasoning_effort_to_anthropic`:
 *   - Adaptive-supporting model: `thinking: { type: "adaptive" }` +
 *     `output_config: { effort: <mapped> }` so the model uses the
 *     stable adaptive-thinking feature and tunes its own budget.
 *   - Non-adaptive model: `thinking: { type: "enabled", budget_tokens }`
 *     using the canonical effort table, clamped to `max_tokens - 1`.
 *     `null` when the resulting budget would fall below
 *     `ANTHROPIC_MIN_THINKING_BUDGET` (max_tokens too small).
 */
export type TReasoningEffortInput = TReasoningEffort | "none";
export type TAnthropicMappedEffort =
  (typeof REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT)[TReasoningEffort];
export type TThinkingFromEffort = {
  readonly thinking:
    | { readonly type: "enabled"; readonly budget_tokens: number }
    | { readonly type: "adaptive" };
  readonly output_config?: { readonly effort: TAnthropicMappedEffort };
};

export const mapReasoningEffortToAnthropic = (
  effort: TReasoningEffortInput,
  model: string,
  maxTokens: number | undefined,
): TThinkingFromEffort | null => {
  if (effort === "none") return null;
  if (supportsAdaptiveThinking(model)) {
    return {
      thinking: { type: "adaptive" },
      output_config: {
        effort: REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT[effort],
      },
    };
  }
  const targetBudget = EFFORT_TO_BUDGET_TOKENS[effort];
  const ceiling =
    maxTokens === undefined ? Number.POSITIVE_INFINITY : maxTokens - 1;
  const budget = Math.min(targetBudget, ceiling);
  if (budget < ANTHROPIC_MIN_THINKING_BUDGET) return null;
  return { thinking: { type: "enabled", budget_tokens: budget } };
};

const filterClearThinking = (cm: unknown): unknown | undefined => {
  if (!isObject(cm)) return cm;
  const edits = cm.edits;
  if (!Array.isArray(edits)) return cm;
  const filtered = edits.filter((e) => {
    if (!isObject(e)) return true;
    const t = e.type;
    return typeof t !== "string" || !t.startsWith("clear_thinking");
  });
  if (filtered.length === edits.length) return cm;
  if (filtered.length === 0) return undefined;
  return { ...cm, edits: filtered };
};

/**
 * Normalise the adaptive-thinking knobs on an Anthropic Messages
 * request body so it works against any chat model.
 *
 * On models that support adaptive thinking natively → no-op.
 * On every other model:
 *   - Replace `thinking: { type: "adaptive" }` (or a bare
 *     `reasoning_effort` / `output_config.effort` / top-level `effort`)
 *     with `thinking: { type: "enabled", budget_tokens: N }`, where N
 *     comes from the canonical effort table (default `medium` when
 *     adaptive was requested without an explicit effort level). The
 *     budget is clamped to `max_tokens - 1` so Anthropic's
 *     `budget_tokens < max_tokens` invariant holds.
 *   - Drop `output_config`, `reasoning_effort`, top-level `effort` —
 *     they are adaptive-only and have no analogue under explicit
 *     `budget_tokens`.
 *   - `thinking: { type: "enabled" }` and unknown other shapes pass
 *     through unchanged.
 *   - If `max_tokens` is too small to fit `ANTHROPIC_MIN_THINKING_BUDGET`,
 *     thinking is dropped and any `clear_thinking_*` strategy in
 *     `context_management.edits` is filtered out (else Anthropic
 *     rejects the orphaned strategy).
 *
 * Returns the input untouched when no rewrite is needed, and a shallow
 * copy otherwise. Caller-provided `tools`, `system`, `metadata`,
 * `cache_control`, beta headers, etc. are never inspected.
 */
export const normaliseAdaptiveThinking = (body: unknown): unknown => {
  if (!isObject(body)) return body;
  if (supportsAdaptiveThinking(body.model)) return body;

  const {
    output_config,
    reasoning_effort,
    effort,
    thinking,
    context_management,
    ...rest
  } = body;

  const thinkingIsAdaptive = isObject(thinking) && thinking.type === "adaptive";
  const wantsThinking =
    thinkingIsAdaptive ||
    typeof reasoning_effort === "string" ||
    typeof effort === "string" ||
    (isObject(output_config) && typeof output_config.effort === "string");

  // No adaptive knobs to translate — nothing to rewrite.
  if (!wantsThinking) return body;

  const effortLevel: TReasoningEffort =
    (isObject(output_config) ? readEffort(output_config.effort) : undefined) ??
    readEffort(reasoning_effort) ??
    readEffort(effort) ??
    "medium";

  const targetBudget = EFFORT_TO_BUDGET_TOKENS[effortLevel];
  const maxTokens =
    typeof body.max_tokens === "number" &&
    Number.isFinite(body.max_tokens) &&
    body.max_tokens > 0
      ? Math.floor(body.max_tokens)
      : undefined;
  const ceiling =
    maxTokens === undefined ? Number.POSITIVE_INFINITY : maxTokens - 1;
  const budget = Math.min(targetBudget, ceiling);

  // Anthropic rejects `thinking: enabled` (the downgrade output) when
  // tool_choice forces tool use. Drop thinking entirely in that case,
  // matching the max_tokens-too-small edge path (filter orphaned
  // `clear_thinking_*` strategies; the forced tool still executes,
  // the model just doesn't get a thinking preamble). Adaptive remains
  // legal on adaptive-supporting models, which short-circuit above.
  const forcedTool = toolChoiceForcesTool(body.tool_choice);

  const out: Record<string, unknown> = { ...rest };
  if (budget >= ANTHROPIC_MIN_THINKING_BUDGET && !forcedTool) {
    out.thinking = { type: "enabled", budget_tokens: budget };
    if (context_management !== undefined) {
      out.context_management = context_management;
    }
  } else if (context_management !== undefined) {
    const filtered = filterClearThinking(context_management);
    if (filtered !== undefined) out.context_management = filtered;
  }
  return out;
};
