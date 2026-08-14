/**
 * Raise-only output-token backfill — the shared rule the cloud dispatch
 * chain and the coreless daemon walker both apply so an alias / binding
 * / pin that resolves to a concrete model gets that model's catalog
 * ceiling instead of the client's conservative default for the
 * unrecognised alias id.
 *
 * Lives in `@openllmsh/wire` (not `@openllm/core`) so the daemon can
 * import it. Same gating as the historical `dispatch-chain` helper:
 * concrete-id requests, unknown ceilings, and already-generous client
 * values all no-op.
 */

export type TMaxTokensBackfillArgs = {
  readonly requestedModel: string;
  readonly getOutputTokenLimit: (modelId: string) => number | null;
};

/**
 * When a hop's catalog id differs from the inbound requested id AND the
 * resolved model's native output ceiling is known AND it exceeds the
 * client's value, return that ceiling. Otherwise `null` (leave the
 * client's cap untouched).
 */
export const backfilledMaxTokens = (
  modelId: string,
  clientMaxTokens: number | undefined,
  args: TMaxTokensBackfillArgs,
): number | null => {
  if (modelId === args.requestedModel) return null;
  const ceiling = args.getOutputTokenLimit(modelId);
  if (ceiling === null) return null;
  if (clientMaxTokens !== undefined && clientMaxTokens >= ceiling) return null;
  return ceiling;
};

const finiteNumber = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) ? value : undefined;

/**
 * Effective client output cap on an already-built upstream body.
 * The smallest present sibling wins: a leftover 32k `max_tokens` must
 * still trigger raise-only backfill even when `max_completion_tokens`
 * is already at the ceiling (first-present would miss it and leave
 * the smaller field to shadow upstream).
 */
export const clientOutputCapFromBody = (body: unknown): number | undefined => {
  if (body === null || typeof body !== "object") return undefined;
  const rec = body as Record<string, unknown>;
  const caps: number[] = [];
  for (const key of [
    "max_output_tokens",
    "max_completion_tokens",
    "max_tokens",
  ] as const) {
    const value = finiteNumber(rec[key]);
    if (value !== undefined) caps.push(value);
  }
  if (caps.length === 0) return undefined;
  return Math.min(...caps);
};

const inboundRequestedModel = (rawBody: unknown): string | null => {
  if (rawBody === null || typeof rawBody !== "object") return null;
  const model = (rawBody as { readonly model?: unknown }).model;
  return typeof model === "string" && model.length > 0 ? model : null;
};

/**
 * Raise whichever output-cap fields the built body already carries so a
 * leftover smaller sibling cannot shadow the backfill. Codex Responses
 * bodies omit the field on purpose (`max_output_tokens` 400s) — do not
 * invent it. When no cap field is present on an openai / anthropic /
 * non-Codex Responses body, write the field that wire honors.
 */
export const raiseOutputCapFields = (
  body: unknown,
  raised: number,
  wire: "anthropic" | "chatgpt" | "openai",
  // `false` = non-Codex Responses (Grok). Undefined / true = Codex.
  codexInstructions?: boolean,
): unknown => {
  if (body === null || typeof body !== "object" || Array.isArray(body)) {
    return body;
  }
  const rec = body as Record<string, unknown>;
  const hasOutput = finiteNumber(rec.max_output_tokens) !== undefined;
  const hasCompletion = finiteNumber(rec.max_completion_tokens) !== undefined;
  const hasTokens = finiteNumber(rec.max_tokens) !== undefined;
  if (!hasOutput && !hasCompletion && !hasTokens) {
    if (wire === "chatgpt" && codexInstructions !== false) return body;
    if (wire === "chatgpt") return { ...rec, max_output_tokens: raised };
    if (wire === "anthropic") return { ...rec, max_tokens: raised };
    return { ...rec, max_completion_tokens: raised };
  }
  const raiseField = (existing: unknown): number => {
    const current = finiteNumber(existing);
    return current === undefined ? raised : Math.max(current, raised);
  };
  // Per-field raise-only: a sibling already above the ceiling stays;
  // a leftover smaller sibling is lifted. Writing every field to
  // `raised` would lower a generous cap.
  return {
    ...rec,
    ...(hasOutput
      ? { max_output_tokens: raiseField(rec.max_output_tokens) }
      : {}),
    ...(hasCompletion
      ? { max_completion_tokens: raiseField(rec.max_completion_tokens) }
      : {}),
    ...(hasTokens ? { max_tokens: raiseField(rec.max_tokens) } : {}),
  };
};

/**
 * Apply raise-only backfill to a built upstream body. `null` get-limit
 * or a no-op rule returns `body` unchanged.
 */
export const applyOutputTokenBackfill = (
  body: unknown,
  args: {
    readonly hopModelId: string;
    readonly rawBody: unknown;
    readonly getOutputTokenLimit: (modelId: string) => number | null;
    readonly wire: "anthropic" | "chatgpt" | "openai";
    readonly codexInstructions?: boolean;
  },
): unknown => {
  const requestedModel = inboundRequestedModel(args.rawBody) ?? args.hopModelId;
  const raised = backfilledMaxTokens(
    args.hopModelId,
    clientOutputCapFromBody(body),
    {
      requestedModel,
      getOutputTokenLimit: args.getOutputTokenLimit,
    },
  );
  return raised === null
    ? body
    : raiseOutputCapFields(body, raised, args.wire, args.codexInstructions);
};
