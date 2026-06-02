/**
 * Cross-provider reasoning round-trip codec.
 *
 * THE BUG THIS FIXES: when an Anthropic-format client (Claude Code) is
 * served by a *reasoning* upstream on the OpenAI Responses API
 * (ChatGPT/Codex, `store: false` + `include: ["reasoning.encrypted_content"]`),
 * the model emits a `reasoning` output item whose `encrypted_content`
 * encodes its chain-of-thought for the turn. The Responses API REQUIRES
 * that item to be echoed back, in order, immediately before its
 * `function_call`, on the next request. If it is dropped the model
 * restarts reasoning every turn, re-derives the identical tool call,
 * and loops forever (the "70k tokens, never exits" symptom).
 *
 * Anthropic's wire format has exactly one channel that a client
 * preserves and replays verbatim across a tool turn: the `thinking`
 * block's opaque `signature`. So we smuggle the litellm-shaped
 * `reasoning` item(s) through that field — encoded here on the way out,
 * decoded here on the way back in. Both directions go through this one
 * module so the encode/decode contract can never drift.
 *
 * Mirrors litellm `completion_extras/litellm_responses_transformation/
 * transformation.py`: `_build_reasoning_item` (~69-96) and
 * `_reasoning_item_to_response_input` (~99-111).
 */

// runtime-only: litellm `ChatCompletionReasoningItem` shape.
export type TReasoningSummaryPart = {
  readonly type: string;
  readonly text: string;
};
export type TReasoningItem = {
  readonly id: string;
  readonly type: "reasoning";
  readonly encrypted_content: string | null;
  readonly summary: ReadonlyArray<TReasoningSummaryPart>;
};

// runtime-only: a `reasoning` item in the Responses API `input` array.
export type TReasoningResponsesInput = {
  readonly type: "reasoning";
  readonly id: string;
  readonly summary: ReadonlyArray<TReasoningSummaryPart>;
  readonly encrypted_content?: string;
};

/**
 * Tag prefix so {@link decodeReasoningSignature} only ever consumes a
 * signature WE produced. A genuine Anthropic `signature` (native
 * pass-through path) lacks this prefix and is left untouched.
 */
const SIGNATURE_PREFIX = "openllm-rs1:";

const toBase64 = (s: string): string => {
  if (typeof Buffer !== "undefined") {
    return Buffer.from(s, "utf8").toString("base64");
  }
  return btoa(unescape(encodeURIComponent(s)));
};

const fromBase64 = (s: string): string => {
  if (typeof Buffer !== "undefined") {
    return Buffer.from(s, "base64").toString("utf8");
  }
  return decodeURIComponent(escape(atob(s)));
};

const normalizeSummary = (raw: unknown): TReasoningSummaryPart[] => {
  if (!Array.isArray(raw)) return [];
  const out: TReasoningSummaryPart[] = [];
  for (const s of raw) {
    if (s !== null && typeof s === "object" && !Array.isArray(s)) {
      const o = s as Record<string, unknown>;
      out.push({
        type: typeof o.type === "string" ? o.type : "summary_text",
        text: typeof o.text === "string" ? o.text : "",
      });
    }
  }
  return out;
};

/**
 * Build a {@link TReasoningItem} from raw Responses-API output. Mirrors
 * litellm `_build_reasoning_item`.
 */
export const buildReasoningItem = (
  id: string,
  encryptedContent: string | null | undefined,
  summaryRaw: unknown,
): TReasoningItem => ({
  id,
  type: "reasoning",
  encrypted_content:
    typeof encryptedContent === "string" ? encryptedContent : null,
  summary: normalizeSummary(summaryRaw),
});

/**
 * Coerce the canonical `reasoning_items` carrier (`S.Array(S.Unknown)`
 * on the chat schema) into typed reasoning items. The single
 * normalizer used everywhere a reasoning item leaves the canonical
 * shape (Anthropic signature emit + Responses `input` build).
 */
export const reasoningItemsFromUnknown = (
  items: ReadonlyArray<unknown> | null | undefined,
): TReasoningItem[] => {
  if (items === undefined || items === null || !Array.isArray(items)) {
    return [];
  }
  const out: TReasoningItem[] = [];
  for (const item of items) {
    if (item === null || typeof item !== "object") continue;
    const rec = item as Record<string, unknown>;
    if (rec.type !== "reasoning") continue;
    out.push(
      buildReasoningItem(
        typeof rec.id === "string" ? rec.id : "",
        typeof rec.encrypted_content === "string"
          ? rec.encrypted_content
          : null,
        rec.summary,
      ),
    );
  }
  return out;
};

/**
 * Encode reasoning items into an Anthropic `thinking.signature`. Returns
 * `null` when there is nothing worth round-tripping (no item carries
 * `encrypted_content`, so the upstream cannot resume from it anyway).
 */
export const encodeReasoningSignature = (
  items: ReadonlyArray<TReasoningItem>,
): string | null => {
  const withState = items.filter(
    (i) =>
      typeof i.encrypted_content === "string" && i.encrypted_content.length > 0,
  );
  if (withState.length === 0) return null;
  return SIGNATURE_PREFIX + toBase64(JSON.stringify(withState));
};

/**
 * Decode a `thinking.signature` produced by
 * {@link encodeReasoningSignature}. Returns `null` for any signature we
 * did not produce (genuine Anthropic signatures, malformed input) so
 * callers fall back to their existing `reasoning_content` text path.
 */
export const decodeReasoningSignature = (
  signature: string | null | undefined,
): TReasoningItem[] | null => {
  if (
    typeof signature !== "string" ||
    !signature.startsWith(SIGNATURE_PREFIX)
  ) {
    return null;
  }
  try {
    const json = fromBase64(signature.slice(SIGNATURE_PREFIX.length));
    const parsed: unknown = JSON.parse(json);
    if (!Array.isArray(parsed)) return null;
    const items = reasoningItemsFromUnknown(parsed);
    return items.length > 0 ? items : null;
  } catch {
    return null;
  }
};

/**
 * Convert a stored reasoning item back to a Responses API `input` item.
 * Mirrors litellm `_reasoning_item_to_response_input`: `summary` is
 * always required (even empty); `encrypted_content` is only sent when
 * present.
 */
export const reasoningItemToResponsesInput = (
  item: TReasoningItem,
): TReasoningResponsesInput => ({
  type: "reasoning",
  id: item.id.length > 0 ? item.id : `rs_${crypto.randomUUID()}`,
  summary: item.summary,
  ...(typeof item.encrypted_content === "string" &&
  item.encrypted_content.length > 0
    ? { encrypted_content: item.encrypted_content }
    : {}),
});
