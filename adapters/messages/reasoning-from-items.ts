/**
 * Flattens LiteLLM-shaped `reasoning_items` summary text for Anthropic `/compact`.
 *
 * Ref:
 * `ref/.litellm/litellm/litellm/completion_extras/litellm_responses_transformation/transformation.py`
 * — `_build_reasoning_item` (dict branch ~78–82) builds each `summary` row as
 * `{ "type": s.get("type", "summary_text"), "text": s.get("text", "") }`.
 * `translate_responses_chunk_to_openai_stream` attaches those items on
 * `response.completed` as `Delta(reasoning_items=...)` (~1321–1356).
 *
 * LiteLLM keeps structured `reasoning_items`; OpenLLM joins non-empty `text`
 * fields with newlines so clients that require user-visible `text` still work.
 */
export const plainTextFromReasoningItems = (
  items: ReadonlyArray<unknown> | null | undefined,
): string => {
  if (items === undefined || items === null || !Array.isArray(items)) {
    return "";
  }
  const parts: string[] = [];
  for (const item of items) {
    if (item === null || typeof item !== "object") continue;
    const rec = item as Record<string, unknown>;
    if (rec.type !== "reasoning") continue;
    const summaryRaw = rec.summary;
    if (!Array.isArray(summaryRaw)) continue;
    for (const s of summaryRaw) {
      if (typeof s === "object" && s !== null && !Array.isArray(s)) {
        const o = s as Record<string, unknown>;
        const text = typeof o.text === "string" ? o.text : "";
        if (text.length > 0) {
          parts.push(text);
        }
      }
    }
  }
  return parts.join("\n");
};
