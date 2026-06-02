/**
 * Rough input-token estimator over a request body, format-agnostic.
 *
 * Walks the JSON value summing string lengths (everything else
 * contributes zero) and divides by 4. Works for both canonical
 * OpenAI `ChatCompletionRequest` and raw Anthropic Messages bodies
 * because both encode all user-visible text inside string values.
 *
 * Used pre-fetch by the chain orchestrator to skip a hop whose
 * provider model can't fit the request — preventing the otherwise
 * inevitable 400 + waste of the chain entry. The estimate is
 * intentionally conservative: in borderline cases we'd rather burn
 * an extra chain step than ship an oversized request that 400s.
 */

const stringCharsFromAny = (v: unknown): number => {
  if (typeof v === "string") return v.length;
  if (Array.isArray(v)) {
    let n = 0;
    for (const x of v) n += stringCharsFromAny(x);
    return n;
  }
  if (v !== null && typeof v === "object") {
    let n = 0;
    for (const k of Object.keys(v as Record<string, unknown>)) {
      n += stringCharsFromAny((v as Record<string, unknown>)[k]);
    }
    return n;
  }
  return 0;
};

export const estimateBodyTokens = (body: unknown): number =>
  Math.ceil(stringCharsFromAny(body) / 4);
