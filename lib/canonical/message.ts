/**
 * Provider-agnostic helpers over the canonical OpenAI ChatCompletion
 * message shape. These are NOT provider quirks — they are the same
 * operation every provider needs on the way out, so they live once
 * here rather than re-inlined per provider.
 *
 * Note: the inbound `adapters/messages` response path deliberately
 * keeps its own text flattener (it joins with "" over a *response*
 * content type, not "\n" over a request message) — that divergence is
 * intentional and must not be folded in here.
 */
import type { TChatMessage } from "@openllm/schema";

/**
 * Flatten a canonical message `content` (string | content-part[]) to
 * plain text: keep only `text` parts, joined by newlines. `null` /
 * `undefined` → `""`.
 */
export const extractMessageText = (
  content: TChatMessage["content"] | null | undefined,
): string => {
  if (content == null) return "";
  if (typeof content === "string") return content;
  return content
    .filter((part) => part.type === "text")
    .map((part) => (part as { type: "text"; text: string }).text)
    .join("\n");
};

/**
 * Deserialize an OpenAI tool-call `arguments` string. Empty → `{}`;
 * valid JSON → parsed value; unparseable → the raw string verbatim
 * (Anthropic `tool_use.input` and the Messages adapter both accept a
 * raw string as a last resort).
 */
export const parseToolArguments = (raw: string): unknown => {
  if (raw.length === 0) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
};
