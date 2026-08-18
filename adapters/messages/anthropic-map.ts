import type { TAnthropicStopReason } from "@openllmsh/protocol";

/**
 * Visible answer after stripping a signed thought the model repeated
 * as `content`. Prefix / exact restatement → hide; thought then extra
 * → only the extra; anything else → the content as-is.
 */
export const visibleAnswerAfterThought = (
  contentSoFar: string,
  thought: string,
): string => {
  if (thought.length === 0) return contentSoFar;
  if (thought.startsWith(contentSoFar)) return "";
  if (contentSoFar.startsWith(thought)) {
    return contentSoFar.slice(thought.length);
  }
  return contentSoFar;
};

export const anthropicStopReasonFrom = (
  finish: string | null | undefined,
): TAnthropicStopReason | null => {
  if (finish == null) return null;
  switch (finish) {
    case "stop":
      return "end_turn";
    case "length":
      return "max_tokens";
    case "tool_calls":
    case "function_call":
      return "tool_use";
    case "content_filter":
      return "refusal";
    default:
      return null;
  }
};
