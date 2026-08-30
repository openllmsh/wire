import type { TChatCompletionChunk } from "@openllmsh/protocol";

type TToolCallFinishReasonChoice = {
  readonly finish_reason: TChatCompletionChunk["choices"][number]["finish_reason"];
  readonly message?: {
    readonly tool_calls?: ReadonlyArray<unknown>;
  };
  readonly delta?: {
    readonly tool_calls?: ReadonlyArray<unknown>;
  };
};

/**
 * Correct providers that emit tool calls but terminate their response as a
 * normal stop. Both Chat Completions responses and stream chunks can carry
 * tool calls, so accept either shape.
 */
export const normalizeToolCallFinishReason = (
  choice: TToolCallFinishReasonChoice,
): TChatCompletionChunk["choices"][number]["finish_reason"] => {
  const hasToolCalls =
    choice.message?.tool_calls !== undefined ||
    choice.delta?.tool_calls !== undefined;

  return hasToolCalls &&
    (choice.finish_reason === null || choice.finish_reason === "stop")
    ? "tool_calls"
    : choice.finish_reason;
};
