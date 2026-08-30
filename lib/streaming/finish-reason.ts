import type { TChatCompletionChunk } from "@openllmsh/protocol";

type TToolCallFinishReasonChoice = {
  readonly finish_reason: TChatCompletionChunk["choices"][number]["finish_reason"];
  /** Internal semantic signal for adapters that have already decoded tool use. */
  readonly hasToolCalls?: boolean;
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
 * tool calls, so accept either shape. Empty external arrays do not represent
 * a tool call; adapters with prior decoded state provide `hasToolCalls`.
 */
export const normalizeToolCallFinishReason = (
  choice: TToolCallFinishReasonChoice,
): TChatCompletionChunk["choices"][number]["finish_reason"] => {
  const hasToolCalls =
    choice.hasToolCalls === true ||
    (choice.message?.tool_calls?.length ?? 0) > 0 ||
    (choice.delta?.tool_calls?.length ?? 0) > 0;

  return hasToolCalls &&
    (choice.finish_reason === null || choice.finish_reason === "stop")
    ? "tool_calls"
    : choice.finish_reason;
};
