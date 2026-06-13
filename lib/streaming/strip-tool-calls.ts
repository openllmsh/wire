import type {
  TChatCompletionChunk,
  TChatCompletionResponse,
} from "@quantidexyz/openllmp";

/**
 * Claude Code `/compact` uses `Xm()` which requires usable summary text.
 * cf. https://github.com/anthropics/claude-code/issues/11095
 *
 * Some OpenAI-compatible providers still stream `tool_calls` when the request
 * omits `tools` (conversation history is tool-heavy). That yields Anthropic
 * `tool_use` + `stop_reason: tool_use` and breaks compaction extraction even
 * with plenty of `text` deltas.
 */
export const coerceOpenAiFinishReasonAfterStrippingToolCalls = (
  fr: string | null | undefined,
): string | null | undefined => {
  if (fr === "tool_calls" || fr === "function_call") return "stop";
  return fr;
};

export const stripToolCallDeltasFromChunk = (
  chunk: TChatCompletionChunk,
): TChatCompletionChunk => ({
  ...chunk,
  choices: chunk.choices.map((choice, idx) => {
    if (idx !== 0) return choice;
    const nextFr = coerceOpenAiFinishReasonAfterStrippingToolCalls(
      choice.finish_reason,
    ) as TChatCompletionChunk["choices"][number]["finish_reason"] | undefined;
    const delta = choice.delta;
    const toolCalls = delta.tool_calls;
    if (
      toolCalls === undefined ||
      toolCalls === null ||
      toolCalls.length === 0
    ) {
      if (nextFr === choice.finish_reason) return choice;
      return { ...choice, finish_reason: nextFr };
    }
    const { tool_calls: _omit, ...restDelta } = delta;
    return {
      ...choice,
      finish_reason: nextFr,
      delta: restDelta,
    };
  }),
});

export const stripToolCallDeltasFromChunkStream = (
  source: ReadableStream<TChatCompletionChunk>,
): ReadableStream<TChatCompletionChunk> => {
  const reader = source.getReader();
  return new ReadableStream<TChatCompletionChunk>({
    async pull(controller) {
      const { value, done } = await reader.read();
      if (done) {
        controller.close();
        return;
      }
      controller.enqueue(stripToolCallDeltasFromChunk(value));
    },
    cancel(reason) {
      reader.cancel(reason).catch(() => {});
    },
  });
};

export const stripAssistantToolCallsFromChatCompletion = (
  resp: TChatCompletionResponse,
): TChatCompletionResponse => {
  const choice = resp.choices[0];
  if (choice === undefined) return resp;
  const nextFr = coerceOpenAiFinishReasonAfterStrippingToolCalls(
    choice.finish_reason,
  ) as TChatCompletionResponse["choices"][number]["finish_reason"];
  const tc = choice.message.tool_calls;
  if (
    (tc === undefined || tc.length === 0) &&
    nextFr === choice.finish_reason
  ) {
    return resp;
  }
  const { tool_calls: _t, ...restMessage } = choice.message;
  return {
    ...resp,
    choices: [
      {
        ...choice,
        finish_reason: nextFr,
        message: restMessage,
      },
    ],
  };
};
