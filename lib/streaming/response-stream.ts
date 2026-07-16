import type {
  TChatCompletionChunk,
  TChatCompletionResponse,
} from "@openllmsh/protocol";

/**
 * Turn a finished canonical response into a one-shot chunk stream so an
 * accumulated (non-streaming) result can still be emitted to a streaming
 * client. The agentic web_search rounds can't stream — each round is read
 * in full to detect follow-up tool calls — so once that path is taken the
 * final answer arrives as a single chunk.
 *
 * Shared verbatim by the cloud orchestrator (`@openllm/core`
 * `applyToolHandlers`) and the coreless daemon walker so both agentic
 * paths emit the final answer identically. The whole assistant turn is
 * carried — content, `tool_calls`, and `reasoning_content` — so a streaming
 * client never silently loses a client-tool call or reasoning trace just
 * because a search ran earlier in the turn.
 */
export const responseToChunkStream = (
  resp: TChatCompletionResponse,
): ReadableStream<TChatCompletionChunk> => {
  const choice = resp.choices[0];
  const message = choice?.message;
  const toolCalls = message?.tool_calls;
  const chunk: TChatCompletionChunk = {
    id: resp.id,
    object: "chat.completion.chunk",
    created: resp.created,
    model: resp.model,
    choices: [
      {
        index: 0,
        delta: {
          role: "assistant",
          content: typeof message?.content === "string" ? message.content : "",
          ...(toolCalls !== undefined &&
          toolCalls !== null &&
          toolCalls.length > 0
            ? {
                tool_calls: toolCalls.map((tc, index) => ({
                  index,
                  id: tc.id,
                  type: "function" as const,
                  function: {
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                  },
                })),
              }
            : {}),
          ...(typeof message?.reasoning_content === "string"
            ? { reasoning_content: message.reasoning_content }
            : {}),
          ...(message?.server_search_calls != null &&
          message.server_search_calls.length > 0
            ? { server_search_calls: message.server_search_calls }
            : {}),
        },
        finish_reason: choice?.finish_reason ?? "stop",
      },
    ],
    ...(resp.usage !== undefined ? { usage: resp.usage } : {}),
  };
  return new ReadableStream<TChatCompletionChunk>({
    start(c) {
      c.enqueue(chunk);
      c.close();
    },
  });
};
