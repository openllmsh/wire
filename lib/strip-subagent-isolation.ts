import type {
  TChatCompletionChunk,
  TChatCompletionResponse,
} from "@openllmsh/protocol";
import { accumulateChunksToResponse } from "./streaming/accumulate";

const SUBAGENT_TOOL_NAMES = new Set<string>(["Agent", "Task"]);

// The shared streaming accumulator reconstructs only choice 0, so the
// streaming strip scopes its target discovery and edits to that choice.
// A secondary choice (OpenAI `n > 1`) reuses per-choice tool-call
// indices, so touching it by a choice-agnostic index would corrupt an
// unrelated call. The non-streaming path matches by tool NAME per choice
// and is safe across all choices.
const TARGET_CHOICE_INDEX = 0;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const stripIsolationArguments = (argumentsJson: string): string => {
  try {
    const parsed: unknown = JSON.parse(argumentsJson);
    if (
      !isRecord(parsed) ||
      (!Object.hasOwn(parsed, "isolation") && !Object.hasOwn(parsed, "cwd"))
    ) {
      return argumentsJson;
    }
    delete parsed.isolation;
    delete parsed.cwd;
    return JSON.stringify(parsed);
  } catch {
    return argumentsJson;
  }
};

/**
 * Drops Claude Code's model-emitted worktree controls from Task/Agent calls.
 * Invalid JSON and calls without either field are returned byte-for-byte.
 */
export const stripIsolationFromResponse = (
  response: TChatCompletionResponse,
): TChatCompletionResponse => {
  let changed = false;
  const choices = response.choices.map((choice) => {
    const toolCalls = choice.message.tool_calls;
    if (
      toolCalls === undefined ||
      toolCalls === null ||
      toolCalls.length === 0
    ) {
      return choice;
    }
    let choiceChanged = false;
    const nextToolCalls = toolCalls.map((toolCall) => {
      if (!SUBAGENT_TOOL_NAMES.has(toolCall.function.name)) return toolCall;
      const argumentsJson = stripIsolationArguments(
        toolCall.function.arguments,
      );
      if (argumentsJson === toolCall.function.arguments) return toolCall;
      choiceChanged = true;
      changed = true;
      return {
        ...toolCall,
        function: { ...toolCall.function, arguments: argumentsJson },
      };
    });
    return choiceChanged
      ? {
          ...choice,
          message: { ...choice.message, tool_calls: nextToolCalls },
        }
      : choice;
  });
  return changed ? { ...response, choices } : response;
};

const chunksFrom = (
  chunks: ReadonlyArray<TChatCompletionChunk>,
): ReadableStream<TChatCompletionChunk> =>
  new ReadableStream<TChatCompletionChunk>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });

type TTargetToolCall = {
  readonly index: number;
  readonly arguments: string;
};

const targetToolCallsFrom = async (
  chunks: ReadonlyArray<TChatCompletionChunk>,
): Promise<ReadonlyArray<TTargetToolCall>> => {
  const response = await accumulateChunksToResponse(
    chunksFrom(chunks),
    chunks[0]?.model ?? "unknown",
  );
  const toolCalls = response.choices[0]?.message.tool_calls ?? [];
  // The shared accumulator reconstructs only choice 0, so targets are
  // scoped to choice 0. Tool-call `index` is per-choice in the OpenAI
  // wire, so a secondary choice (`n > 1`) can reuse index 0 for an
  // unrelated call — gathering indices across choices would strip those
  // too. Collect streamed indices from choice 0 alone.
  const streamedIndices = new Set<number>();
  for (const chunk of chunks) {
    for (const choice of chunk.choices) {
      if (choice.index !== TARGET_CHOICE_INDEX) continue;
      for (const toolCall of choice.delta.tool_calls ?? []) {
        streamedIndices.add(toolCall.index);
      }
    }
  }
  const orderedIndices = [...streamedIndices].sort((a, b) => a - b);
  return toolCalls.flatMap((toolCall, index) => {
    if (!SUBAGENT_TOOL_NAMES.has(toolCall.function.name)) return [];
    const argumentsJson = stripIsolationArguments(toolCall.function.arguments);
    // No streamed choice-0 index for this accumulator position means the
    // accumulator folded in a secondary choice (`n > 1`). Guessing an index
    // could collide with a real choice-0 call and re-emit the wrong payload,
    // so leave the tail byte-for-byte — the documented fallback for every
    // other unhandled shape.
    const streamedIndex = orderedIndices[index];
    if (streamedIndex === undefined) return [];
    return argumentsJson === toolCall.function.arguments
      ? []
      : [{ index: streamedIndex, arguments: argumentsJson }];
  });
};

const withoutTargetArguments = (
  chunk: TChatCompletionChunk,
  targetIndices: ReadonlySet<number>,
): TChatCompletionChunk => {
  let chunkChanged = false;
  const choices = chunk.choices.map((choice) => {
    // Targets live in choice 0 only (see `targetToolCallsFrom`); leave a
    // secondary choice's tool calls — which may reuse the same index —
    // byte-for-byte.
    if (choice.index !== TARGET_CHOICE_INDEX) return choice;
    const toolCalls = choice.delta.tool_calls;
    if (toolCalls === undefined || toolCalls === null || toolCalls.length === 0)
      return choice;
    let choiceChanged = false;
    const nextToolCalls = toolCalls.map((toolCall) => {
      if (
        !targetIndices.has(toolCall.index) ||
        toolCall.function?.arguments === undefined
      ) {
        return toolCall;
      }
      choiceChanged = true;
      chunkChanged = true;
      const { arguments: _arguments, ...functionRest } = toolCall.function;
      return { ...toolCall, function: functionRest };
    });
    return choiceChanged
      ? { ...choice, delta: { ...choice.delta, tool_calls: nextToolCalls } }
      : choice;
  });
  return chunkChanged ? { ...chunk, choices } : chunk;
};

const replacementChunk = (
  template: TChatCompletionChunk,
  targetToolCalls: ReadonlyArray<TTargetToolCall>,
): TChatCompletionChunk => ({
  id: template.id,
  object: "chat.completion.chunk",
  created: template.created,
  model: template.model,
  choices: [
    {
      index: 0,
      delta: {
        tool_calls: targetToolCalls.map((toolCall) => ({
          index: toolCall.index,
          function: { arguments: toolCall.arguments },
        })),
      },
    },
  ],
});

const endsTurn = (chunk: TChatCompletionChunk): boolean =>
  chunk.choices.some((choice) => choice.finish_reason != null);

const hasToolCallDelta = (chunk: TChatCompletionChunk): boolean =>
  chunk.choices.some((choice) => {
    const toolCalls = choice.delta.tool_calls;
    return (
      toolCalls !== undefined && toolCalls !== null && toolCalls.length > 0
    );
  });

/**
 * Reassemble only the tool-call tail. Buffering starts at `first` (the
 * first chunk carrying tool-call deltas); everything before it has
 * already been forwarded. Drains the rest of the source, reconstructs
 * the Task/Agent arguments with the shared wire accumulator, and — when
 * a target is found — emits one clean replacement delta before the
 * terminal chunk. No target (or malformed JSON) → the tail is returned
 * byte-for-byte.
 */
const rebuildToolCallTail = async (
  first: TChatCompletionChunk,
  reader: ReadableStreamDefaultReader<TChatCompletionChunk>,
): Promise<TChatCompletionChunk[]> => {
  const buffered: TChatCompletionChunk[] = [first];
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffered.push(value);
  }
  const targetToolCalls = await targetToolCallsFrom(buffered);
  const template = buffered[0];
  if (targetToolCalls.length === 0 || template === undefined) {
    return buffered;
  }
  const targetIndices = new Set(
    targetToolCalls.map((toolCall) => toolCall.index),
  );
  const replacement = replacementChunk(template, targetToolCalls);
  let emittedReplacement = false;
  const output: TChatCompletionChunk[] = [];
  for (const nextChunk of buffered) {
    if (!emittedReplacement && endsTurn(nextChunk)) {
      output.push(replacement);
      emittedReplacement = true;
    }
    output.push(withoutTargetArguments(nextChunk, targetIndices));
  }
  if (!emittedReplacement) output.push(replacement);
  return output;
};

/**
 * Streams through chunks that precede the first tool call untouched — a
 * leading role/content chunk reaches the client immediately, not after
 * the whole turn is buffered. From the first tool-call delta onward the
 * tail is buffered and reassembled (see `rebuildToolCallTail`): the two
 * controls are removed from complete Task/Agent JSON and one clean
 * replacement delta is emitted before the terminal chunk. Malformed
 * arguments and non-target tool calls remain byte-for-byte.
 */
export const stripIsolationFromChunks = (
  source: ReadableStream<TChatCompletionChunk>,
): ReadableStream<TChatCompletionChunk> => {
  const reader = source.getReader();
  let tail: TChatCompletionChunk[] | null = null;

  return new ReadableStream<TChatCompletionChunk>({
    async pull(controller): Promise<void> {
      // Draining the reconstructed tool-call tail.
      if (tail !== null) {
        const next = tail.shift();
        if (next === undefined) controller.close();
        else controller.enqueue(next);
        return;
      }
      // Prelude: forward one chunk per pull until a tool call appears.
      const { value, done } = await reader.read();
      if (done) {
        controller.close();
        return;
      }
      if (!hasToolCallDelta(value)) {
        controller.enqueue(value);
        return;
      }
      // First tool-call chunk: buffer + reconstruct from here on.
      tail = await rebuildToolCallTail(value, reader);
      const next = tail.shift();
      if (next === undefined) controller.close();
      else controller.enqueue(next);
    },
    cancel(reason) {
      reader.cancel(reason).catch(() => {});
    },
  });
};
