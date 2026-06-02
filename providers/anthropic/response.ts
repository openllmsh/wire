import type {
  TAnthropicContentBlock,
  TAnthropicProviderOptions,
  TAnthropicResponse,
  TAnthropicStopReason,
  TAnthropicUsage,
  TChatCompletionResponse,
  TToolCall,
  TUsage,
} from "@openllm/schema";

const finishReasonFor = (
  stop: TAnthropicStopReason | null,
): TChatCompletionResponse["choices"][number]["finish_reason"] => {
  if (stop === null) return null;
  switch (stop) {
    case "end_turn":
    case "stop_sequence":
    case "pause_turn":
      return "stop";
    case "max_tokens":
      return "length";
    case "tool_use":
      return "tool_calls";
    case "refusal":
      return "content_filter";
  }
};

const usageFor = (u: TAnthropicUsage): TUsage => {
  const cached = u.cache_read_input_tokens ?? 0;
  const created = u.cache_creation_input_tokens ?? 0;
  return {
    prompt_tokens: u.input_tokens,
    completion_tokens: u.output_tokens,
    total_tokens: u.input_tokens + u.output_tokens,
    ...(cached > 0 || created > 0
      ? {
          prompt_tokens_details: {
            cached_tokens: cached,
            cache_creation_tokens: created,
          },
        }
      : {}),
  };
};

const blocksToText = (
  blocks: ReadonlyArray<TAnthropicContentBlock>,
): string => {
  const parts: string[] = [];
  for (const b of blocks) {
    if (b.type === "text") {
      parts.push(b.text);
      continue;
    }
    if (b.type === "compaction") {
      const raw = b.content;
      if (raw === null || raw === undefined) continue;
      if (typeof raw === "string") {
        parts.push(raw);
        continue;
      }
      if (typeof raw === "number" || typeof raw === "boolean") {
        parts.push(String(raw));
        continue;
      }
      try {
        parts.push(JSON.stringify(raw));
      } catch {
        /* skip */
      }
    }
  }
  return parts.join("");
};

const blocksToToolCalls = (
  blocks: ReadonlyArray<TAnthropicContentBlock>,
): TToolCall[] => {
  const out: TToolCall[] = [];
  for (const block of blocks) {
    if (block.type === "tool_use") {
      out.push({
        id: block.id,
        type: "function",
        function: {
          name: block.name,
          arguments: JSON.stringify(block.input ?? {}),
        },
      });
    }
  }
  return out;
};

export const fromAnthropicResponse = (
  resp: TAnthropicResponse,
  options: TAnthropicProviderOptions,
): TChatCompletionResponse => {
  const text = blocksToText(resp.content);
  const toolCalls = blocksToToolCalls(resp.content);
  const created = Math.floor(Date.now() / 1000);
  return {
    id: resp.id,
    object: "chat.completion",
    created,
    model: options.providerModelId,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: text,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
        finish_reason: finishReasonFor(resp.stop_reason),
      },
    ],
    usage: usageFor(resp.usage),
  };
};
