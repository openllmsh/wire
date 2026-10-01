import type { TAnthropicUsage } from "@openllmsh/protocol";

/**
 * Convert inclusive OpenAI-shaped prompt usage into Anthropic Messages
 * usage. OpenAI `prompt_tokens` INCLUDES cache read/creation buckets;
 * Anthropic `input_tokens` EXCLUDES them and reports those buckets in
 * their own optional fields. Returning the inclusive total as
 * `input_tokens` while also emitting cache fields double-counts.
 *
 * Mirror LiteLLM `adapters/transformation.py` usage mapping and the
 * non-streaming Messages adapter: subtract once, floor at 0, omit
 * zero/absent cache fields.
 */
export type AnthropicMessagesUsageFromInclusiveParams = {
  promptTokens: number;
  completionTokens: number;
  cachedTokens?: number;
  cacheCreationTokens?: number;
};

export type AnthropicMessagesUsageFromInclusiveReturnType = Pick<
  TAnthropicUsage,
  "input_tokens" | "output_tokens"
> &
  Partial<
    Pick<
      TAnthropicUsage,
      "cache_read_input_tokens" | "cache_creation_input_tokens"
    >
  >;

export const anthropicMessagesUsageFromInclusive = (
  params: AnthropicMessagesUsageFromInclusiveParams,
): AnthropicMessagesUsageFromInclusiveReturnType => {
  const cached = params.cachedTokens ?? 0;
  const created = params.cacheCreationTokens ?? 0;
  return {
    input_tokens: Math.max(0, params.promptTokens - cached - created),
    output_tokens: params.completionTokens,
    ...(cached > 0 ? { cache_read_input_tokens: cached } : {}),
    ...(created > 0 ? { cache_creation_input_tokens: created } : {}),
  };
};
