import type {
  TChatCompletionResponse,
  TChatMessage,
  TToolCall,
} from "@openllm/schema";

/**
 * Pure helpers for the web-search tool — no DB, no fetch. The
 * gateway-side `runConfiguredSearch` (in
 * `packages/api/handlers/web-search.ts`) is still the credential
 * loader + provider invoker; these are the request-shape and
 * tool-call inspection primitives both the gateway intercept path and
 * `webSearchTool.handle` rely on.
 */

/**
 * Upper bound on agentic web_search rounds per hop — bounds a misbehaving
 * model that keeps requesting searches without ever answering. Shared by
 * the cloud orchestrator (`@openllm/core` `applyToolHandlers`) and the
 * coreless daemon walker so both paths agent to the same depth and fail
 * the same way on exhaustion.
 */
export const MAX_WEB_SEARCH_ROUNDS = 4;

export const WEB_SEARCH_NAMES = new Set([
  "web_search",
  "WebSearch",
  "litellm_web_search",
]);

export const functionNameUsesWebSearch = (name: string): boolean =>
  WEB_SEARCH_NAMES.has(name) || name.startsWith("web_search_");

export const toolCallUsesWebSearch = (call: TToolCall): boolean =>
  functionNameUsesWebSearch(call.function.name);

const parseToolArguments = (raw: string): unknown => {
  if (raw.length === 0) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return {};
  }
};

export const extractQueryFromToolCall = (call: TToolCall): string => {
  const parsed = parseToolArguments(call.function.arguments);
  if (parsed !== null && typeof parsed === "object" && "query" in parsed) {
    const query = (parsed as { readonly query?: unknown }).query;
    return typeof query === "string" ? query.trim() : "";
  }
  return "";
};

/**
 * Build a synthetic `assistant` message echoing the model's
 * tool-call request, used to anchor the follow-up turn that carries
 * the tool results. Returns `null` when there are no calls to echo —
 * the caller treats that as "no web_search interception happened".
 *
 * Carries `reasoning_content` forward when the model emitted any: Kimi
 * (`kimi-for-coding`) hard-rejects the follow-up turn with
 *   "thinking is enabled but reasoning_content is missing in assistant
 *    tool call message at index N"
 * because its protocol requires every assistant tool-call message to
 * be paired with the reasoning that produced it. Dropping the field
 * before the follow-up makes the second hop unrecoverable. Models
 * that DON'T require it (chatgpt, alibaba) simply ignore the extra
 * field — round-tripping it is safe.
 */
export const buildAssistantToolCallMessage = (params: {
  readonly response: TChatCompletionResponse;
  readonly toolCalls: ReadonlyArray<TToolCall>;
}): TChatMessage | null => {
  if (params.toolCalls.length === 0) return null;
  const choice = params.response.choices[0];
  const reasoningContent =
    typeof choice?.message.reasoning_content === "string" &&
    choice.message.reasoning_content.length > 0
      ? choice.message.reasoning_content
      : undefined;
  return {
    role: "assistant",
    content: choice?.message.content ?? "",
    tool_calls: [...params.toolCalls],
    ...(reasoningContent !== undefined
      ? { reasoning_content: reasoningContent }
      : {}),
  };
};

/**
 * Build the `tool`-role follow-up messages carrying the search
 * results for each invoked `tool_call_id`. `contentsById` is the
 * result string per call id (already encoded — error strings are
 * tolerated and useful to the model).
 */
export const buildToolResultMessages = (params: {
  readonly calls: ReadonlyArray<TToolCall>;
  readonly contentsById: ReadonlyMap<string, string>;
}): TChatMessage[] =>
  params.calls.map((call) => ({
    role: "tool",
    tool_call_id: call.id,
    content:
      params.contentsById.get(call.id) ??
      "Search error: tool result was not produced",
  }));
