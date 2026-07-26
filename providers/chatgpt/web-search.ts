/**
 * ChatGPT/Codex provider-native hosted search for handrolled Responses hops.
 *
 * Codex enables the same capability on its bridge through
 * `thread/start.config.web_search: "live"` (see
 * `packages/daemon/src/native-runtime/codex-web-search.ts`). On the Responses
 * API, that hosted capability is the OpenAI-native `{ type: "web_search" }`
 * tool, whose completed lifecycle is emitted as `web_search_call` and decoded
 * by `./streaming` into canonical `server_search_calls`.
 *
 * When an Anthropic client explicitly declares a `web_search_*` server tool,
 * the walker replaces the canonicalised client `web_search` function with this
 * provider-owned tool. This keeps exactly one search owner on the turn and
 * prevents an unexecutable client function call from leaking from a
 * handrolled ChatGPT hop.
 */

/** The OpenAI Responses API hosted-search tool supported by Codex. */
export const CHATGPT_NATIVE_SEARCH_TOOL: Readonly<Record<string, unknown>> = {
  type: "web_search",
};

/**
 * Swap the canonicalised `web_search` function for ChatGPT's native hosted
 * search tool on a BUILT Responses-wire payload. Every other tool passes
 * through unchanged and in order. Re-applying the transform does not add a
 * duplicate native tool.
 */
const isCanonicalSearchTool = (tool: unknown): boolean =>
  tool !== null &&
  typeof tool === "object" &&
  (tool as { readonly type?: unknown }).type === "function" &&
  (tool as { readonly name?: unknown }).name === "web_search";

const isForcedCanonicalSearchChoice = (choice: unknown): boolean =>
  choice !== null &&
  typeof choice === "object" &&
  (choice as { readonly type?: unknown }).type === "function" &&
  (choice as { readonly name?: unknown }).name === "web_search";

export const withChatGptNativeSearch = (body: unknown): unknown => {
  if (body === null || typeof body !== "object") return body;
  const record = body as Record<string, unknown>;
  const { tool_choice: toolChoice, ...rest } = record;
  const tools = Array.isArray(record.tools) ? record.tools : [];
  const kept = tools.filter((tool) => !isCanonicalSearchTool(tool));
  const removedCanonicalSearch = kept.length !== tools.length;
  const nativeSearchPresent = kept.some(
    (tool) =>
      tool !== null &&
      typeof tool === "object" &&
      (tool as { readonly type?: unknown }).type === "web_search",
  );
  if (!removedCanonicalSearch && !nativeSearchPresent) return body;

  return {
    ...rest,
    tools: nativeSearchPresent ? kept : [...kept, CHATGPT_NATIVE_SEARCH_TOOL],
    ...(!isForcedCanonicalSearchChoice(toolChoice) && toolChoice !== undefined
      ? { tool_choice: toolChoice }
      : {}),
  };
};
