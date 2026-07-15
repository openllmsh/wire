/**
 * Grok (xAI) provider-native search — the grok analogue of
 * `codex-web-search.ts`'s hosted search. The xAI CLI chat proxy speaks the
 * OpenAI Responses wire and natively supports two SERVER-executed search
 * tools (probed live 2026-07-15 against `cli-chat-proxy.grok.com/v1/responses`):
 *
 *   - `{type: "web_search"}` — web search; the completed `web_search_call`
 *     item carries the query AND `action.sources` result urls;
 *   - `{type: "x_search"}`  — X/Twitter search; executed server-side as
 *     `custom_tool_call` items named `x_semantic_search`/`x_keyword_search`
 *     (status "completed"), query in the item's JSON `input`.
 *
 * When the CLIENT declared the Anthropic `web_search_*` server tool
 * (`declaresAnthropicServerSearchTool`), the walker swaps its canonicalised
 * `web_search` FUNCTION tool for these native tools — one search owner per
 * turn, execution fully provider-side (no gateway loop; the turn completes
 * in one request). The chatgpt streaming decoder re-emits the lifecycle as
 * canonical `server_search_calls`, and the messages encoders re-encode the
 * blocks + usage the client expects.
 */

/** The native tools injected on a search-declared grok hop. */
export const GROK_NATIVE_SEARCH_TOOLS: ReadonlyArray<
  Readonly<Record<string, unknown>>
> = [{ type: "web_search" }, { type: "x_search" }];

/**
 * Swap the canonicalised `web_search` function tool for grok's NATIVE search
 * tools on a BUILT Responses-wire payload. Every other tool passes through
 * unchanged; the payload is otherwise untouched. Idempotent-ish: native
 * entries are only appended when absent.
 */
export const withGrokNativeSearch = (body: unknown): unknown => {
  if (body === null || typeof body !== "object") return body;
  const record = body as Record<string, unknown>;
  const tools = Array.isArray(record.tools) ? record.tools : [];
  const kept = tools.filter(
    (t) =>
      !(
        t !== null &&
        typeof t === "object" &&
        (t as { readonly type?: unknown }).type === "function" &&
        (t as { readonly name?: unknown }).name === "web_search"
      ),
  );
  const present = new Set(
    kept.flatMap((t) =>
      t !== null &&
      typeof t === "object" &&
      typeof (t as { readonly type?: unknown }).type === "string"
        ? [(t as { readonly type: string }).type]
        : [],
    ),
  );
  const injected = GROK_NATIVE_SEARCH_TOOLS.filter(
    (t) => !present.has(t.type as string),
  );
  return { ...record, tools: [...kept, ...injected] };
};
