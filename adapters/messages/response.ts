import type {
  TAnthropicContentBlock,
  TAnthropicResponse,
  TAnthropicStopReason,
  TChatCompletionResponse,
  TToolCall,
} from "@openllmsh/protocol";
import { ensureCompactionSafeVisibleText } from "../../features/compaction/compaction-text";
import { parseToolArguments } from "../../lib/canonical/message";
import { plainTextFromReasoningItems } from "./reasoning-from-items";
import {
  encodeReasoningSignature,
  reasoningItemsFromUnknown,
} from "./reasoning-signature";

/** Mirrors the streaming adapter's empty-summary thinking placeholder. */
const REASONING_PLACEHOLDER_TEXT = "[reasoning]";

const toAnthropicStopReason = (
  finish: TChatCompletionResponse["choices"][number]["finish_reason"],
): TAnthropicStopReason | null => {
  if (finish === null) return null;
  switch (finish) {
    case "stop":
      return "end_turn";
    case "length":
      return "max_tokens";
    case "tool_calls":
    case "function_call":
      return "tool_use";
    case "content_filter":
      return "refusal";
  }
};

const extractText = (
  content: TChatCompletionResponse["choices"][number]["message"]["content"],
): string => {
  if (content == null) return "";
  if (typeof content === "string") return content;
  return content
    .filter((p): p is { type: "text"; text: string } => p.type === "text")
    .map((p) => p.text)
    .join("");
};

const toolCallToUseBlock = (call: TToolCall): TAnthropicContentBlock => ({
  type: "tool_use",
  id: call.id,
  name: call.function.name,
  input: parseToolArguments(call.function.arguments),
});

/**
 * Adapt the canonical OpenAI ChatCompletionResponse back to an
 * Anthropic Messages response for a client that called `/v1/messages`.
 *
 * Phase 2: text + tool_use content blocks. The output `content` array
 * carries both a text block (if any text was generated) and one
 * tool_use block per OpenAI `tool_calls[*]`.
 */
export const toAnthropicMessagesResponse = (
  resp: TChatCompletionResponse,
): TAnthropicResponse => {
  const choice = resp.choices[0];
  const text = choice !== undefined ? extractText(choice.message.content) : "";
  const reasoningFromMessage =
    choice !== undefined &&
    typeof choice.message.reasoning_content === "string" &&
    choice.message.reasoning_content.length > 0
      ? choice.message.reasoning_content
      : "";
  const reasoningFromItems =
    choice !== undefined
      ? plainTextFromReasoningItems(choice.message.reasoning_items)
      : "";
  /** Same fields as ref `_build_reasoning_item` summary text; prefer streamed `reasoning_content` first. */
  const reasoning =
    reasoningFromMessage.length > 0 ? reasoningFromMessage : reasoningFromItems;
  const stopReason =
    choice !== undefined ? toAnthropicStopReason(choice.finish_reason) : null;

  // Round-trip the upstream `reasoning` item(s) (Codex/Responses
  // `encrypted_content`) through the thinking block's opaque
  // `signature` so an Anthropic client replays them next turn. Without
  // this a reasoning upstream loses chain-of-thought state every turn
  // and loops forever.
  const reasoningSignature =
    choice !== undefined
      ? encodeReasoningSignature(
          reasoningItemsFromUnknown(choice.message.reasoning_items),
        )
      : null;

  const content: TAnthropicContentBlock[] = [];
  // Provider-executed hosted searches (e.g. Codex hosted web_search on a
  // chatgpt hop) → the Anthropic wire's server-tool vocabulary, BEFORE the
  // text that cites them. `server_tool_use.input` carries the real query the
  // provider searched; the result CONTENT stays provider-internal (Codex
  // never exposes result items), so the paired `web_search_tool_result` is
  // an empty result list — honest: the search ran, its findings ride the
  // grounded answer text. Clients like Claude Code count these blocks (and
  // `usage.server_tool_use.web_search_requests`) for their "Did N searches".
  const serverSearches = choice?.message.server_search_calls ?? [];
  for (const search of serverSearches) {
    content.push({
      type: "server_tool_use",
      id: search.id,
      name: "web_search",
      input: { query: search.query },
    });
    content.push({
      type: "web_search_tool_result",
      tool_use_id: search.id,
      content: (search.results ?? []).map((r) => ({
        type: "web_search_result",
        url: r.url,
        title: r.title ?? r.url,
      })),
    });
  }
  // A `thinking` block is emitted ONLY when it carries a replay-safe
  // `signature` (the Codex/Responses `encrypted_content` round-trip via
  // our `openllm-rs1:` codec). Anthropic hard-rejects a signature-less
  // `thinking` block the moment Claude Code replays the assistant turn
  // (`messages.N.content.0.thinking.signature: Field required`) — which
  // then sends Claude Code into a retry storm and breaks prompt-cache
  // reuse, draining the user's subscription. Reasoning WITHOUT a
  // signature (every non-Codex reasoner, or a genuine-Anthropic turn
  // that fell to the canonical path on a fallback hop) is therefore
  // surfaced as plain visible `text`: always replay-safe.
  if (reasoningSignature !== null) {
    content.push({
      type: "thinking",
      thinking: reasoning.length > 0 ? reasoning : REASONING_PLACEHOLDER_TEXT,
      signature: reasoningSignature,
    });
  }
  const visibleReasoning =
    reasoningSignature === null && reasoning.length > 0 ? reasoning : "";
  if (text.length > 0) {
    const body =
      visibleReasoning.length > 0 ? `${visibleReasoning}\n\n${text}` : text;
    content.push({
      type: "text",
      text: ensureCompactionSafeVisibleText(body),
    });
  } else if (visibleReasoning.length > 0) {
    content.push({
      type: "text",
      text: ensureCompactionSafeVisibleText(visibleReasoning),
    });
  }
  for (const call of choice?.message.tool_calls ?? []) {
    content.push(toolCallToUseBlock(call));
  }

  // Claude Code `/compact` requires a non-empty user-visible `text`
  // block for “conversation summary” responses. A Codex reply that
  // filled ONLY the signed `thinking` block (no `text`) would otherwise
  // fail with “did not contain valid text content”. Tool-only (or
  // other) replies with no reasoning must not synthesize the compaction
  // fallback as visible assistant text.
  if (!content.some((b) => b.type === "text") && reasoning.length > 0) {
    content.push({
      type: "text",
      text: ensureCompactionSafeVisibleText(reasoning),
    });
  }

  const cached = resp.usage.prompt_tokens_details?.cached_tokens ?? 0;
  const created = resp.usage.prompt_tokens_details?.cache_creation_tokens ?? 0;

  // Anthropic's `input_tokens` EXCLUDES cache reads and cache-creation
  // tokens — they are reported in their own fields. OpenAI's
  // `prompt_tokens` INCLUDES `prompt_tokens_details.cached_tokens` (and,
  // for providers that report it, the creation tokens). Returning
  // `prompt_tokens` verbatim as `input_tokens` while ALSO surfacing
  // `cache_read_input_tokens` double-counts cached prompt tokens and
  // over-bills every cached request. Mirror LiteLLM: subtract the cache
  // tokens back out, flooring at 0. Ref: litellm
  // `adapters/transformation.py` usage mapping (~lines 1326-1357).
  const inputTokens = Math.max(0, resp.usage.prompt_tokens - cached - created);

  return {
    id: resp.id,
    type: "message",
    role: "assistant",
    model: resp.model,
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: inputTokens,
      output_tokens: resp.usage.completion_tokens,
      ...(cached > 0 ? { cache_read_input_tokens: cached } : {}),
      ...(created > 0 ? { cache_creation_input_tokens: created } : {}),
      ...(serverSearches.length > 0
        ? { server_tool_use: { web_search_requests: serverSearches.length } }
        : {}),
    },
  };
};
