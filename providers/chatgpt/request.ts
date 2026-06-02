import type {
  TChatCompletionRequest,
  TChatGptProviderOptions,
  TChatMessage,
} from "@openllm/schema";
import {
  reasoningItemsFromUnknown,
  reasoningItemToResponsesInput,
  type TReasoningResponsesInput,
} from "../../adapters/messages/reasoning-signature";
import { extractMessageText } from "../../lib/canonical/message";
import { CHATGPT_DEFAULT_INSTRUCTIONS } from "./common";

const CHATGPT_NAME_RE = /^[a-zA-Z0-9_-]+$/;
const CHATGPT_NAME_SUB_RE = /[^a-zA-Z0-9_-]/g;
const COLLAPSE_UNDERSCORE_RE = /_+/g;

/**
 * Coerce a `name` field to match `^[a-zA-Z0-9_-]+$`. ChatGPT 400s on
 * any other character with a `pattern` error which triggers a retry
 * spiral. Mirrors `chat/transformation.py:64-79`.
 */
const sanitizeName = (name: string): string => {
  if (name === "" || CHATGPT_NAME_RE.test(name)) return name;
  const cleaned = name
    .replace(CHATGPT_NAME_SUB_RE, "_")
    .replace(COLLAPSE_UNDERSCORE_RE, "_")
    .replace(/^_+|_+$/g, "");
  return cleaned.length > 0 ? cleaned : "tool";
};

// runtime-only: a single Responses API content part. The input/output
// distinction matters — the chatgpt.com endpoint rejects an
// `output_text` part on a `user` message and vice versa.
type TResponsesContentPart =
  | { readonly type: "input_text"; readonly text: string }
  | { readonly type: "output_text"; readonly text: string }
  | {
      readonly type: "input_image";
      readonly image_url: string;
      readonly detail?: "auto" | "low" | "high";
    };

// runtime-only: a single item in the Responses API `input` array.
// Mirrors the union from openai-python's `ResponseInputItem`.
type TResponsesInputItem =
  | {
      readonly type: "message";
      readonly role: "user" | "assistant" | "system" | "developer";
      readonly content: ReadonlyArray<TResponsesContentPart>;
    }
  | {
      readonly type: "function_call";
      readonly call_id: string;
      readonly name: string;
      readonly arguments: string;
    }
  | {
      readonly type: "function_call_output";
      readonly call_id: string;
      readonly output: ReadonlyArray<TResponsesContentPart>;
    }
  | TReasoningResponsesInput;

const contentToInputParts = (
  content: TChatMessage["content"] | null | undefined,
): TResponsesContentPart[] => {
  if (content == null) return [{ type: "input_text", text: "" }];
  if (typeof content === "string") {
    return [{ type: "input_text", text: content }];
  }
  const parts: TResponsesContentPart[] = [];
  for (const block of content) {
    if (block.type === "text") {
      parts.push({ type: "input_text", text: block.text });
    } else if (block.type === "image_url") {
      parts.push({
        type: "input_image",
        image_url: block.image_url.url,
        ...(block.image_url.detail !== undefined
          ? { detail: block.image_url.detail }
          : {}),
      });
    }
  }
  if (parts.length === 0) parts.push({ type: "input_text", text: "" });
  return parts;
};

const contentToOutputParts = (
  content: TChatMessage["content"] | null | undefined,
): TResponsesContentPart[] => {
  if (content == null) return [{ type: "output_text", text: "" }];
  if (typeof content === "string") {
    return [{ type: "output_text", text: content }];
  }
  const parts: TResponsesContentPart[] = [];
  for (const block of content) {
    if (block.type === "text") {
      parts.push({ type: "output_text", text: block.text });
    }
  }
  if (parts.length === 0) parts.push({ type: "output_text", text: "" });
  return parts;
};

/**
 * Pull every `role: "system"` message out of the array, return both the
 * trimmed message list and the concatenated text. ChatGPT's
 * `/backend-api/codex/responses` endpoint rejects system turns on the
 * wire — they must ride in the top-level `instructions` field.
 *
 * Mirrors `_merge_system_and_developer_into_instruction_text` from
 * `chat/transformation.py:41-61`.
 */
const extractSystemInstructions = (
  messages: ReadonlyArray<TChatMessage>,
): { conversation: TChatMessage[]; instructions: string } => {
  const parts: string[] = [];
  const conversation: TChatMessage[] = [];
  for (const msg of messages) {
    if (msg.role === "system") {
      const text = extractMessageText(msg.content);
      if (text.trim().length > 0) parts.push(text);
      continue;
    }
    conversation.push(msg);
  }
  return {
    conversation,
    instructions: parts.filter((p) => p.trim().length > 0).join("\n\n"),
  };
};

/**
 * Convert the canonical OpenAI ChatCompletion message array into
 * Responses API input items. Mirrors
 * `convert_chat_completion_messages_to_responses_api` from
 * `completion_extras/litellm_responses_transformation/transformation.py:203-289`.
 *
 * - `user` / `system` content -> `input_text` parts (system already
 *   pulled into `instructions` upstream of this call).
 * - `assistant` text content  -> `output_text` parts.
 * - `assistant.tool_calls`    -> one `function_call` item per call.
 * - `tool` (tool result)      -> `function_call_output` with parts.
 */
const messagesToInputItems = (
  messages: ReadonlyArray<TChatMessage>,
): TResponsesInputItem[] => {
  const items: TResponsesInputItem[] = [];
  for (const msg of messages) {
    if (msg.role === "user") {
      items.push({
        type: "message",
        role: "user",
        content: contentToInputParts(msg.content),
      });
      continue;
    }
    if (msg.role === "tool") {
      items.push({
        type: "function_call_output",
        call_id: msg.tool_call_id,
        // Codex/Responses API expects `output` as a string, not parts.
        // Coerce parts -> joined text. Mirrors LiteLLM tool message
        // -> function_call_output convention.
        output: contentToInputParts(msg.content),
      });
      continue;
    }
    if (msg.role === "assistant") {
      // Echo prior `reasoning` item(s) back, in order, immediately
      // before the assistant's tool calls / content. The Responses API
      // requires this for reasoning models (`store: false`); dropping
      // it makes the model restart reasoning and loop. Mirrors litellm
      // `transformation.py:261-262, 279-280`.
      const reasoningItems = reasoningItemsFromUnknown(msg.reasoning_items);
      for (const r of reasoningItems) {
        items.push(reasoningItemToResponsesInput(r));
      }
      const toolCalls = msg.tool_calls;
      if (toolCalls !== undefined && toolCalls.length > 0) {
        for (const call of toolCalls) {
          items.push({
            type: "function_call",
            call_id: call.id,
            name: sanitizeName(call.function.name),
            arguments: call.function.arguments,
          });
        }
        // Assistant text alongside tool_calls is rare, but allowed —
        // emit a separate message item if present.
        if (msg.content != null) {
          const text = extractMessageText(msg.content);
          if (text.trim().length > 0) {
            items.push({
              type: "message",
              role: "assistant",
              content: contentToOutputParts(msg.content),
            });
          }
        }
        continue;
      }
      if (msg.content != null) {
        items.push({
          type: "message",
          role: "assistant",
          content: contentToOutputParts(msg.content),
        });
      }
    }
    // system messages are filtered out before this call.
  }
  return items;
};

// runtime-only: a single tool definition in the Responses API. Note
// the FLAT shape — the chat-completions tool wrapper
// (`{type:"function", function:{name,...}}`) is not accepted here.
type TResponsesToolDef = {
  readonly type: "function";
  readonly name: string;
  readonly description?: string;
  readonly parameters?: unknown;
  readonly strict?: boolean;
};

// runtime-only: a Codex built-in / non-function tool carried verbatim from the
// inbound Responses request (`custom` apply_patch, `web_search`,
// `image_generation`, `tool_search`). Opaque — re-emitted as-is to the chatgpt
// upstream, which is the same endpoint Codex sends them to natively.
type TResponsesPassthroughToolDef = {
  readonly type: string;
  readonly [key: string]: unknown;
};

const toolsToResponses = (
  tools: NonNullable<TChatCompletionRequest["tools"]>,
): TResponsesToolDef[] =>
  tools.map((tool) => ({
    type: "function",
    name: sanitizeName(tool.function.name),
    ...(tool.function.description !== undefined
      ? { description: tool.function.description }
      : {}),
    ...(tool.function.parameters !== undefined
      ? { parameters: tool.function.parameters }
      : {}),
    ...(tool.function.strict !== undefined
      ? { strict: tool.function.strict }
      : {}),
  }));

// runtime-only: tool_choice in the Responses API. Mirrors the tools
// shape — FLAT `{type:"function", name}`, NOT the chat-completions
// `{type:"function", function:{name}}` wrapper. Forwarding the chat
// shape verbatim 400s with `Unknown parameter: 'tool_choice.function'.`
type TResponsesToolChoice =
  | "auto"
  | "none"
  | "required"
  | { readonly type: "function"; readonly name: string };

const toResponsesToolChoice = (
  choice: NonNullable<TChatCompletionRequest["tool_choice"]>,
): TResponsesToolChoice =>
  choice === "auto" || choice === "none" || choice === "required"
    ? choice
    : { type: "function", name: sanitizeName(choice.function.name) };

// runtime-only: payload sent to `/backend-api/codex/responses`. Strictly
// the keys allowed by `ChatGPTResponsesAPIConfig.transform_responses_api_request`
// (`responses/transformation.py:215-227`). Anything outside this list
// is dropped to avoid `Unsupported parameter` 400s.
//
// Notably ABSENT: `max_output_tokens`, `temperature`, `top_p`,
// `frequency_penalty`, `presence_penalty`, `seed`, `response_format`,
// `metadata`, `user`. The Codex endpoint silently drops the standard
// Responses API token-cap field, so we don't bother forwarding it.
export type TChatGptRequestBody = {
  readonly model: string;
  readonly input: ReadonlyArray<TResponsesInputItem>;
  readonly instructions: string;
  readonly stream: true;
  readonly store: false;
  readonly include: ReadonlyArray<string>;
  readonly tools?: ReadonlyArray<
    TResponsesToolDef | TResponsesPassthroughToolDef
  >;
  readonly tool_choice?: TResponsesToolChoice;
  readonly reasoning?: { readonly effort: "low" | "medium" | "high" };
  readonly previous_response_id?: string;
  readonly truncation?: "auto" | "disabled";
};

/**
 * Convert canonical OpenAI ChatCompletion → ChatGPT/Codex Responses API body.
 *
 * 1. Pull system messages into `instructions`.
 * 2. Prepend the Codex preamble if not already present (required by
 *    gpt-5.x or the server returns empty `output`).
 * 3. Convert `messages` -> `input` items.
 * 4. Sanitize every tool name + assistant tool_call name.
 * 5. Force `stream: true`, `store: false`,
 *    `include: ["reasoning.encrypted_content"]`.
 * 6. Map `max_tokens` / `max_completion_tokens` -> `max_output_tokens`.
 * 7. Map `reasoning_effort` -> `reasoning.effort`.
 * 8. DROP every other key — only the allowed-list is forwarded.
 *
 * Mirrors `transform_request` in `chat/transformation.py:212-248` plus
 * the Responses-API allowed-list filter in
 * `responses/transformation.py:215-229`.
 */
export const toChatGptRequest = (
  req: TChatCompletionRequest,
  options: TChatGptProviderOptions,
): TChatGptRequestBody => {
  const { conversation, instructions: fromSystem } = extractSystemInstructions(
    req.messages,
  );

  let instructions = fromSystem;
  if (!instructions.includes(CHATGPT_DEFAULT_INSTRUCTIONS)) {
    instructions =
      instructions.length > 0
        ? `${CHATGPT_DEFAULT_INSTRUCTIONS}\n\n${instructions}`
        : CHATGPT_DEFAULT_INSTRUCTIONS;
  }

  const input = messagesToInputItems(conversation);
  // Prefer the verbatim `responses_tools` passthrough (Codex's full original
  // tool set, function + non-function) — it round-trips apply_patch /
  // web_search / image_generation / tool_search intact to the same endpoint
  // Codex speaks to natively. Fall back to the function-only canonical tools
  // (a cross-wire client, or a non-Codex caller that set `tools`).
  const responsesTools:
    | ReadonlyArray<TResponsesToolDef | TResponsesPassthroughToolDef>
    | undefined =
    req.responses_tools !== undefined && req.responses_tools.length > 0
      ? (req.responses_tools as ReadonlyArray<TResponsesPassthroughToolDef>)
      : req.tools !== undefined && req.tools.length > 0
        ? toolsToResponses(req.tools)
        : undefined;

  return {
    model: options.providerModelId,
    input,
    instructions,
    stream: true,
    store: false,
    include: ["reasoning.encrypted_content"],
    ...(responsesTools !== undefined ? { tools: responsesTools } : {}),
    ...(req.tool_choice !== undefined
      ? { tool_choice: toResponsesToolChoice(req.tool_choice) }
      : {}),
    ...(() => {
      // ChatGPT's Responses API only accepts `low | medium | high`.
      // Map the wider canonical enum (`minimal/xhigh/max/none`) down to
      // the closest supported neighbour: `minimal` → low, `xhigh`/`max`
      // → high, `none` → reasoning omitted entirely.
      const e = req.reasoning_effort;
      if (e === undefined || e === "none") return {};
      const effort: "low" | "medium" | "high" =
        e === "minimal" || e === "low"
          ? "low"
          : e === "medium"
            ? "medium"
            : "high";
      return { reasoning: { effort } };
    })(),
  };
};
