import type {
  TChatCompletionRequest,
  TChatGptProviderOptions,
  TChatMessage,
} from "@openllmsh/protocol";
import type { TReasoningResponsesInput } from "../../adapters/messages/reasoning-signature";
import {
  reasoningItemsFromUnknown,
  reasoningItemToResponsesInput,
} from "../../adapters/messages/reasoning-signature";
import { extractMessageText } from "../../lib/canonical/message";
import { effectiveDeny, responsesWirePolicy } from "../upstream-deny";

const CHATGPT_NAME_RE = /^[a-zA-Z0-9_-]+$/;
const CHATGPT_NAME_SUB_RE = /[^a-zA-Z0-9_-]/g;
const COLLAPSE_UNDERSCORE_RE = /_+/g;
const CODEX_IDENTIFIER_MAX_LENGTH = 64;

/**
 * A stable, deterministic 64-bit hash (two FNV-1a passes with distinct offset
 * bases → 16 hex chars). Non-cryptographic — a prompt-cache key is a routing
 * hint, not a security boundary — but sync + env-agnostic (no `node:crypto` /
 * async WebCrypto), which the pure wire layer requires.
 */
const stableHash = (s: string): string => {
  let h1 = 0x811c9dc5; // FNV offset basis
  let h2 = 0x1000193; // FNV prime, reused as a second, independent seed
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 0x01000193) >>> 0;
    h2 = Math.imul(h2 ^ c, 0x85ebca77) >>> 0;
  }
  return h1.toString(16).padStart(8, "0") + h2.toString(16).padStart(8, "0");
};

/**
 * Hash the IMMUTABLE conversation prefix — the first user turn's COMPLETE
 * content plus canonical system text (`instructions`) — into a stable 16-hex
 * digest. Shared
 * by {@link derivePromptCacheKey} and {@link deriveChatGptSessionId}: both key
 * off this prefix (it never changes as the conversation grows), so both are
 * stable across turns and distinct across conversations. JSON (not a delimited
 * concat) keeps conversations that differ only by non-text parts (images/files)
 * distinct and removes delimiter ambiguity.
 */
const conversationPrefixHash = (
  conversation: ReadonlyArray<TChatMessage>,
  instructions: string,
): string => {
  const firstUser = conversation.find((m) => m.role === "user");
  const firstUserContent = firstUser !== undefined ? firstUser.content : null;
  return stableHash(JSON.stringify({ firstUserContent, instructions }));
};

/**
 * Derive a prompt-cache key that is STABLE across every turn of one
 * conversation and DISTINCT across conversations. Codex uses its own
 * `thread_id`; the stateless gateway has none, so we key off the immutable
 * conversation prefix — the canonical system text plus the first user turn —
 * which never changes as the conversation grows. Without this, OpenAI's
 * automatic prefix-hash routing can collide distinct conversations onto the
 * same cache lane, causing them to evict each other and cache-hit rate to
 * collapse — burning subscription quota. See the reference
 * `codex-rs/core/src/client.rs::prompt_cache_key`.
 */
const derivePromptCacheKey = (
  instructions: string,
  conversation: ReadonlyArray<TChatMessage>,
): string => `openllm-${conversationPrefixHash(conversation, instructions)}`;

/** Max `prompt_cache_key` length the OpenAI Responses backend accepts. A
 *  longer key 400s (`prompt_cache_key` too long); the partner client clamps to
 *  the same bound. Our synthesized key is 24 chars — this only bites a
 *  client's forwarded key. */
const PROMPT_CACHE_KEY_MAX = 64;

/** Clamp by Unicode code point (not UTF-16 unit) so a multi-byte key isn't
 *  split mid-character. */
const clampPromptCacheKey = (key: string): string => {
  const cps = Array.from(key);
  return cps.length <= PROMPT_CACHE_KEY_MAX
    ? key
    : cps.slice(0, PROMPT_CACHE_KEY_MAX).join("");
};

/**
 * Clamp a Codex tool identifier to the Responses backend's 64-character
 * bound. Long values retain a 47-character prefix and gain a `_` plus the
 * stable 16-hex digest of the complete original value, making the result
 * deterministic and collision-resistant without changing short identifiers.
 */
const clampCodexIdentifier = (value: string): string => {
  if (value.length <= CODEX_IDENTIFIER_MAX_LENGTH) return value;
  const suffix = `_${stableHash(value)}`;
  return `${value.slice(0, CODEX_IDENTIFIER_MAX_LENGTH - suffix.length)}${suffix}`;
};

/**
 * Coerce a `name` field to match `^[a-zA-Z0-9_-]+$`, then clamp it to the
 * Codex Responses backend's 64-character limit. ChatGPT 400s on any other
 * character with a `pattern` error which triggers a retry spiral. Mirrors
 * `chat/transformation.py:64-79`.
 *
 * A request-scoped reverse map built by {@link buildChatGptToolNameMap} restores
 * any lossy rewrite before the streaming decoder emits a client tool call.
 */
const sanitizeName = (name: string): string => {
  if (name === "" || CHATGPT_NAME_RE.test(name)) {
    return clampCodexIdentifier(name);
  }
  const cleaned = name
    .replace(CHATGPT_NAME_SUB_RE, "_")
    .replace(COLLAPSE_UNDERSCORE_RE, "_")
    .replace(/^_+|_+$/g, "");
  return clampCodexIdentifier(cleaned.length > 0 ? cleaned : "tool");
};

/** Call IDs do not share tool-name charset constraints, only the 64-character
 * Responses API length limit. Applying the same pure clamp on both the prior
 * function call and its tool result preserves their pairing. */
const clampCodexCallId = (callId: string): string =>
  clampCodexIdentifier(callId);

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
    }
  | {
      readonly type: "input_file";
      readonly filename?: string;
      readonly file_data?: string;
      readonly file_id?: string;
      readonly file_url?: string;
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
      readonly output: string;
    }
  | {
      /** Opaque Codex harness declarations, preserved from Responses input. */
      readonly type: "additional_tools";
      readonly role: "developer";
      readonly tools: ReadonlyArray<unknown>;
      readonly [key: string]: unknown;
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
    } else if (block.type === "file") {
      // Responses `input_file` takes exactly ONE file carrier. Pick a single
      // one by precedence — a public `url` (clean-context, no bytes) wins,
      // then a provider `file_id`, then an inline `file_data` data URL — so a
      // caller that sets more than one can't emit an ambiguous part.
      const fileCarrier =
        block.file.url !== undefined
          ? { file_url: block.file.url }
          : block.file.file_id !== undefined
            ? { file_id: block.file.file_id }
            : block.file.file_data !== undefined
              ? { file_data: block.file.file_data }
              : {};
      parts.push({
        type: "input_file",
        ...(block.file.filename !== undefined
          ? { filename: block.file.filename }
          : {}),
        ...fileCarrier,
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
 * Pull every `role: "system"` message out of the array, returning both the
 * non-system conversation and concatenated system text. This helper exists
 * only for stable prompt-cache and session hashing: outbound system turns are
 * emitted as `developer` input items, while the top-level `instructions` field
 * is always empty.
 *
 * Mirrors the immutable-prefix portion of
 * `_merge_system_and_developer_into_instruction_text` from
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
 * Derive a STABLE per-conversation `session_id` for the ChatGPT Codex backend's
 * prompt-cache affinity. Without a stable `session_id` the backend routes each
 * request to a cold machine and caches NOTHING — every turn re-bills the full
 * conversation input at full rate (verified live: `cached=0` without it vs
 * ~90% with it; audit 2026-07-14-codex-handrolled-quota-drain). Keyed on the
 * immutable conversation prefix (first user turn + system instructions),
 * mirroring {@link derivePromptCacheKey}: stable across every turn of one
 * conversation, distinct across conversations (so traffic doesn't hot-spot one
 * machine). The hash retains canonical system text even though the outbound
 * `instructions` field is always empty.
 */
export const deriveChatGptSessionId = (req: TChatCompletionRequest): string => {
  const { conversation, instructions } = extractSystemInstructions(
    req.messages,
  );
  return `openllm-sess-${conversationPrefixHash(conversation, instructions)}`;
};

const TOOL_RESULT_IMAGE_REPLAY_TEXT = "Attached image(s) from tool result:";

/**
 * Convert one canonical tool-result message into Responses input items.
 * `function_call_output.output` is a STRING — the shape Codex itself and
 * litellm's reference transformation send, and the one field xAI's partner
 * client (openclaw) force-coerces for Grok before every request (see
 * docs/audit/2026-07-14-grok-upstream-wire-openclaw-comparison.md §F1).
 * Non-text parts can't ride in the string: images are replayed as an
 * IMMEDIATELY-FOLLOWING `user` message (adjacent, not end-of-input, so the
 * conversation prefix stays byte-stable across turns for prompt caching);
 * media without a text sibling leaves a placeholder so the model knows the
 * tool returned something.
 */
const toolResultToItems = (
  msg: Extract<TChatMessage, { readonly role: "tool" }>,
): TResponsesInputItem[] => {
  const content = msg.content;
  const text = extractMessageText(content);
  const images: TResponsesContentPart[] = [];
  let hasOtherMedia = false;
  if (content != null && typeof content !== "string") {
    for (const block of content) {
      if (block.type === "image_url") {
        images.push({
          type: "input_image",
          image_url: block.image_url.url,
          ...(block.image_url.detail !== undefined
            ? { detail: block.image_url.detail }
            : {}),
        });
      } else if (block.type !== "text") {
        hasOtherMedia = true;
      }
    }
  }
  const output =
    text.trim().length > 0
      ? text
      : hasOtherMedia
        ? "(see attached media)"
        : images.length > 0
          ? "(see attached image)"
          : "";
  const items: TResponsesInputItem[] = [
    {
      type: "function_call_output",
      call_id: clampCodexCallId(msg.tool_call_id),
      output,
    },
  ];
  if (images.length > 0) {
    items.push({
      type: "message",
      role: "user",
      content: [
        { type: "input_text", text: TOOL_RESULT_IMAGE_REPLAY_TEXT },
        ...images,
      ],
    });
  }
  return items;
};

/**
 * Convert the canonical OpenAI ChatCompletion message array into
 * Responses API input items. Mirrors
 * `convert_chat_completion_messages_to_responses_api` from
 * `completion_extras/litellm_responses_transformation/transformation.py:203-289`.
 *
 * - `system` content          -> `input_text` parts on a `developer` message.
 * - `user` content            -> `input_text` parts on a `user` message.
 * - `assistant` text content  -> `output_text` parts.
 * - `assistant.tool_calls`    -> one `function_call` item per call.
 * - `tool` (tool result)      -> `function_call_output` (string) via
 *   {@link toolResultToItems}, plus an image-replay user message.
 */
const messagesToInputItems = (
  messages: ReadonlyArray<TChatMessage>,
  toolNames: ReadonlyMap<string, string>,
): TResponsesInputItem[] => {
  const items: TResponsesInputItem[] = [];
  for (const msg of messages) {
    if (msg.role === "system") {
      const text = extractMessageText(msg.content);
      if (text.trim().length === 0) continue;
      items.push({
        type: "message",
        role: "developer",
        content: contentToInputParts(msg.content),
      });
      continue;
    }
    if (msg.role === "user") {
      items.push({
        type: "message",
        role: "user",
        content: contentToInputParts(msg.content),
      });
      continue;
    }
    if (msg.role === "tool") {
      items.push(...toolResultToItems(msg));
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
            call_id: clampCodexCallId(call.id),
            name: outboundToolName(call.function.name, toolNames),
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
  }
  return items;
};

/**
 * Recover the opaque leading `additional_tools` input items carried through the
 * canonical request. They originate at the validated Responses boundary; the
 * narrow runtime check keeps an arbitrary canonical caller from injecting a
 * malformed input item while preserving accepted payloads byte-for-byte.
 */
const additionalToolsToInputItems = (
  items: ReadonlyArray<unknown> | undefined,
): TResponsesInputItem[] => {
  if (items === undefined) return [];
  const result: TResponsesInputItem[] = [];
  for (const item of items) {
    if (
      item !== null &&
      typeof item === "object" &&
      !Array.isArray(item) &&
      (item as { readonly type?: unknown }).type === "additional_tools" &&
      (item as { readonly role?: unknown }).role === "developer" &&
      Array.isArray((item as { readonly tools?: unknown }).tools)
    ) {
      result.push(item as TResponsesInputItem);
    }
  }
  return result;
};

const LOOKAROUND_RE = /\(\?[=!]|\(\?<[=!]/;

/**
 * Deep-walk a JSON-schema value and drop any `pattern` key whose string value
 * contains a regex lookaround construct (`(?=`, `(?!`, `(?<=`, `(?<!`). The
 * codex upstream 400s with `Invalid JSON schema: regex lookaround is not
 * supported. Found at $.properties.contact.properties.email.pattern.` on any
 * such schema, so tool params carrying one (e.g. an MCP tool validating email
 * format) must be stripped before forwarding. Pure — never mutates `params`;
 * a `pattern` key whose value isn't a string (e.g. a user data property
 * literally named "pattern") is left untouched.
 */
export const sanitizeToolParameters = (params: unknown): unknown => {
  if (Array.isArray(params)) {
    return params.map((item) => sanitizeToolParameters(item));
  }
  if (params === null || typeof params !== "object") return params;
  // Build via `fromEntries`, not `result[key] = ...` — a schema property
  // literally named `__proto__` would otherwise hit the inherited setter
  // and rewrite the object's prototype instead of becoming an own property.
  return Object.fromEntries(
    Object.entries(params)
      .filter(
        ([key, value]) =>
          !(
            key === "pattern" &&
            typeof value === "string" &&
            LOOKAROUND_RE.test(value)
          ),
      )
      .map(([key, value]) => [key, sanitizeToolParameters(value)]),
  );
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

const EMPTY_TOOL_NAME_MAP: ReadonlyMap<string, string> = new Map();

type TChatGptToolNameMaps = {
  /** Original client name → final unique Responses name. */
  readonly outbound: ReadonlyMap<string, string>;
  /** Final unique Responses name → original client name. */
  readonly inbound: ReadonlyMap<string, string>;
};

const EMPTY_TOOL_NAME_MAPS: TChatGptToolNameMaps = {
  outbound: EMPTY_TOOL_NAME_MAP,
  inbound: EMPTY_TOOL_NAME_MAP,
};

const functionResponsesToolName = (tool: unknown): string | undefined => {
  if (tool === null || typeof tool !== "object" || Array.isArray(tool)) {
    return undefined;
  }
  const record = tool as Record<string, unknown>;
  return record.type === "function" && typeof record.name === "string"
    ? record.name
    : undefined;
};

/**
 * Visit declared function names in their request order. Stable declaration
 * ordering makes suffix assignment reproducible across the request encoder and
 * stream decoder setup.
 */
const forEachFunctionToolName = (
  req: Pick<TChatCompletionRequest, "tools" | "responses_tools">,
  visit: (name: string) => void,
): void => {
  for (const tool of req.tools ?? []) visit(tool.function.name);
  for (const tool of req.responses_tools ?? []) {
    const name = functionResponsesToolName(tool);
    if (name !== undefined) visit(name);
  }
};

/** Append a numeric suffix while retaining Codex's 64-character bound. */
const makeUniqueToolName = (
  base: string,
  used: ReadonlySet<string>,
): string => {
  if (!used.has(base)) return base;
  for (let suffixNumber = 1; ; suffixNumber++) {
    const suffix = `_${suffixNumber}`;
    const prefixLength = Math.max(0, CODEX_IDENTIFIER_MAX_LENGTH - suffix.length);
    const candidate = `${base.slice(0, prefixLength)}${suffix}`;
    if (!used.has(candidate)) return candidate;
  }
};

/**
 * Build the paired request-scoped maps for Codex tool identifiers. Distinct
 * originals that normalize to the same Responses name are assigned `_1`, `_2`,
 * … in declaration order rather than rejecting the whole request. Repeated
 * declarations of one original reuse its first assignment.
 */
const buildChatGptToolNameMaps = (
  req: Pick<TChatCompletionRequest, "tools" | "responses_tools">,
): TChatGptToolNameMaps => {
  let hasLossyRewrite = false;
  forEachFunctionToolName(req, (name) => {
    if (sanitizeName(name) !== name) hasLossyRewrite = true;
  });
  if (!hasLossyRewrite) return EMPTY_TOOL_NAME_MAPS;

  const outbound = new Map<string, string>();
  const inbound = new Map<string, string>();
  const used = new Set<string>();
  forEachFunctionToolName(req, (originalName) => {
    if (outbound.has(originalName)) return;
    const assignedName = makeUniqueToolName(sanitizeName(originalName), used);
    outbound.set(originalName, assignedName);
    used.add(assignedName);
    if (assignedName !== originalName) {
      inbound.set(assignedName, originalName);
    }
  });
  return { outbound, inbound };
};

/**
 * Build the request-scoped reverse mapping the Responses stream needs to put
 * every client tool name back exactly as it was sent to OpenLLM.
 */
export const buildChatGptToolNameMap = (
  req: Pick<TChatCompletionRequest, "tools" | "responses_tools">,
): ReadonlyMap<string, string> => buildChatGptToolNameMaps(req).inbound;

const outboundToolName = (
  originalName: string,
  names: ReadonlyMap<string, string>,
): string => names.get(originalName) ?? sanitizeName(originalName);

const toolsToResponses = (
  tools: NonNullable<TChatCompletionRequest["tools"]>,
  toolNames: ReadonlyMap<string, string>,
): TResponsesToolDef[] =>
  tools.map((tool) => ({
    type: "function",
    name: outboundToolName(tool.function.name, toolNames),
    ...(tool.function.description !== undefined
      ? { description: tool.function.description }
      : {}),
    ...(tool.function.parameters !== undefined
      ? { parameters: sanitizeToolParameters(tool.function.parameters) }
      : {}),
    ...(tool.function.strict !== undefined
      ? { strict: tool.function.strict }
      : {}),
  }));

/**
 * Sanitize the verbatim `responses_tools` passthrough (Codex's original tool
 * set). Function tools carry a JSON-schema `parameters` and so are subject to
 * the same lookaround-`pattern` 400 as {@link toolsToResponses} — this is
 * actually the path real Codex traffic takes. Non-function tools
 * (`web_search`, `apply_patch`, `image_generation`, `tool_search`, …) must
 * round-trip byte-identical, since they're opaque and re-emitted as-is.
 */
const sanitizeResponsesTools = (
  tools: ReadonlyArray<TResponsesPassthroughToolDef>,
  toolNames: ReadonlyMap<string, string>,
): ReadonlyArray<TResponsesPassthroughToolDef> =>
  tools.map((tool) => {
    if (tool.type !== "function") return tool;
    return {
      ...tool,
      ...(typeof tool.name === "string"
        ? { name: outboundToolName(tool.name, toolNames) }
        : {}),
      ...("parameters" in tool
        ? { parameters: sanitizeToolParameters(tool.parameters) }
        : {}),
    };
  });

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
  toolNames: ReadonlyMap<string, string>,
): TResponsesToolChoice =>
  choice === "auto" || choice === "none" || choice === "required"
    ? choice
    : {
        type: "function",
        name: outboundToolName(choice.function.name, toolNames),
      };

/**
 * The Codex client provides the only reliable Responses-Lite signal: it mirrors
 * the internal websocket header into `client_metadata`. Model ids and account
 * tiers are deliberately NOT used — both Lite and full Responses can serve the
 * same catalog ids.
 */
// The Codex "spark" family (e.g. `gpt-5.3-codex-spark`) rejects
// `reasoning.context: "all_turns"` with a 400 — it only accepts `auto` and
// `current_turn`. Every other Codex model requires `all_turns`. Detect spark
// by model-id suffix (mirrors CLIProxyAPI's `strings.HasSuffix(baseModel,
// "spark")`) and omit `context` for it, falling back to the backend default.
const isCodexSparkModel = (modelId: string): boolean =>
  modelId.toLowerCase().endsWith("spark");

const isCodexResponsesLite = (clientMetadata: unknown): boolean => {
  if (
    clientMetadata === null ||
    typeof clientMetadata !== "object" ||
    Array.isArray(clientMetadata)
  ) {
    return false;
  }
  const marker = (clientMetadata as Record<string, unknown>)
    .ws_request_header_x_openai_internal_codex_responses_lite;
  return (
    marker === true ||
    (typeof marker === "string" && marker.trim().toLowerCase() === "true")
  );
};

// runtime-only: payload sent to `/backend-api/codex/responses`. Strictly
// the keys allowed by `ChatGPTResponsesAPIConfig.transform_responses_api_request`
// (`responses/transformation.py:215-227`), plus `max_output_tokens`. Anything
// outside this list is dropped to avoid `Unsupported parameter` 400s.
//
// Notably ABSENT: `temperature`, `top_p`, `frequency_penalty`,
// `presence_penalty`, `seed`, `response_format`, `metadata`, `user`.
// `temperature`/`top_p` are ACCEPTED by the Grok chat proxy (verified live
// 2026-07-14) but stay off this wire pending a chatgpt.com probe — the Codex
// allowed-list doesn't include them. `response_format`/`text.format` is
// deliberately dropped: the Grok proxy accepts the field but does NOT
// enforce it and derails into a runaway generation (a 16k-token garbage
// completion, verified live — audit 2026-07-14 §5).
export type TChatGptRequestBody = {
  readonly model: string;
  readonly input: ReadonlyArray<TResponsesInputItem>;
  // Always empty: canonical system messages are emitted as `developer` input.
  readonly instructions: string;
  readonly stream: true;
  readonly store: false;
  readonly include: ReadonlyArray<string>;
  // Responses-Lite requires false; full Codex Responses requires true. The
  // authoritative Lite marker lives in `client_metadata`, never a model id.
  readonly parallel_tool_calls?: boolean;
  /** Opaque Codex request metadata, including the Responses-Lite marker. */
  readonly client_metadata?: unknown;
  // The client's token cap — emitted ONLY on the non-Codex Responses
  // variant (`codexInstructions: false`, i.e. grok), where the chat proxy
  // honors it as a hard cap (verified live 2026-07-14). The chatgpt.com
  // Codex endpoint rejects it (`Unsupported parameter: max_output_tokens`),
  // so Codex hops keep dropping it. Grok also allow-backs temperature/top_p
  // (same live verify); both stay off the Codex body via WIRE_DENY.chatgpt.
  readonly max_output_tokens?: number;
  readonly temperature?: number;
  readonly top_p?: number;
  readonly tools?: ReadonlyArray<
    TResponsesToolDef | TResponsesPassthroughToolDef
  >;
  readonly tool_choice?: TResponsesToolChoice;
  // `summary: "auto"` rides with every effort — Codex itself sends it, the
  // Grok proxy accepts it (verified live), and the stream decoder already
  // maps `response.reasoning_summary_text.delta` → `reasoning_content`.
  // `context: "all_turns"` is REQUIRED by the ChatGPT Codex "Responses-Lite"
  // backend (gpt-5.6-terra/luna) and sent natively by codex v0.147. Codex path
  // only — the Grok chat proxy rejects it, so it's omitted there. The Codex
  // `spark` family also rejects it (only `auto`/`current_turn`), so it's
  // omitted for spark too — see `isCodexSparkModel`.
  readonly reasoning?: {
    readonly effort: "low" | "medium" | "high";
    readonly summary: "auto";
    readonly context?: "all_turns";
  };
  readonly previous_response_id?: string;
  readonly truncation?: "auto" | "disabled";
  // Stable per-conversation prompt-cache routing hint (preserved from the
  // inbound Codex request or synthesized from the prefix). Codex ALWAYS sends
  // this; omitting it collapses cache-hit rate. See `derivePromptCacheKey`.
  readonly prompt_cache_key?: string;
};

/**
 * Convert canonical OpenAI ChatCompletion → ChatGPT/Codex Responses API body.
 *
 * 1. Preserve system messages as `developer` items inside `input`.
 * 2. Keep top-level `instructions` as an empty string.
 * 3. Convert `messages` -> `input` items.
 * 4. Sanitize every tool name + assistant tool_call name.
 * 5. Force `stream: true`, `store: false`,
 *    `include: ["reasoning.encrypted_content"]`.
 * 6. Map `max_tokens` / `max_completion_tokens` -> `max_output_tokens` —
 *    non-Codex upstreams only (`codexInstructions: false`).
 * 7. Map `reasoning_effort` -> `reasoning.effort` (+ `summary: "auto"`).
 * 8. DROP every other key — only the allowed-list is forwarded.
 *
 * System text still feeds prompt-cache/session hashing, but never the body
 * `instructions` field.
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
  const isCodex = options.codexInstructions !== false;
  // Assign names once so function definitions, tool choices, and replayed
  // assistant calls all use the same final identifier.
  const toolNameMaps = buildChatGptToolNameMaps(req);
  // Codex requires harness declarations before every conversational turn.
  // They are not part of Grok's Responses contract.
  const input = [
    ...(isCodex
      ? additionalToolsToInputItems(req.responses_additional_tools)
      : []),
    ...messagesToInputItems(req.messages, toolNameMaps.outbound),
  ];
  // Prefer the verbatim `responses_tools` passthrough (Codex's full original
  // tool set, function + non-function) — it round-trips apply_patch /
  // web_search / image_generation / tool_search intact to the same endpoint
  // Codex speaks to natively. Fall back to the function-only canonical tools
  // (a cross-wire client, or a non-Codex caller that set `tools`).
  const responsesTools:
    | ReadonlyArray<TResponsesToolDef | TResponsesPassthroughToolDef>
    | undefined =
    req.responses_tools !== undefined && req.responses_tools.length > 0
      ? sanitizeResponsesTools(
          req.responses_tools as ReadonlyArray<TResponsesPassthroughToolDef>,
          toolNameMaps.outbound,
        )
      : req.tools !== undefined && req.tools.length > 0
        ? toolsToResponses(req.tools, toolNameMaps.outbound)
        : undefined;

  return {
    model: options.providerModelId,
    input,
    instructions: "",
    stream: true,
    store: false,
    include: ["reasoning.encrypted_content"],
    // Lite always disables parallel calls. Full Codex honors an explicit client
    // value and otherwise preserves Codex's default of parallel calls enabled.
    ...(isCodex
      ? {
          parallel_tool_calls: isCodexResponsesLite(
            req.responses_client_metadata,
          )
            ? false
            : (req.parallel_tool_calls ?? true),
        }
      : {}),
    ...(isCodex && req.responses_client_metadata !== undefined
      ? { client_metadata: req.responses_client_metadata }
      : {}),
    // Preserve the caller's key when present (a genuine Codex request already
    // carries a stable per-thread one); otherwise synthesize one off the
    // conversation prefix so turns of the same conversation share a cache lane.
    prompt_cache_key: clampPromptCacheKey(
      req.prompt_cache_key !== undefined && req.prompt_cache_key.length > 0
        ? req.prompt_cache_key
        : derivePromptCacheKey(fromSystem, conversation),
    ),
    ...(responsesTools !== undefined ? { tools: responsesTools } : {}),
    ...(req.tool_choice !== undefined
      ? {
          tool_choice: toResponsesToolChoice(
            req.tool_choice,
            toolNameMaps.outbound,
          ),
        }
      : {}),
    ...(() => {
      // Axis 1+2: Codex keeps the full WIRE_DENY.chatgpt drop set (body
      // byte-identical). Grok (`codexInstructions: false`) allow-backs
      // temperature / top_p / max_output_tokens. Map the OpenAI cap aliases
      // onto max_output_tokens; keep response_format / stop / seed / top_k
      // denied.
      const deny = effectiveDeny(
        "chatgpt",
        responsesWirePolicy(options.codexInstructions),
      );
      const extra: {
        temperature?: number;
        top_p?: number;
        max_output_tokens?: number;
      } = {};
      if (!deny.has("temperature") && req.temperature !== undefined) {
        extra.temperature = req.temperature;
      }
      if (!deny.has("top_p") && req.top_p !== undefined) {
        extra.top_p = req.top_p;
      }
      if (!deny.has("max_output_tokens")) {
        const cap = req.max_completion_tokens ?? req.max_tokens;
        if (cap !== undefined) extra.max_output_tokens = cap;
      }
      return extra;
    })(),
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
      return {
        reasoning: {
          effort,
          summary: "auto" as const,
          // Codex path requires `context: "all_turns"`, except the spark
          // family which 400s on it (accepts only `auto`/`current_turn`).
          ...(options.codexInstructions !== false &&
          !isCodexSparkModel(options.providerModelId)
            ? { context: "all_turns" as const }
            : {}),
        },
      };
    })(),
  };
};
