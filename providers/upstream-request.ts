import type {
  TAnthropicRequest,
  TChatCompletionRequest,
  TResponsesRequest,
} from "@openllmsh/protocol";
import { fromAnthropicMessagesRequest } from "../adapters/messages/request";
import { fromResponsesRequest } from "../adapters/responses";
import { requestHasImageContent } from "../lib/canonical/content-part";
import { normaliseAdaptiveThinking } from "./anthropic/adaptive-thinking";
import {
  ANTHROPIC_FILES_API_BETA,
  ANTHROPIC_OAUTH_BETA,
  deriveAnthropicBetaHeader,
} from "./anthropic/beta-headers";
import { toAnthropicRequest } from "./anthropic/request";
import { deriveChatGptSessionId, toChatGptRequest } from "./chatgpt/request";

/**
 * The SINGLE recipe for preparing an upstream provider request from an inbound
 * one — body + wire-derived headers — for every `(clientWire × upstreamWire)`
 * pairing. The cloud runner (`@openllm/core`) and the coreless daemon walker
 * (`@openllmsh/daemon`) BOTH call this; neither re-derives the recipe.
 *
 * This exists because the pure transforms were single-sourced in `@openllmsh/wire`
 * but their COMPOSITION was open-coded in core's runner AND the daemon's
 * walker — which can't share (the daemon is core-free) — and drifted, dropping
 * the client's `anthropic-beta` and skipping `normaliseAdaptiveThinking`. See
 * `docs/proposals/unified-upstream-request-builder.md`.
 *
 * Response decode (upstream → canonical) + encode (canonical → client) are
 * already shared wire concerns; this module is the REQUEST side.
 */

export type TClientSurface = "messages" | "chat_completions" | "responses";
export type TUpstreamWire = "anthropic" | "chatgpt" | "openai";

const DEFAULT_ANTHROPIC_VERSION = "2023-06-01";

/** The client's upstream wire, derived from the surface it hit. `responses`
 *  rides the OpenAI family for the passthrough decision (it never passes
 *  through — its body is Responses-shaped — so it always transforms). */
export const clientWireOf = (surface: TClientSurface): TUpstreamWire =>
  surface === "messages" ? "anthropic" : "openai";

/** Inbound (client-shaped) body → canonical ChatCompletion, per surface. */
export const canonicalFromInbound = (
  surface: TClientSurface,
  rawBody: unknown,
): TChatCompletionRequest =>
  surface === "messages"
    ? fromAnthropicMessagesRequest(rawBody as TAnthropicRequest)
    : surface === "responses"
      ? fromResponsesRequest(rawBody as TResponsesRequest)
      : (rawBody as TChatCompletionRequest);

/**
 * Canonical request → an UPSTREAM body, per upstream wire. The cross-wire half
 * of the recipe: `toChatGptRequest` / `toAnthropicRequest` / OpenAI-identity.
 * Exported because the daemon's web_search agentic loop rebuilds the upstream
 * body each round from an ALREADY-canonical request (not the inbound body).
 */
const inlineAnthropicSystemContent = (content: unknown): unknown[] =>
  typeof content === "string"
    ? [{ type: "text", text: content }]
    : Array.isArray(content)
      ? content
      : [];

/**
 * Mid-conversation inline `role: "system"` messages are only present in bodies
 * coming from Claude Code and are not universally accepted by Anthropic models.
 * The rewrite is therefore triggered by BODY SHAPE (presence of inline system
 * messages) and never by model id.
 *
 * Commit e6ac9b93 rewrote inline system messages unconditionally and destroyed
 * cache behavior (constant `cache_read`, 250k-285k `cache_creation`/request).
 * Commit 2f7357be then added a model gate, but this was still wrong because
 * acceptance is not monotonic by family/version.
 *
 * Verified live ground truth (2026-07-26, handrolled claude_code hop, verbatim
 * Claude Code preamble, hoist bypassed so this reads RAW model capability —
 * re-measure with `tests/server/inline-system-capability.e2e.test.ts`):
 * - accept: `claude-opus-5`, `claude-sonnet-5`, `claude-opus-4-8`,
 *   `claude-fable-5`
 * - reject: `claude-opus-4-7`, `claude-opus-4-1`, `claude-sonnet-4-6`,
 *   `claude-sonnet-4-5`, `claude-haiku-4-5`
 *
 * `claude-opus-4-1` rejects while `claude-opus-4-8` accepts — NON-MONOTONIC
 * within one family, which is exactly why no regex or list can work. The
 * `anthropic-beta: mid-conversation-system-2026-04-07` header makes no
 * difference either way.
 *
 * Two properties keep the prompt-cache invariant intact:
 *  1. No inline system message → the exact same object is returned (no
 *     rewrite, no re-serialisation, no key reordering).
 *  2. When the hoist DOES run, moved content is APPENDED AFTER the existing
 *     top-level `system` blocks, so the client's `cache_control` breakpoint
 *     keeps its index and the cached prefix boundary is undisturbed —
 *     appended content lands OUTSIDE the cached prefix. Measured live over
 *     3-turn conversations (sonnet-4-6, opus-4-8, fable-5), including a
 *     breakpoint on a NON-final system block: `cache_read` stays flat at the
 *     prefix size with `cache_creation` 0 after warm-up, on both paths.
 *     e6ac9b93 was catastrophic because it rewrote EVERY request — not
 *     because appending after a breakpoint is inherently fatal.
 *     Locked in by `tests/transport/cache-prefix-stability.test.ts`.
 */
/**
 * Claude Code can emit mid-conversation inline system messages. Anthropic models
 * accept equivalent top-level system prompts, so move inline system content into
 * `system` when present and remove inline system messages from `messages`. Keep
 * all moved blocks verbatim except for stripping inline `cache_control` fields.
 */
const stripInlineSystemCacheMarker = (block: unknown): unknown => {
  if (block === null || typeof block !== "object" || Array.isArray(block)) {
    return block;
  }

  const message = block as Record<string, unknown>;
  if (!Object.hasOwn(message, "cache_control")) {
    return block;
  }

  const { cache_control: _cacheControl, ...rest } = message;
  return rest;
};

const hoistInlineAnthropicSystemMessages = (
  body: Record<string, unknown>,
): Record<string, unknown> => {
  if (!Array.isArray(body.messages)) return body;

  const inlineSystemMessages = body.messages.filter(
    (message) =>
      typeof message === "object" &&
      message !== null &&
      (message as { readonly role?: unknown }).role === "system",
  );
  if (inlineSystemMessages.length === 0) return body;

  const existingSystem =
    typeof body.system === "string"
      ? [{ type: "text", text: body.system }]
      : Array.isArray(body.system)
        ? body.system
        : [];
  const hoistedSystem = inlineSystemMessages.flatMap((message) =>
    inlineAnthropicSystemContent(
      (message as { readonly content?: unknown }).content,
    ).map((contentBlock) => stripInlineSystemCacheMarker(contentBlock)),
  );

  return {
    ...body,
    system: [...existingSystem, ...hoistedSystem],
    messages: body.messages.filter(
      (message) =>
        !(
          typeof message === "object" &&
          message !== null &&
          (message as { readonly role?: unknown }).role === "system"
        ),
    ),
  };
};

export const canonicalToUpstreamBody = (
  upstreamWire: TUpstreamWire,
  canonical: TChatCompletionRequest,
  providerModelId: string,
  stream: boolean,
  // Whether the chatgpt (Responses) encode injects the Codex preamble. Undefined
  // → inject (Codex default); `false` suppresses it for non-Codex Responses-wire
  // providers (xAI Grok). No effect for the anthropic / openai wires.
  codexInstructions?: boolean,
): unknown => {
  if (upstreamWire === "chatgpt") {
    return toChatGptRequest(canonical, { providerModelId, codexInstructions });
  }
  const options = { providerModelId };
  if (upstreamWire === "anthropic") {
    return { ...toAnthropicRequest(canonical, options), stream };
  }
  // openai-identity passthrough: forward canonical verbatim, but DROP the
  // Responses-only opaque carriers — chatgpt re-emits them; openai-compatible
  // upstreams 400 on unknown keys.
  const {
    responses_tools: _responsesTools,
    responses_additional_tools: _responsesAdditionalTools,
    responses_client_metadata: _responsesClientMetadata,
    ...openai
  } = canonical;
  const streamOptions =
    stream === true
      ? { ...openai.stream_options, include_usage: true }
      : openai.stream_options;
  return {
    ...openai,
    ...(streamOptions !== undefined ? { stream_options: streamOptions } : {}),
    model: providerModelId,
    stream,
  };
};

/** Inbound (client-shaped) body → an upstream body, per `(surface,
 *  upstreamWire)`. Passthrough (+ adaptive-thinking normalise for anthropic)
 *  or cross-wire via canonical. Exported so the daemon's single-shot serve and
 *  the cloud runner share the exact body recipe. */
export const buildUpstreamBody = (
  surface: TClientSurface,
  upstreamWire: TUpstreamWire,
  rawBody: unknown,
  providerModelId: string,
  // `undefined` → PRESERVE the body's own stream flag (the cloud passthrough
  // forwards verbatim); a boolean pins it (the daemon, off the 307's intent).
  stream: boolean | undefined,
  // Codex-preamble injection for the chatgpt wire (see canonicalToUpstreamBody).
  codexInstructions?: boolean,
): unknown => {
  // No gateway prompt prefix is injected anywhere. The gateway forwards the
  // client's system prompt verbatim on EVERY hop — a gateway-injected prefix
  // both breaks prompt-cache prefix stability (any variance collapses the
  // shared prefix to a cache rebuild) and, on the subscription OAuth hop,
  // self-identifies the request as a multi-provider gateway (the shape
  // Anthropic's AUP "reverse engineering / duplicating model outputs"
  // safeguard blocks). Steering prefixes now live ONLY at the CLI/client
  // level (e.g. the `openllm claude` overlay), never in this chain.
  // Passthrough: same wire in + out (NEVER for `responses` — its body is
  // Responses-shaped). Only the model id + stream flag are pinned. The
  // Anthropic passthrough additionally normalises adaptive-thinking knobs
  // (`thinking:adaptive` / `output_config.effort` / top-level `effort`) for
  // the RESOLVED model — they 400 on haiku/claude-3.
  if (upstreamWire === clientWireOf(surface) && surface !== "responses") {
    // Pin the concrete model id + stream flag. The daemon always supplies a
    // resolved `providerModelId` (off the 307); the cloud passthrough (whose
    // body already carries the right model) passes the body's own model — and
    // an empty id means "preserve the body's model", so it stays a true
    // passthrough.
    const raw = rawBody as Record<string, unknown>;
    const effectiveStream = stream ?? raw.stream === true;
    const pinned = {
      ...raw,
      ...(upstreamWire === "openai" && effectiveStream
        ? {
            stream_options: {
              ...(raw.stream_options as Record<string, unknown> | undefined),
              include_usage: true,
            },
          }
        : {}),
      ...(providerModelId.length > 0 ? { model: providerModelId } : {}),
      ...(stream !== undefined ? { stream } : {}),
    };
    return upstreamWire === "anthropic"
      ? normaliseAdaptiveThinking(hoistInlineAnthropicSystemMessages(pinned))
      : pinned;
  }
  // Cross-wire: route through canonical, then encode to the upstream's wire.
  return canonicalToUpstreamBody(
    upstreamWire,
    canonicalFromInbound(surface, rawBody),
    providerModelId,
    stream ?? false,
    codexInstructions,
  );
};

/**
 * True when a cross-wire (canonical-shaped) body carries a `file` part
 * referencing an uploaded file — `toAnthropicRequest` encodes that to a
 * Files-API document source, which Anthropic only honours under the
 * Files API beta.
 */
const canonicalUsesFileIds = (canonical: TChatCompletionRequest): boolean =>
  canonical.messages.some(
    (m) =>
      typeof m.content !== "string" &&
      m.content != null &&
      m.content.some((p) => p.type === "file" && p.file.file_id !== undefined),
  );

/** Wire-derived headers (layered OVER the caller's auth/identity). Only the
 *  Anthropic upstream contributes here: version + the merged `anthropic-beta`
 *  (OAuth beta + the client's inbound betas + body-derived betas). */
const wireHeaders = (
  surface: TClientSurface,
  upstreamWire: TUpstreamWire,
  rawBody: unknown,
  inboundBeta: string | null,
  isOAuth: boolean,
  apiVersion: string | undefined,
): Record<string, string> => {
  if (upstreamWire !== "anthropic")
    return { "content-type": "application/json" };
  // Body-derived betas (web_search/web_fetch/files-api) only read the
  // Anthropic-shaped request; for a non-messages (cross-wire) request there's
  // no Anthropic-shaped request, so an empty stand-in yields no false derived
  // betas — the Files-API beta is instead derived from the canonical body's
  // `file_id` parts (which `toAnthropicRequest` encodes to file sources).
  const request =
    surface === "messages"
      ? (rawBody as TAnthropicRequest)
      : ({ model: "", messages: [], max_tokens: 0 } as TAnthropicRequest);
  const extraBetas =
    surface !== "messages" &&
    canonicalUsesFileIds(canonicalFromInbound(surface, rawBody))
      ? [ANTHROPIC_FILES_API_BETA]
      : [];
  const beta = deriveAnthropicBetaHeader({
    inboundBeta,
    request,
    isOAuth,
    extraBetas,
  });
  return {
    "anthropic-version": apiVersion ?? DEFAULT_ANTHROPIC_VERSION,
    "content-type": "application/json",
    ...(beta !== undefined
      ? { "anthropic-beta": beta }
      : isOAuth
        ? { "anthropic-beta": ANTHROPIC_OAUTH_BETA }
        : {}),
  };
};

export type TBuildUpstreamRequestInput = {
  readonly surface: TClientSurface;
  readonly upstreamWire: TUpstreamWire;
  /** The inbound body in the CLIENT's wire shape. */
  readonly rawBody: unknown;
  /** Concrete upstream model id to pin. */
  readonly providerModelId: string;
  /**
   * The client's stream intent. Undefined preserves the inbound body's stream
   * flag for same-wire passthrough requests.
   */
  readonly stream: boolean | undefined;
  /**
   * Auth + identity the CALLER owns — BYOK `x-api-key` / OAuth `authorization`
   * (cloud), or the local CLI's `authorization` + vendor identity headers
   * (daemon). Wire-derived headers (anthropic-version / anthropic-beta /
   * content-type) are layered ON TOP so they win on collision.
   */
  readonly baseHeaders: Record<string, string>;
  /** The client's inbound `anthropic-beta` header (or null). */
  readonly inboundBeta?: string | null;
  /** True when the Anthropic auth is a subscription OAuth token. */
  readonly isOAuth?: boolean;
  /** Pinned `anthropic-version` (else the ground-floor default). */
  readonly apiVersion?: string;
  /**
   * chatgpt-wire only: inject the Codex preamble. Undefined → inject (Codex
   * default); `false` suppresses it for a non-Codex Responses-wire provider
   * (xAI Grok). See {@link canonicalToUpstreamBody}.
   */
  readonly codexInstructions?: boolean;
  /**
   * Catalog capabilities for the resolved hop. Empty / missing = unknown
   * (custom / passthrough) — never treated as non-vision. The vision
   * gate rejects only when this is NON-empty and lacks `"vision"`.
   */
  readonly capabilities?: ReadonlyArray<string>;
};

/**
 * The caller's auth/identity (`baseHeaders`) with the wire-derived headers
 * (anthropic-version / merged anthropic-beta / content-type) layered ON TOP so
 * they win on collision. Exported so the daemon's web_search loop (which
 * rebuilds the body per round but keeps one header set) can compute headers
 * once, independently of the body.
 */
export const buildUpstreamHeaders = (
  i: TBuildUpstreamRequestInput,
): Record<string, string> => {
  const headers: Record<string, string> = {
    ...i.baseHeaders,
    ...wireHeaders(
      i.surface,
      i.upstreamWire,
      i.rawBody,
      i.inboundBeta ?? null,
      i.isOAuth ?? false,
      i.apiVersion,
    ),
  };
  if (i.upstreamWire === "chatgpt") {
    ensureChatGptSessionAffinity(headers, i.surface, i.rawBody);
  }
  return headers;
};

/**
 * Guarantee a STABLE `session_id` on the ChatGPT Codex wire (chatgpt + grok).
 * The backend routes prompt-cache affinity by `session_id`; with none, every
 * request lands on a cold machine and caches nothing (`cached=0`), re-billing
 * the full conversation input every turn — a subscription drain for agentic
 * coding (audit 2026-07-14-codex-handrolled-quota-drain). We PRESERVE the
 * client's own session (a real Codex CLI already sends one, in either
 * `session_id` or codex-rs `session-id` form) and only SYNTHESIZE a stable
 * per-conversation one — derived from the immutable conversation prefix — when
 * the client sent none, so bare clients (chat-completions, custom agents) also
 * get cache affinity. `x-client-request-id` rides along for vendor-client
 * parity (caching keys on `session_id`, not this).
 */
const ensureChatGptSessionAffinity = (
  headers: Record<string, string>,
  surface: TClientSurface,
  rawBody: unknown,
): void => {
  const hasSession = Object.keys(headers).some((k) => {
    const lk = k.toLowerCase();
    return lk === "session_id" || lk === "session-id";
  });
  if (hasSession) return;
  const sessionId = deriveChatGptSessionId(
    canonicalFromInbound(surface, rawBody),
  );
  headers.session_id = sessionId;
  headers["x-client-request-id"] = sessionId;
};

/**
 * Known-non-vision: catalog hit with a NON-empty capability set that
 * does not include `"vision"`. Empty / missing capabilities are UNKNOWN
 * (custom / passthrough) and must never be blocked.
 */
const isKnownNonVision = (
  capabilities: ReadonlyArray<string> | undefined,
): boolean =>
  capabilities !== undefined &&
  capabilities.length > 0 &&
  !capabilities.includes("vision");

const inboundHasImageContent = (
  surface: TClientSurface,
  rawBody: unknown,
): boolean => {
  if (surface === "chat_completions") {
    if (
      typeof rawBody !== "object" ||
      rawBody === null ||
      !("messages" in rawBody) ||
      !Array.isArray((rawBody as { messages: unknown }).messages)
    ) {
      return false;
    }
    return requestHasImageContent(rawBody as TChatCompletionRequest);
  }
  try {
    return requestHasImageContent(canonicalFromInbound(surface, rawBody));
  } catch {
    return false;
  }
};

/**
 * Thrown when a known-non-vision hop would otherwise leak `image_url`
 * to a text-only OpenAI-compat backend. `type` + `name` are both the
 * `unsupported_content` tag the tests (and callers) match.
 */
export class UnsupportedContentError extends Error {
  readonly type = "unsupported_content";
  constructor(providerModelId: string) {
    super(
      `unsupported_content: model "${providerModelId}" does not support image content`,
    );
    this.name = "unsupported_content";
  }
}

/**
 * Prepare the `{ body, headers }` for ONE upstream call. The only place the
 * `(clientWire × upstreamWire)` request recipe lives.
 */
export const buildUpstreamRequest = (
  i: TBuildUpstreamRequestInput,
): { readonly body: unknown; readonly headers: Record<string, string> } => {
  if (
    isKnownNonVision(i.capabilities) &&
    inboundHasImageContent(i.surface, i.rawBody)
  ) {
    throw new UnsupportedContentError(i.providerModelId);
  }
  return {
    body: buildUpstreamBody(
      i.surface,
      i.upstreamWire,
      i.rawBody,
      i.providerModelId,
      i.stream,
      i.codexInstructions,
    ),
    headers: buildUpstreamHeaders(i),
  };
};
