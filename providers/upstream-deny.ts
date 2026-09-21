/**
 * Open-by-default param policy for upstream request bodies.
 *
 * Stage 1 wires Axis 1 (per-encoder deny) + Axis 2 (provider overlay) for the
 * ChatGPT Responses encoder only. Compose:
 *
 *   effective_deny = (WIRE_DENY[wire] \ provider.allowParams)
 *                    ∪ provider.unsupportedParams
 *
 * `WIRE_DENY.chatgpt` is the complement of today's `toChatGptRequest` closed
 * allow-list — real Codex (`codexInstructions !== false`) keeps that full
 * drop set so the Responses body stays byte-identical. Grok shares the same
 * encoder but the Grok proxy accepts a subset of those keys; `allowParams`
 * re-admits only the live-verified ones.
 *
 * Evidence (2026-07-14, `cli-chat-proxy.grok.com/v1/responses`):
 * temperature / top_p / max_output_tokens accepted; response_format accepted
 * but runaway (deny); reasoning.context 400s (already omitted on grok path);
 * stop unprobed (deny). Codex 400s on max_output_tokens.
 *
 * Re-probe (2026-08-24, grok-4.5, full signed-307 daemon path): temperature and
 * top_p re-confirmed accepted. `max_output_tokens` is accepted but NOT enforced
 * (cap=16 returned 491 output tokens) — kept as harmless client-intent
 * passthrough, do NOT rely on it as a hard output cap on grok. See
 * docs/audit/2026-08-24-grok-responses-live-probe.md.
 */

import type { TModelCaps } from "@openllmsh/protocol";
import { applyModelCaps } from "../features/model-caps";

export type TParamWire = "anthropic" | "chatgpt" | "openai";

export type TWireDeny = Readonly<Record<TParamWire, readonly string[]>>;

export type TProviderParamPolicy = {
  readonly unsupportedParams?: readonly string[];
  readonly allowParams?: readonly string[];
  readonly aliases?: Readonly<Record<string, string>>;
  /**
   * Narrow replayed assistant messages to the shape THIS provider's documented
   * request contract accepts. Opt-in per provider, never per wire: the openai
   * WIRE is spoken by a dozen backends that accept (and sometimes require)
   * extra message fields, so a wire-wide rule would strip real vendor
   * extensions. See {@link sanitizeOpenAiAssistantHistory}.
   */
  readonly sanitizeAssistantHistory?: "openai";
};

/**
 * Put a replayed assistant message back into the shape api.openai.com's
 * REQUEST contract accepts (`ChatCompletionRequestAssistantMessage`, pinned
 * spec), for the one provider whose contract is exactly that.
 *
 * This exists only because the response decoder now PRESERVES `annotations`
 * and `audio` instead of discarding them. That preservation is the fix; this
 * is its outbound half. An agent that replays our own response verbatim — the
 * normal loop — must not start sending a body the endpoint rejects.
 *
 * The rule is BASELINE-PRESERVING: send what the baseline sent, plus at most
 * the one field the contract defines.
 *  - `annotations` is REMOVED. It has no request-side counterpart, the cited
 *    text is already in `content`, and the baseline sent nothing here.
 *  - `audio` is reduced to `{id}` when a usable string id exists — the id is
 *    the whole semantic payload on a request (it references the prior audio
 *    turn), and it is the only member the contract defines.
 *  - `audio` with NO usable id (`{}`, or a `{data, transcript}` fragment
 *    reassembled from a stream) is REMOVED, exactly as the baseline sent it.
 *    It is never rewritten to `audio: {}` — that would manufacture a shape
 *    neither side defines — and it is never forwarded as-is, which would
 *    inject input the baseline never sent. Unrepresentable history data
 *    becomes absent, which is what it already was.
 *
 * Everything else is left ALONE — `reasoning_content`, `reasoning_items`,
 * `server_search_calls`, `cache_control` and any vendor extension keep
 * flowing exactly as before. This is two named fields, not a purge, and it is
 * opt-in per provider so no other openai-wire backend is narrowed.
 */
const sanitizeOpenAiAssistantHistory = (
  body: Record<string, unknown>,
): Record<string, unknown> => {
  const messages = body.messages;
  if (!Array.isArray(messages)) return body;
  let changed = false;
  const next = messages.map((message) => {
    if (
      typeof message !== "object" ||
      message === null ||
      (message as { readonly role?: unknown }).role !== "assistant"
    ) {
      return message;
    }
    const m = message as Record<string, unknown>;
    const hasAnnotations = Object.hasOwn(m, "annotations");
    const hasAudio = Object.hasOwn(m, "audio");
    if (!hasAnnotations && !hasAudio) return message;
    const audio = m.audio;
    const audioId =
      typeof audio === "object" && audio !== null && !Array.isArray(audio)
        ? (audio as { readonly id?: unknown }).id
        : undefined;
    const keepAudio = typeof audioId === "string" && audioId.length > 0;
    const alreadyMinimal =
      !hasAnnotations && keepAudio && Object.keys(audio as object).length === 1;
    if (alreadyMinimal) return message;
    changed = true;
    const { annotations: _annotations, audio: _audio, ...rest } = m;
    return keepAudio ? { ...rest, audio: { id: audioId } } : rest;
  });
  return changed ? { ...body, messages: next } : body;
};

/** Params the Responses request struct / Codex allow-list does not carry. */
export const WIRE_DENY: TWireDeny = {
  chatgpt: [
    "temperature",
    "top_p",
    "top_k",
    "max_output_tokens",
    "response_format",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "stop",
    "user",
    "metadata",
  ],
  // Stage 2: Anthropic actually accepts temperature/top_p/top_k/stop_sequences.
  anthropic: [
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "n",
  ],
  openai: [],
};

export const PROVIDER_POLICY: Readonly<Record<string, TProviderParamPolicy>> = {
  grok: { allowParams: ["temperature", "top_p", "max_output_tokens"] },
  kimi_code: {
    aliases: {
      max_completion_tokens: "max_tokens",
    },
    unsupportedParams: ["functions"],
  },
  google: {
    unsupportedParams: ["top_k"],
  },
  /**
   * `top_k` is NOT a Chat Completions parameter: it is absent from
   * `CreateChatCompletionRequest` in the pinned MIT spec
   * (`tests/helpers/fixtures/openai-chat-surface.json`), and api.openai.com 400s on it
   * as an unrecognised request argument — so forwarding it turned an optional
   * sampling hint into a hard request failure for the whole hop.
   *
   * Dropped by PROVIDER POLICY rather than refused pre-dispatch, and scoped to
   * `openai` alone, for two reasons. First, consistency: `google` already
   * resolves the same parameter on the same wire this way, and a caller whose
   * chain spans both must not get a 400 from one hop and a served answer from
   * the other. Second, honesty is preserved without the 400 — `top_k` is in
   * `SIGNAL_FIELDS`, so a caller who sent it is told it was dropped on
   * `x-openllm-dropped-params` instead of being left to assume it applied.
   *
   * Crucially this is a PROVIDER overlay, never `WIRE_DENY.openai`: every
   * other openai-compatible upstream (Kimi, DashScope, Grok chat, custom
   * endpoints, local runtimes) documents `top_k` as a real vendor extension,
   * and stripping it wire-wide would remove a parameter those backends honour.
   */
  openai: {
    unsupportedParams: ["top_k"],
    // api.openai.com's request contract for a replayed assistant turn is
    // narrower than its response. Provider-scoped so no other openai-wire
    // backend is touched. See `sanitizeOpenAiAssistantHistory`.
    sanitizeAssistantHistory: "openai",
  },
};

export const applyProviderPolicy = (
  body: Record<string, unknown>,
  policy: TProviderParamPolicy | undefined,
): Record<string, unknown> => {
  const filtered = { ...body };
  const aliases = policy?.aliases;
  if (aliases !== undefined) {
    for (const [from, to] of Object.entries(aliases)) {
      if (Object.hasOwn(filtered, from)) {
        if (!Object.hasOwn(filtered, to)) {
          filtered[to] = filtered[from];
        }
        delete filtered[from];
      }
    }
  }

  for (const param of policy?.unsupportedParams ?? []) {
    if (Object.hasOwn(filtered, param)) {
      delete filtered[param];
    }
  }

  return policy?.sanitizeAssistantHistory === "openai"
    ? sanitizeOpenAiAssistantHistory(filtered)
    : filtered;
};

/** Apply the provider overlay and resolved model caps once the final wire body exists. */
export const finalizeUpstreamBody = (
  body: Record<string, unknown>,
  provider: string,
  caps: TModelCaps | undefined,
  wire?: TParamWire,
): Record<string, unknown> => {
  const policy = applyProviderPolicy(body, PROVIDER_POLICY[provider]);
  if (wire === undefined) {
    return applyModelCaps(policy, caps);
  }

  const denied = effectiveDeny(wire, PROVIDER_POLICY[provider]);
  const deniedByWire: Record<string, unknown> = { ...policy };
  for (const name of denied) {
    delete deniedByWire[name];
  }

  return applyModelCaps(deniedByWire, caps);
};

export const effectiveDeny = (
  wire: TParamWire,
  policy: TProviderParamPolicy | undefined,
): ReadonlySet<string> => {
  const allow = new Set(policy?.allowParams ?? []);
  const denied = new Set<string>();
  for (const name of WIRE_DENY[wire]) {
    if (!allow.has(name)) denied.add(name);
  }
  for (const name of policy?.unsupportedParams ?? []) {
    denied.add(name);
  }
  return denied;
};

/**
 * Canonical request fields whose loss is observable to Chat Completions clients.
 * `WIRE_DENY` covers explicit policy denial; this map covers intentional builder
 * omissions recorded in the chat-completions parameter-fate audit.
 */
const SIGNAL_FIELDS = [
  "response_format",
  "seed",
  "presence_penalty",
  "frequency_penalty",
  "logit_bias",
  "n",
  "metadata",
  "stream_options",
  "top_k",
  "parallel_tool_calls",
  // Chat-Completions parameters this gateway now DECODES (they used to vanish
  // at the decoder, so there was nothing to report). Decoding them is only
  // half the fix: a cross-wire hop still cannot carry most of them, and a
  // caller who asked for `modalities: ["audio"]` and got text back deserves to
  // be told which knob was not honoured rather than to infer it.
  "service_tier",
  "store",
  "verbosity",
  "safety_identifier",
  "prompt_cache_retention",
  "web_search_options",
  "modalities",
  "audio",
  "prediction",
] as const;

/**
 * Per-wire builder omissions. The cross-wire encoders build an explicit body
 * for their own API rather than forwarding canonical keys, so every field
 * listed here is knowingly absent from that upstream — verified against the
 * encoders themselves, not assumed.
 */
const CROSS_WIRE_CHAT_ONLY = [
  "service_tier",
  "verbosity",
  "safety_identifier",
  "prompt_cache_retention",
  "web_search_options",
  "modalities",
  "audio",
  "prediction",
] as const;

const WIRE_BUILDER_OMISSIONS: Readonly<
  Record<TParamWire, ReadonlySet<string>>
> = {
  anthropic: new Set([
    "response_format",
    "metadata",
    "stream_options",
    // `parallel_tool_calls` is NOT listed: the Anthropic encoder TRANSLATES it
    // (`false` → `tool_choice.disable_parallel_tool_use: true`, and `true` is
    // Anthropic's own default, so it is honoured by omission). Reporting a
    // translation as a drop is the same failure as reporting nothing —
    // it tells the caller their setting was ignored when it was obeyed.
    // `store` stays: the Anthropic body has no counterpart at all.
    "store",
    ...CROSS_WIRE_CHAT_ONLY,
  ]),
  // `toChatGptRequest` is a closed allow-list; it DOES emit `store` (pinned
  // `false`), so `store` is a translation rather than a drop and stays out.
  chatgpt: new Set([
    "stream_options",
    "n",
    "logit_bias",
    ...CROSS_WIRE_CHAT_ONLY,
  ]),
  openai: new Set(),
};

/**
 * Return signal fields the client supplied that this resolved upstream wire will
 * not carry. Renamed fields deliberately stay out: those are translations, not
 * drops. The result preserves `SIGNAL_FIELDS` order for a stable response header.
 */
export const droppedSignalParams = (
  request: Record<string, unknown>,
  wire: TParamWire,
  provider: string,
): ReadonlyArray<string> => {
  const denied = effectiveDeny(wire, PROVIDER_POLICY[provider]);
  const omitted = WIRE_BUILDER_OMISSIONS[wire];
  return SIGNAL_FIELDS.filter(
    (field): boolean =>
      Object.hasOwn(request, field) &&
      (denied.has(field) || omitted.has(field)),
  );
};

/**
 * Non-Codex Responses-wire hops (today: Grok) share `toChatGptRequest` and
 * signal via `codexInstructions: false`. Map that flag to the grok overlay
 * without an `if (provider === "grok")` branch.
 */
export const responsesWirePolicy = (
  codexInstructions: boolean | undefined,
): TProviderParamPolicy | undefined =>
  codexInstructions === false ? PROVIDER_POLICY.grok : undefined;
