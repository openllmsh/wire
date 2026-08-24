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

  return filtered;
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
 * Non-Codex Responses-wire hops (today: Grok) share `toChatGptRequest` and
 * signal via `codexInstructions: false`. Map that flag to the grok overlay
 * without an `if (provider === "grok")` branch.
 */
export const responsesWirePolicy = (
  codexInstructions: boolean | undefined,
): TProviderParamPolicy | undefined =>
  codexInstructions === false ? PROVIDER_POLICY.grok : undefined;
