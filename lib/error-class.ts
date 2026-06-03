import type { TErrorEnvelope } from "@openllm/schema";

/**
 * Per-hop upstream-error classification. Lives in `@openllm/wire` (pure,
 * schema-only) so the cloud orchestrator (`@openllm/core`) and the coreless
 * daemon walker share ONE definition of "transient" instead of forking it.
 *
 * - `transient` — retrying THIS model with THIS input would still fail, but a
 *   different model on the chain might succeed. The chain walks. Covers
 *   content/policy filters, capability refusals, context length,
 *   model-not-found, entitlement, auth, rate-limit, and transport faults.
 * - `client_bug` — the inbound request itself is malformed; every provider
 *   will reject it. Surface immediately; do NOT fan the error across the chain.
 * - `abort` — the inbound `request.signal` fired. Surface immediately.
 *
 * `reason` is a stable short tag suitable for `requests.hop_log`; it never
 * carries the raw provider message.
 *
 * NOTE: the cloud uses the full classifier for its chain policy; the daemon
 * keeps its own deliberately narrower status policy (4xx terminal except
 * 408/429/5xx — see `packages/daemon/src/walker.ts` `retryable`) and only
 * consults this classifier to recognise a transient 400 envelope.
 */
export type TFallbackClass =
  | { readonly kind: "transient"; readonly reason: TFallbackReason }
  | { readonly kind: "client_bug"; readonly reason: TFallbackReason }
  | { readonly kind: "abort" };

export type TFallbackReason =
  | "network"
  | "timeout"
  | "rate_limit"
  | "auth"
  | "payment"
  | "server_error"
  | "not_found"
  | "payload_too_large"
  | "unprocessable"
  | "content_filter"
  | "context_length"
  | "capability"
  | "entitlement"
  | "malformed_request"
  | "legal_block"
  | "unknown_4xx";

export type TClassifierInput = {
  readonly status: number;
  readonly envelope?: TErrorEnvelope | undefined;
  readonly providerFormat: "openai" | "anthropic";
  readonly aborted: boolean;
};

// Provider 400 messages whose substrings we read as "transient" rather
// than "client_bug". Kept loose on purpose — false positives walk the
// chain (one extra hop's spend); false negatives surface a content
// filter to the client (the bug the new policy is supposed to fix).
const TRANSIENT_400_PATTERN =
  /content|policy|safety|filter|moderation|context length|too long|maximum context|tool|function|vision|image|unsupported/i;

const CONTENT_FILTER_PATTERN = /content|policy|safety|filter|moderation/i;
const CONTEXT_LENGTH_PATTERN = /context length|too long|maximum context/i;
const CAPABILITY_PATTERN = /tool|function|vision|image|unsupported/i;

const reasonFor400 = (message: string): TFallbackReason => {
  if (CONTENT_FILTER_PATTERN.test(message)) return "content_filter";
  if (CONTEXT_LENGTH_PATTERN.test(message)) return "context_length";
  if (CAPABILITY_PATTERN.test(message)) return "capability";
  return "capability";
};

export const classifyHopError = (i: TClassifierInput): TFallbackClass => {
  if (i.aborted) return { kind: "abort" };

  // In-band refusal carries `code:"content_filter"` regardless of the
  // upstream HTTP status (the runner may synthesize a 200 + refusal
  // envelope when the provider returned 200 OK with
  // `finish_reason:"content_filter"`). Catch it before the status
  // branches so the synthesized 200 path also walks.
  if (i.envelope?.error?.code === "content_filter") {
    return { kind: "transient", reason: "content_filter" };
  }

  const status = i.status;

  if (status === 0) {
    return { kind: "transient", reason: "network" };
  }
  if (status === 408) {
    return { kind: "transient", reason: "timeout" };
  }
  if (status === 429) {
    return { kind: "transient", reason: "rate_limit" };
  }
  if (status >= 500) {
    return { kind: "transient", reason: "server_error" };
  }
  if (status === 401 || status === 403 || status === 407) {
    return { kind: "transient", reason: "auth" };
  }
  if (status === 402) {
    return { kind: "transient", reason: "payment" };
  }
  if (status === 404) {
    return { kind: "transient", reason: "not_found" };
  }
  if (status === 413) {
    return { kind: "transient", reason: "payload_too_large" };
  }
  if (status === 422) {
    return { kind: "transient", reason: "unprocessable" };
  }
  if (status === 451) {
    return { kind: "client_bug", reason: "legal_block" };
  }
  if (status === 400) {
    const err = i.envelope?.error;
    const message = err?.message ?? "";
    const code = err?.code ?? "";
    const type = err?.type ?? "";

    if (i.providerFormat === "openai" && code === "content_filter") {
      return { kind: "transient", reason: "content_filter" };
    }

    if (
      type === "invalid_request_error" &&
      TRANSIENT_400_PATTERN.test(message)
    ) {
      return { kind: "transient", reason: reasonFor400(message) };
    }

    return { kind: "client_bug", reason: "malformed_request" };
  }

  return { kind: "client_bug", reason: "unknown_4xx" };
};
