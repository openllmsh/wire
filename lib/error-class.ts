import type { TCooldownReason, TErrorEnvelope } from "@openllmsh/protocol";

/**
 * Classifies an uncommitted upstream hop failure. Lives in
 * `@openllmsh/wire` so the cloud orchestrator and coreless daemon walker
 * share one policy.
 *
 * Once the gateway has validated the inbound request, an upstream rejection
 * cannot prove that every other provider/model will reject the canonical
 * request. Therefore every non-abort failure walks the chain. This decision is
 * deliberately independent of provider names, error prose, and status-code
 * allow-lists; `reason` is only a stable observability/cooldown tag.
 *
 * A committed response never reaches this classifier. After output has been
 * committed, the serving transport owns any failure and cannot restart on a
 * different model safely.
 *
 * Some Envoy edge-connect failures wear a 5xx status despite describing a
 * network failure between the edge and its upstream. The narrow conjunction
 * below keeps those failures from cooling a model, while ordinary vendor 5xx
 * responses remain `server_error` and retain their cooldown.
 */
export type TFallbackClass =
  | { readonly kind: "transient"; readonly reason: TCooldownReason }
  | { readonly kind: "abort" };

export type TClassifierInput = {
  readonly status: number;
  readonly envelope?: TErrorEnvelope | undefined;
  readonly providerFormat: "openai" | "anthropic";
  readonly aborted: boolean;
};

const QUOTA_BODY =
  /usage limit|usage balance|balance exhausted|quota|billing cycle|upgrade your plan|out of credits|purchase extra usage|insufficient_quota/i;
/**
 * The vendor error phrasings that mean "request too large for this model's
 * context window". Exported so callers that classify a raw upstream body — e.g.
 * the daemon walker's last-resort compaction trigger — reuse the SAME matcher
 * instead of maintaining a drifting copy.
 */
export const CONTEXT_OVERFLOW_BODY =
  /maximum (prompt|context) length|exceeds the context window|too many tokens|reduce the length of the messages|prompt is too long|exceeds? (?:the )?model token limit|token limit: [\d,]+/i;
// Envoy's first sentence is version-invariant; the trailing reset-reason
// clause drifts between Envoy versions ("reset reason: …" vs "retried and the
// latest reset reason: …"), so the matcher pins only the first sentence.
const ENVOY_CONNECT_ERROR_BODY = [
  /upstream connect error/i,
  /disconnect\/reset before headers/i,
] as const;

const isEnvoyConnectError = (message: string): boolean =>
  ENVOY_CONNECT_ERROR_BODY.every((pattern) => pattern.test(message));

const reasonForStatus = (i: TClassifierInput): TCooldownReason => {
  if (i.envelope?.error?.code === "content_filter") return "content_filter";
  const message = i.envelope?.error?.message ?? "";
  if (CONTEXT_OVERFLOW_BODY.test(message)) return "context_overflow";
  if (i.status === 0) return "network";
  if (i.status === 408) return "timeout";
  if (i.status === 429) {
    return QUOTA_BODY.test(message) ? "quota_exhausted" : "rate_limit";
  }
  if (i.status >= 500) {
    return isEnvoyConnectError(message) ? "network" : "server_error";
  }
  if (i.status === 401 || i.status === 403 || i.status === 407) {
    return QUOTA_BODY.test(message) ? "quota_exhausted" : "auth";
  }
  if (i.status === 402) {
    return QUOTA_BODY.test(message) ? "quota_exhausted" : "payment";
  }
  if (i.status === 404) return "not_found";
  if (i.status === 413) return "payload_too_large";
  if (i.status === 422) return "unprocessable";
  return "upstream_rejection";
};

export const classifyHopError = (i: TClassifierInput): TFallbackClass =>
  i.aborted
    ? { kind: "abort" }
    : { kind: "transient", reason: reasonForStatus(i) };
