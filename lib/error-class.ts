import type { TErrorEnvelope } from "@openllmsh/protocol";

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
 */
export type TFallbackClass =
  | { readonly kind: "transient"; readonly reason: TFallbackReason }
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
  | "upstream_rejection";

export type TClassifierInput = {
  readonly status: number;
  readonly envelope?: TErrorEnvelope | undefined;
  readonly providerFormat: "openai" | "anthropic";
  readonly aborted: boolean;
};

const reasonForStatus = (i: TClassifierInput): TFallbackReason => {
  if (i.envelope?.error?.code === "content_filter") return "content_filter";
  if (i.status === 0) return "network";
  if (i.status === 408) return "timeout";
  if (i.status === 429) return "rate_limit";
  if (i.status >= 500) return "server_error";
  if (i.status === 401 || i.status === 403 || i.status === 407) return "auth";
  if (i.status === 402) return "payment";
  if (i.status === 404) return "not_found";
  if (i.status === 413) return "payload_too_large";
  if (i.status === 422) return "unprocessable";
  return "upstream_rejection";
};

export const classifyHopError = (i: TClassifierInput): TFallbackClass =>
  i.aborted
    ? { kind: "abort" }
    : { kind: "transient", reason: reasonForStatus(i) };
