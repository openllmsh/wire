import type { TChatCompletionResponse } from "@openllmsh/protocol";

/**
 * Thrown by provider streaming decoders when the upstream sends an
 * error event mid-stream (e.g. Anthropic's `event: error` for
 * overloaded_error / api_error). Surfaces through the canonical
 * `ReadableStream<TChatCompletionChunk>` so the surface-specific SSE
 * encoder can emit a trailing error frame instead of silently
 * truncating the response.
 */
export class UpstreamStreamError extends Error {
  readonly upstreamType: string;
  constructor(upstreamType: string, message: string) {
    super(message);
    this.name = "UpstreamStreamError";
    this.upstreamType = upstreamType;
  }
}

/**
 * A stream ended before a terminal chunk after its decoder had already observed
 * usage. The accumulator wraps every failure type, including transport errors,
 * so callers can retain that wire-observed usage without treating an incomplete
 * answer as successful.
 */
export class IncompleteStreamError extends Error {
  readonly cause: unknown;
  readonly usage: TChatCompletionResponse["usage"] | null;

  constructor(cause: unknown, usage: TChatCompletionResponse["usage"] | null) {
    super(cause instanceof Error ? cause.message : String(cause));
    this.name = "IncompleteStreamError";
    this.cause = cause;
    this.usage = usage;
  }
}

/** Return usage observed before an incomplete stream error, if any. */
export const partialUsageFrom = (
  err: unknown,
): TChatCompletionResponse["usage"] | null =>
  err instanceof IncompleteStreamError ? err.usage : null;

/** Recover the original decoder or transport error from an accumulator wrapper. */
export const streamFailureCause = (err: unknown): unknown =>
  err instanceof IncompleteStreamError ? err.cause : err;

export const upstreamErrorFrom = (
  err: unknown,
): { type: string; message: string } => {
  if (err instanceof UpstreamStreamError) {
    return { type: err.upstreamType, message: err.message };
  }
  if (err instanceof Error) {
    return { type: "stream_error", message: err.message };
  }
  return { type: "stream_error", message: String(err) };
};
