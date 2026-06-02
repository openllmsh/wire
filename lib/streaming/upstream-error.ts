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
