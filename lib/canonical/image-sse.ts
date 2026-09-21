import type {
  TImageStreamCompletedEvent,
  TImageStreamEvent,
  TImageStreamPartialEvent,
} from "@openllmsh/protocol";
import { sseEventStream } from "../streaming/sse";

/**
 * Canonical `/v1/images/generations` SSE.
 *
 * THE RULE THIS FILE EXISTS TO ENFORCE: the gateway always serves an honest
 * stream, and never a simulated one. Concretely:
 *
 *  - A provider that genuinely emits progressive frames has them forwarded as
 *    they arrive, in order, unbuffered.
 *  - A provider that returns one finished image (Gemini, Grok, any
 *    OpenAI-compatible endpoint that ignores `stream`) yields exactly ONE
 *    `image_generation.completed` event and ZERO partials. We never chop a
 *    buffered final image into fake `partial_image` frames, and we never
 *    invent a `partial_image_index`.
 *  - Keepalive is an SSE COMMENT line, never a zero-progress event, so it
 *    cannot be mistaken for progress by any client.
 *
 * Framing follows the pinned spec: NAMED events (`event: image_generation.…`)
 * and NO `[DONE]` sentinel — `[DONE]` is chat framing and does not appear in
 * the images spec. The `completed` event is the terminator.
 */

const encoder = new TextEncoder();

/** One named SSE frame. The spec frames image events by name, not by `data` alone. */
export const encodeImageSseEvent = (event: TImageStreamEvent): Uint8Array =>
  encoder.encode(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);

/**
 * A terminal error frame. Used when generation SUCCEEDED but the gateway
 * could not durably store the image: the caller must not be told the
 * operation completed when the media-library URL it implies does not exist.
 */
export const encodeImageSseError = (message: string): Uint8Array =>
  encoder.encode(
    `event: error\ndata: ${JSON.stringify({ error: { message, type: "persistence_error" } })}\n\n`,
  );

/**
 * An inert SSE COMMENT. A comment can never be parsed as an event by any
 * client, which is exactly why it is the only honest vehicle for something
 * that is not a result: a keepalive, or a note explaining what the stream
 * will NOT contain. Never use an event frame for either — that is how a
 * zero-progress frame ends up masquerading as progress.
 */
export const encodeImageSseComment = (text: string): Uint8Array =>
  encoder.encode(`: ${text}\n\n`);

/** Liveness only, for a stream opened before the work completes. */
export const encodeImageSseKeepalive = (): Uint8Array =>
  encodeImageSseComment("keepalive");

/**
 * Told to the caller when they asked for partial frames the resolved model is
 * KNOWN not to emit. Without it the request silently yields one completion
 * and the caller is left to guess whether partials were lost or never
 * existed. Emitted only on a positive catalog fact
 * (`nativeProgressiveImages === false`) — never on absent/unknown caps, which
 * must stay permissive.
 */
export const encodeImageSsePartialsUnavailable = (): Uint8Array =>
  encodeImageSseComment(
    "partial_images requested, but this model does not emit progressive image frames; a single completion event will follow",
  );

const isPartial = (v: unknown): v is TImageStreamPartialEvent =>
  typeof v === "object" &&
  v !== null &&
  (v as { type?: unknown }).type === "image_generation.partial_image";

const isCompleted = (v: unknown): v is TImageStreamCompletedEvent =>
  typeof v === "object" &&
  v !== null &&
  (v as { type?: unknown }).type === "image_generation.completed";

export type TUpstreamImageEvents = {
  /** Genuine progressive frames, in arrival order. Empty for a final-only provider. */
  readonly partials: ReadonlyArray<TImageStreamPartialEvent>;
  readonly completed: TImageStreamCompletedEvent | null;
};

/**
 * Read an upstream SSE image stream into its events.
 *
 * Callers that need true progressive forwarding use {@link readImageSse}
 * instead; this collector exists for the non-progressive case and for tests.
 */
export const collectImageSse = async (
  upstream: Response,
): Promise<TUpstreamImageEvents> => {
  const partials: TImageStreamPartialEvent[] = [];
  let completed: TImageStreamCompletedEvent | null = null;
  for await (const event of readImageSse(upstream)) {
    if (isPartial(event)) partials.push(event);
    else if (isCompleted(event)) completed = event;
  }
  return { partials, completed };
};

/**
 * Yield upstream image events as they arrive. Unknown event types are skipped
 * rather than failing the stream — a provider adding an event we don't model
 * must not break a generation that otherwise succeeded.
 */
export async function* readImageSse(
  upstream: Response,
): AsyncGenerator<TImageStreamEvent> {
  const body = upstream.body;
  if (body === null) return;
  const reader = sseEventStream(body).getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      if (value === undefined || value.kind !== "data") continue;
      let parsed: unknown;
      try {
        parsed = JSON.parse(value.data);
      } catch {
        continue;
      }
      if (isPartial(parsed) || isCompleted(parsed)) yield parsed;
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Build a `completed` event for a provider that returned a finished image
 * rather than a stream. Every field carries a REAL value from that response —
 * notably `usage`, which is preserved rather than zeroed. Nothing here
 * fabricates progress: the absence of partials IS the honest signal that the
 * upstream was not progressive.
 */
export const completedEventFromFinalImage = (input: {
  readonly b64_json: string;
  readonly createdAt: number;
  readonly size?: string;
  readonly quality?: string;
  readonly background?: string;
  readonly outputFormat?: string;
  readonly usage?: TImageStreamCompletedEvent["usage"];
}): TImageStreamCompletedEvent => ({
  type: "image_generation.completed",
  b64_json: input.b64_json,
  created_at: input.createdAt,
  ...(input.size !== undefined ? { size: input.size } : {}),
  ...(input.quality !== undefined ? { quality: input.quality } : {}),
  ...(input.background !== undefined ? { background: input.background } : {}),
  ...(input.outputFormat !== undefined
    ? { output_format: input.outputFormat }
    : {}),
  ...(input.usage !== undefined ? { usage: input.usage } : {}),
});

/**
 * Collapse an image SSE stream into the JSON body a non-streaming consumer
 * expects.
 *
 * Intended for a caller WITHOUT an SSE reader. The previous approach silently
 * rewrote such a request to `stream: false`, which overrode a choice the
 * caller had made and meant an agent asking to stream got something else with
 * no indication. Aggregating instead honours the request end to end: the HTTP
 * surface still streams and progressive frames still reach anyone reading
 * them, while the tool consumer receives a final result it can render.
 *
 * Partials are DISCARDED rather than concatenated: they are successive
 * previews of one image, not fragments of it. Returns null when the stream
 * carried no completion.
 *
 * ⚠ SINGLE-IMAGE ONLY, AND CURRENTLY UNUSED. It keeps the LAST completion, so
 * an `n > 1` response — which emits one completion PER image — collapses to
 * the final one and the rest are lost. No production caller reaches it today:
 * the MCP tool deliberately implements its own `aggregateImageSseText` in
 * `packages/cli/src/mcp/openllm/tools.ts`, which keeps EVERY completion.
 * Before wiring this into any consumer that can request `n > 1`, make it
 * return all completions rather than adopting it as-is.
 */
export const aggregateImageSse = async (
  upstream: Response,
): Promise<TImageStreamCompletedEvent | null> => {
  let last: TImageStreamCompletedEvent | null = null;
  for await (const event of readImageSse(upstream)) {
    if (isCompleted(event)) last = event;
  }
  return last;
};

/** The aggregated completion as an `ImagesResponse`-shaped JSON body. */
export const imageSseToJsonBody = (
  completed: TImageStreamCompletedEvent,
): Record<string, unknown> => ({
  created: completed.created_at,
  data: [
    {
      b64_json: completed.b64_json,
      ...(completed.url !== undefined ? { url: completed.url } : {}),
    },
  ],
  ...(completed.usage !== undefined ? { usage: completed.usage } : {}),
  ...(completed.size !== undefined ? { size: completed.size } : {}),
  ...(completed.quality !== undefined ? { quality: completed.quality } : {}),
  ...(completed.background !== undefined
    ? { background: completed.background }
    : {}),
  ...(completed.output_format !== undefined
    ? { output_format: completed.output_format }
    : {}),
});
