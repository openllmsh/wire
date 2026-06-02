/**
 * Coreless upstream-SSE decode driver — the `@openllm/core`-free analogue
 * of `core/lib/streaming/event-stream.ts`'s `providerEventStream`. The
 * coreless daemon walker uses it to turn an upstream provider's raw SSE
 * byte stream into canonical `TChatCompletionChunk`s, then re-encodes
 * those to the client's wire.
 *
 * `core`'s version takes a whole `TChatProviderSpec`; this takes just the
 * three streaming pieces (`eventSchema` + `initialState` + `eventToChunk`)
 * so it has no dependency on `core`'s spec machinery — only `effect`'s
 * `Schema` (for decode) + `@openllm/schema` + the wire SSE primitives.
 */
import type { TChatCompletionChunk } from "@openllm/schema";
import { Schema } from "effect";
import { encodeSseDone, encodeSseEvent, sseEventStream } from "./sse";
import { upstreamErrorFrom } from "./upstream-error";

export type TStreamDecodeSpec<TEvent, TState, TOpts> = {
  readonly eventSchema: Schema.Schema<TEvent>;
  readonly initialState: (options: TOpts) => TState;
  readonly eventToChunk: (
    event: TEvent,
    state: TState,
    options: TOpts,
  ) => TChatCompletionChunk | null;
};

/**
 * Decode an upstream provider SSE stream into canonical chunks. Mirrors
 * `core`'s `providerEventStream` exactly, including the DashScope
 * `choices: []` recovery (harmless for non-OpenAI event schemas — it only
 * runs on the already-failed decode path).
 */
export const decodeProviderEventStream = <TEvent, TState, TOpts>(
  raw: ReadableStream<Uint8Array>,
  spec: TStreamDecodeSpec<TEvent, TState, TOpts>,
  options: TOpts,
): ReadableStream<TChatCompletionChunk> => {
  const state = spec.initialState(options);
  const events = sseEventStream(raw);
  const decode = Schema.decodeUnknownSync(spec.eventSchema);
  const reader = events.getReader();
  return new ReadableStream<TChatCompletionChunk>({
    async pull(controller) {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) {
          controller.close();
          return;
        }
        if (value.kind !== "data") continue;
        let parsedJson: unknown;
        try {
          parsedJson = JSON.parse(value.data);
        } catch {
          continue;
        }
        let event: TEvent;
        try {
          event = decode(parsedJson);
        } catch (firstErr) {
          let recovered: TEvent | null = null;
          if (
            parsedJson !== null &&
            typeof parsedJson === "object" &&
            !("choices" in parsedJson)
          ) {
            try {
              recovered = decode({ ...parsedJson, choices: [] });
            } catch {
              recovered = null;
            }
          }
          if (recovered === null) {
            if (process.env.OPENLLM_DEBUG_STREAM === "1") {
              console.warn(
                "[decodeProviderEventStream] dropped undecodable SSE chunk:",
                firstErr instanceof Error ? firstErr.message : String(firstErr),
                value.data.slice(0, 600),
              );
            }
            continue;
          }
          event = recovered;
        }
        const chunk = spec.eventToChunk(event, state, options);
        if (chunk !== null) {
          controller.enqueue(chunk);
          return;
        }
      }
    },
    cancel(reason) {
      reader.cancel(reason).catch(() => {});
    },
  });
};

/**
 * Encode canonical chunks as an OpenAI-wire SSE byte stream (the client
 * encoder for the chat-completions surface). Verbatim port of `core`'s
 * `chunksToSseBytes`; on a mid-stream upstream error it emits a final
 * error frame + `[DONE]` so the client sees a clean failure.
 */
export const chunksToSseBytes = (
  chunks: ReadableStream<TChatCompletionChunk>,
): ReadableStream<Uint8Array> => {
  const reader = chunks.getReader();
  let doneSent = false;
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { value, done } = await reader.read();
        if (done) {
          if (!doneSent) {
            controller.enqueue(encodeSseDone());
            doneSent = true;
          }
          controller.close();
          return;
        }
        controller.enqueue(encodeSseEvent(value));
      } catch (err) {
        const { type, message } = upstreamErrorFrom(err);
        controller.enqueue(encodeSseEvent({ error: { type, message } }));
        if (!doneSent) {
          controller.enqueue(encodeSseDone());
          doneSent = true;
        }
        controller.close();
      }
    },
    cancel(reason) {
      reader.cancel(reason).catch(() => {});
    },
  });
};
