/**
 * Coreless upstream-SSE decode driver — the `@openllm/core`-free analogue
 * of `core/lib/streaming/event-stream.ts`'s `providerEventStream`. The
 * coreless daemon walker uses it to turn an upstream provider's raw SSE
 * byte stream into canonical `TChatCompletionChunk`s, then re-encodes
 * those to the client's wire.
 *
 * `core`'s version takes a whole `TChatProviderSpec`; this takes just the
 * streaming pieces (`eventSchema` + `initialState` + `eventToChunk` +
 * optional `isTerminalEvent`) so it has no dependency on `core`'s spec
 * machinery — only `effect`'s `Schema` (for decode) + `@openllmsh/protocol`
 * + the wire SSE primitives.
 */
import type { TChatCompletionChunk } from "@openllmsh/protocol";
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
  /**
   * Optional semantic-end predicate. When it returns true for a decoded
   * event, the driver converts the event, closes the canonical stream,
   * then cancels the source without awaiting transport EOF. Providers
   * without this keep reading until EOF so usage trailers can still
   * arrive (generic OpenAI-compat).
   */
  readonly isTerminalEvent?: (event: TEvent) => boolean;
};

/**
 * Decode an upstream provider SSE stream into canonical chunks. Mirrors
 * `core`'s `providerEventStream` exactly, including the DashScope
 * `choices: []` recovery (harmless for non-OpenAI event schemas — it only
 * runs on the already-failed decode path).
 *
 * Alibaba DashScope (and other OpenAI-compatible upstreams) OMIT the
 * `choices` key entirely on the trailing `stream_options.include_usage`
 * chunk — spec-correct providers send `choices: []`. The OpenAI chunk
 * schema makes `choices` required, so that chunk fails decode. Since it
 * is the ONLY chunk carrying token counts, a silent drop means zero
 * usage for Alibaba (no live feed, no final total) while spec-correct
 * providers work. Retry once with an empty `choices` filled in. (For
 * non-OpenAI event schemas the extra key is ignored on decode, so this
 * is provider-safe and only ever runs on the already-failed path.)
 *
 * A genuinely undecodable frame is still dropped so one odd chunk can't
 * kill the stream — but surface it under a debug flag so the next
 * provider divergence isn't invisible.
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
  let settled = false;

  const initiateCancel = (reason?: unknown): void => {
    void reader.cancel(reason).catch(() => {});
  };

  const settleClose = (
    controller: ReadableStreamDefaultController<TChatCompletionChunk>,
  ): void => {
    if (settled) return;
    settled = true;
    controller.close();
    initiateCancel("semantic-terminal");
  };

  const settleError = (
    controller: ReadableStreamDefaultController<TChatCompletionChunk>,
    err: unknown,
  ): void => {
    if (settled) return;
    settled = true;
    controller.error(err);
    initiateCancel(err);
  };

  return new ReadableStream<TChatCompletionChunk>({
    async pull(controller) {
      if (settled) {
        controller.close();
        return;
      }
      for (;;) {
        try {
          const read = await reader.read();
          if (settled) return;
          if (read.done) {
            settled = true;
            controller.close();
            return;
          }
          const value = read.value;
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
                  firstErr instanceof Error
                    ? firstErr.message
                    : String(firstErr),
                  value.data.slice(0, 600),
                );
              }
              continue;
            }
            event = recovered;
          }
          let chunk: TChatCompletionChunk | null = null;
          try {
            chunk = spec.eventToChunk(event, state, options);
          } catch (err) {
            settleError(controller, err);
            return;
          }
          const terminal = spec.isTerminalEvent?.(event) === true;
          if (chunk !== null && !settled) {
            controller.enqueue(chunk);
          }
          if (terminal) {
            settleClose(controller);
            return;
          }
          if (chunk !== null) return;
        } catch (err) {
          if (settled) return;
          settleError(controller, err);
          return;
        }
      }
    },
    cancel(reason) {
      settled = true;
      initiateCancel(reason);
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
