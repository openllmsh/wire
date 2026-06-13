import type {
  TChatCompletionChunk,
  THeartbeatOptions,
  TSseEvent,
} from "@quantidexyz/openllmp";

const SSE_DONE = "[DONE]";

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

export const encodeSseEvent = (data: unknown): Uint8Array => {
  const payload = data === SSE_DONE ? SSE_DONE : JSON.stringify(data);
  return textEncoder.encode(`data: ${payload}\n\n`);
};

export const encodeSseDone = (): Uint8Array =>
  textEncoder.encode(`data: ${SSE_DONE}\n\n`);

export const encodeSseComment = (comment: string): Uint8Array =>
  textEncoder.encode(`: ${comment}\n\n`);

// Anthropic's documented keepalive: a real `ping` event (NOT a bare
// SSE comment). Claude Code drives the official Anthropic SDK whose
// event-typed stream switch-cases on `type` — emitting the genuine
// ping is what its standby loop waits on between content blocks while
// the model is thinking / running tools. Ref: Anthropic streaming
// messages spec ("ping events ... periodically").
export const encodeAnthropicPing = (): Uint8Array =>
  textEncoder.encode('event: ping\ndata: {"type": "ping"}\n\n');

const splitLines = (buffer: string): { events: string[]; rest: string } => {
  const events: string[] = [];
  let rest = buffer;
  for (;;) {
    const i = rest.indexOf("\n\n");
    if (i === -1) return { events, rest };
    events.push(rest.slice(0, i));
    rest = rest.slice(i + 2);
  }
};

const parseEvent = (raw: string): TSseEvent | null => {
  const trimmed = raw.replace(/\r/g, "");
  if (trimmed === "") return null;
  if (trimmed.startsWith(":"))
    return { kind: "comment", comment: trimmed.slice(1).trimStart() };
  const lines = trimmed.split("\n");
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (dataLines.length === 0) return null;
  const data = dataLines.join("\n");
  if (data === SSE_DONE) return { kind: "done" };
  return { kind: "data", data };
};

export const sseEventStream = (
  source: ReadableStream<Uint8Array>,
): ReadableStream<TSseEvent> => {
  const reader = source.getReader();
  let buffer = "";
  return new ReadableStream<TSseEvent>({
    async pull(controller) {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) {
          const trailing = parseEvent(buffer);
          if (trailing) controller.enqueue(trailing);
          controller.close();
          return;
        }
        buffer += textDecoder.decode(value, { stream: true });
        const { events, rest } = splitLines(buffer);
        buffer = rest;
        if (events.length === 0) continue;
        for (const e of events) {
          const parsed = parseEvent(e);
          if (parsed) controller.enqueue(parsed);
        }
        return;
      }
    },
    cancel(reason) {
      reader.cancel(reason).catch(() => {});
    },
  });
};

// runtime-only: holds ReadableStream<T> branches from .tee(); cannot be a Schema.
export type TTeeResult<T> = {
  client: ReadableStream<T>;
  observer: ReadableStream<T>;
};

export const teeStream = <T>(source: ReadableStream<T>): TTeeResult<T> => {
  const [a, b] = source.tee();
  return { client: a, observer: b };
};

/**
 * Inject a keepalive frame whenever the upstream goes quiet so a CLI
 * sitting in its streaming read loop (Claude Code, Codex, …) doesn't
 * hit its socket read timeout — and so idle-connection-reaping
 * intermediaries (CDNs, corporate proxies) don't drop the connection —
 * while a reasoning model is thinking or a tool is running.
 *
 * IMPORTANT: only safe over a source that enqueues COMPLETE SSE frames
 * per chunk (our adapter encoders do). It must NOT wrap a raw upstream
 * byte stream whose chunk boundaries are arbitrary — a keepalive could
 * land mid-frame and corrupt the event. Native pass-through therefore
 * relies on the upstream's own `ping`/`message_start` instead.
 *
 * The first beat fires immediately (not after `intervalMs`): the
 * dangerous window is the silence between HTTP 200 and the first token
 * of a slow reasoning run, which a delayed first beat would leave
 * uncovered for a full interval.
 */
export const withHeartbeat = (
  source: ReadableStream<Uint8Array>,
  options: THeartbeatOptions,
): ReadableStream<Uint8Array> => {
  const reader = source.getReader();
  let timer: ReturnType<typeof setInterval> | null = null;
  let controllerRef: ReadableStreamDefaultController<Uint8Array> | null = null;
  let closed = false;

  const beat =
    options.kind === "anthropic_ping"
      ? encodeAnthropicPing
      : (): Uint8Array => encodeSseComment("keepalive");

  const clear = (): void => {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  };

  const safeEnqueue = (chunk: Uint8Array): void => {
    if (closed || controllerRef === null) return;
    try {
      controllerRef.enqueue(chunk);
    } catch {
      // Controller already closed/errored (client hung up between the
      // timer firing and this microtask) — stop beating.
      closed = true;
      clear();
    }
  };

  return new ReadableStream<Uint8Array>({
    start(controller) {
      controllerRef = controller;
      // Immediate first beat closes the post-200 / pre-first-token gap.
      safeEnqueue(beat());
      timer = setInterval(() => safeEnqueue(beat()), options.intervalMs);
    },
    async pull(controller) {
      const { value, done } = await reader.read();
      if (done) {
        closed = true;
        clear();
        controller.close();
        return;
      }
      if (value) controller.enqueue(value);
    },
    cancel(reason) {
      closed = true;
      clear();
      reader.cancel(reason).catch(() => {});
    },
  });
};

/**
 * Frame-aligned keepalive for a RAW upstream byte stream (native
 * Anthropic pass-through). `withHeartbeat` cannot be used there: the
 * upstream's chunk boundaries are arbitrary, so a keepalive injected
 * between two reads could land in the middle of a half-sent SSE event
 * and corrupt it. That is why pass-through historically injected
 * nothing and relied on the upstream's own `ping` events — which
 * leaves the client stranded the moment an anthropic-wire provider
 * goes quiet for minutes during a long reasoning / tool run (the
 * reported "chat stopped while it was actually doing something").
 *
 * This wrapper makes injection safe by re-aligning the stream to SSE
 * frame boundaries: incoming bytes are buffered and only COMPLETE
 * frames (terminated by a blank line) are forwarded; a partial frame
 * is held back until its terminator arrives. Because the client only
 * ever sees whole frames, a keepalive enqueued on the interval is
 * always emitted at a frame boundary — never spliced inside an event
 * — regardless of how the upstream chunked its bytes.
 *
 * Trailing bytes with no terminator (an upstream that ends mid-frame)
 * are flushed verbatim on close so nothing is silently dropped.
 */
export const withFrameAlignedHeartbeat = (
  source: ReadableStream<Uint8Array>,
  options: THeartbeatOptions,
): ReadableStream<Uint8Array> => {
  const reader = source.getReader();
  let timer: ReturnType<typeof setInterval> | null = null;
  let controllerRef: ReadableStreamDefaultController<Uint8Array> | null = null;
  let closed = false;
  let buffer = "";

  const beat =
    options.kind === "anthropic_ping"
      ? encodeAnthropicPing
      : (): Uint8Array => encodeSseComment("keepalive");

  const clear = (): void => {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  };

  const safeEnqueue = (chunk: Uint8Array): void => {
    if (closed || controllerRef === null) return;
    try {
      controllerRef.enqueue(chunk);
    } catch {
      closed = true;
      clear();
    }
  };

  return new ReadableStream<Uint8Array>({
    start(controller) {
      controllerRef = controller;
      // Immediate first beat closes the post-200 / pre-first-token gap.
      safeEnqueue(beat());
      timer = setInterval(() => safeEnqueue(beat()), options.intervalMs);
    },
    async pull(controller) {
      for (;;) {
        const { value, done } = await reader.read();
        if (done) {
          // Flush any unterminated trailing bytes verbatim so an
          // upstream that ends mid-frame loses nothing.
          if (buffer.length > 0) {
            controller.enqueue(textEncoder.encode(buffer));
            buffer = "";
          }
          closed = true;
          clear();
          controller.close();
          return;
        }
        buffer += textDecoder.decode(value, { stream: true });
        const { events, rest } = splitLines(buffer);
        buffer = rest;
        if (events.length === 0) continue;
        for (const e of events) {
          controller.enqueue(textEncoder.encode(`${e}\n\n`));
        }
        return;
      }
    },
    cancel(reason) {
      closed = true;
      clear();
      reader.cancel(reason).catch(() => {});
    },
  });
};

/**
 * Self-imposed graceful deadline for a CANONICAL chunk stream.
 *
 * Vercel hard-kills a function the instant it hits `maxDuration`
 * (clamped to the plan ceiling, which may be well below the declared
 * value). Mid-stream that severs the SSE connection with no terminal
 * frame: the client SDK sees a truncated stream and throws — it
 * cannot distinguish a truncation from a crash, so it cannot
 * continue. The idle heartbeat does NOT help here; it keeps the
 * socket warm right up until the platform guillotine.
 *
 * `withStreamDeadline` cuts us off a touch early instead. It forwards
 * real chunks until `deadlineMs` elapses, then:
 *   1. cancels the source — propagating to the upstream `fetch` body
 *      so we stop paying the provider for tokens we'll never deliver;
 *   2. emits ONE synthetic terminal chunk with
 *      `finish_reason: "length"` and closes cleanly.
 *
 * The text deltas already streamed ARE the partial answer (they're in
 * the client's hands). The synthetic terminal makes the SSE encoders
 * close properly — `data: [DONE]` for OpenAI, `message_delta`
 * (`stop_reason: "max_tokens"`) + `message_stop` for Anthropic — so
 * the caller gets a clean, truncated turn it can re-prompt to resume,
 * instead of a thrown network error.
 *
 * Wrap BEFORE `.tee()` so the usage-drain branch also terminates.
 */
export const withStreamDeadline = (
  source: ReadableStream<TChatCompletionChunk>,
  options: {
    readonly deadlineMs: number;
    /**
     * Fires exactly once if the deadline expires before the source
     * closes naturally. Callers use this to skip caching a body that
     * was sealed by the synthetic terminator instead of the upstream's
     * own `finish_reason`. Omit to ignore truncation.
     */
    readonly onTruncate?: () => void;
  },
): ReadableStream<TChatCompletionChunk> => {
  const reader = source.getReader();
  let lastId = "chatcmpl-openllm-deadline";
  let lastModel = "";
  let finished = false;
  const DEADLINE: unique symbol = Symbol("deadline");
  let timer: ReturnType<typeof setTimeout> | null = null;
  const deadline = new Promise<typeof DEADLINE>((resolve) => {
    timer = setTimeout(
      () => resolve(DEADLINE),
      Math.max(0, options.deadlineMs),
    );
  });
  const clear = (): void => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };
  const terminal = (): TChatCompletionChunk => ({
    id: lastId,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model: lastModel,
    choices: [{ index: 0, delta: {}, finish_reason: "length" }],
  });

  return new ReadableStream<TChatCompletionChunk>({
    async pull(controller) {
      if (finished) {
        controller.close();
        return;
      }
      const read = reader.read();
      // The losing race branch must not surface as an unhandled
      // rejection once we cancel the reader on deadline.
      read.catch(() => {});
      let r: Awaited<typeof read> | typeof DEADLINE;
      try {
        r = await Promise.race([read, deadline]);
      } catch (err) {
        // Source errored before the deadline — preserve existing
        // behaviour (the SSE encoder's catch emits a clean error
        // frame + terminator).
        finished = true;
        clear();
        throw err;
      }
      if (r === DEADLINE) {
        finished = true;
        clear();
        reader.cancel("openllm-stream-deadline").catch(() => {});
        if (options.onTruncate !== undefined) {
          try {
            options.onTruncate();
          } catch {
            // Callback errors must not crash the stream.
          }
        }
        controller.enqueue(terminal());
        controller.close();
        return;
      }
      if (r.done) {
        finished = true;
        clear();
        controller.close();
        return;
      }
      const value = r.value;
      if (typeof value.id === "string" && value.id.length > 0) {
        lastId = value.id;
      }
      if (typeof value.model === "string" && value.model.length > 0) {
        lastModel = value.model;
      }
      controller.enqueue(value);
    },
    cancel(reason) {
      finished = true;
      clear();
      reader.cancel(reason).catch(() => {});
    },
  });
};

/**
 * Result of wrapping a raw byte stream with a deadline guard.
 *
 * `isTruncated()` reflects whether the deadline fired before the
 * source closed naturally — the caller checks this AFTER the stream
 * is consumed to decide whether to cache or discard the accumulated
 * body. A truncated body would be a broken replay.
 */
export type TByteStreamDeadlineHandle = {
  readonly stream: ReadableStream<Uint8Array>;
  readonly isTruncated: () => boolean;
};

/**
 * Sibling of `withStreamDeadline` for the Anthropic pass-through:
 * the source is raw provider SSE bytes (not canonical chunks), and
 * the synthetic terminator must be a pre-encoded SSE byte sequence
 * because there is no canonical-decode step in the pipeline.
 *
 * On cutoff:
 *   1. cancel the source — propagating to the upstream `fetch` body
 *      so we stop paying the provider for tokens we'll never deliver;
 *   2. enqueue ONE pre-encoded `message_delta` + `message_stop` pair
 *      with `stop_reason: "max_tokens"` (matching what the canonical
 *      adapter already emits for `finish_reason: "length"`);
 *   3. close cleanly so SSE clients see a terminated turn instead of
 *      a severed connection.
 *
 * Wrap BEFORE `.tee()` so the usage-drain branch also receives the
 * terminator and the accumulator can finalise. The caller checks
 * `isTruncated()` after consumption to skip caching a truncated body.
 *
 * Best-effort frame alignment: the deadline can fire mid-frame on the
 * raw byte stream, but the synthetic terminator starts with a fresh
 * `event:` line — so the worst case is a torn frame followed by two
 * well-formed events, which the Anthropic SDK parser handles by
 * discarding the unterminated lead bytes.
 */
export const withByteStreamDeadline = (
  source: ReadableStream<Uint8Array>,
  options: { deadlineMs: number },
): TByteStreamDeadlineHandle => {
  const reader = source.getReader();
  let finished = false;
  let truncated = false;
  const DEADLINE: unique symbol = Symbol("byte-stream-deadline");
  let timer: ReturnType<typeof setTimeout> | null = null;
  const deadline = new Promise<typeof DEADLINE>((resolve) => {
    timer = setTimeout(
      () => resolve(DEADLINE),
      Math.max(0, options.deadlineMs),
    );
  });
  const clear = (): void => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };
  const terminator = (): Uint8Array =>
    textEncoder.encode(
      'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":0}}\n\n' +
        'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    );

  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (finished) {
        controller.close();
        return;
      }
      const read = reader.read();
      // The losing race branch must not surface as an unhandled
      // rejection once we cancel the reader on deadline.
      read.catch(() => {});
      let r: Awaited<typeof read> | typeof DEADLINE;
      try {
        r = await Promise.race([read, deadline]);
      } catch (err) {
        finished = true;
        clear();
        throw err;
      }
      if (r === DEADLINE) {
        finished = true;
        truncated = true;
        clear();
        reader.cancel("openllm-byte-stream-deadline").catch(() => {});
        controller.enqueue(terminator());
        controller.close();
        return;
      }
      if (r.done) {
        finished = true;
        clear();
        controller.close();
        return;
      }
      controller.enqueue(r.value);
    },
    cancel(reason) {
      finished = true;
      clear();
      reader.cancel(reason).catch(() => {});
    },
  });

  return { stream, isTruncated: () => truncated };
};
