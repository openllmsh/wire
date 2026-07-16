import type { TChatCompletionChunk } from "@openllmsh/protocol";

/**
 * Outbound adapter (streaming): canonical ChatCompletion chunks → OpenAI
 * **Responses API** SSE events, for a `/v1/responses` streaming client (Codex).
 * The inverse of `chatGptEventToChunk` (the decoder the daemon uses on the
 * chatgpt UPSTREAM), so the same event vocabulary round-trips.
 *
 * Emits named events (`event: <type>\ndata: <json>\n\n`) with a monotonic
 * `sequence_number`, covering the core choreography:
 *   response.created
 *   → response.output_item.added (message | function_call)
 *   → response.output_text.delta | response.function_call_arguments.delta
 *   → response.output_item.done
 *   → response.completed   (full output[] + usage)
 *
 * Output items are assigned `output_index` in first-appearance order: a text
 * message (claimed on the first content delta) and one function_call per
 * canonical tool-call index. Usage rides in on the trailing usage-only chunk
 * (OpenAI delivers token counts in a separate final chunk) and is folded into
 * `response.completed`.
 */

const enc = new TextEncoder();

type TMessageItem = {
  readonly kind: "message";
  readonly outputIndex: number;
  readonly id: string;
  text: string;
};
type TFunctionItem = {
  readonly kind: "function_call";
  readonly outputIndex: number;
  readonly id: string;
  readonly callId: string;
  name: string;
  args: string;
};
type TOutputItem = TMessageItem | TFunctionItem;

const itemJson = (it: TOutputItem): unknown =>
  it.kind === "message"
    ? {
        type: "message",
        id: it.id,
        status: "completed",
        role: "assistant",
        content: [{ type: "output_text", text: it.text, annotations: [] }],
      }
    : {
        type: "function_call",
        id: it.id,
        call_id: it.callId,
        name: it.name,
        arguments: it.args,
        status: "completed",
      };

export const chunksToResponsesSseBytes = (
  chunks: ReadableStream<TChatCompletionChunk>,
): ReadableStream<Uint8Array> => {
  const reader = chunks.getReader();
  let seq = 0;
  let createdEmitted = false;
  let responseId = "resp_unknown";
  let createdAt = 0;
  let model = "";
  let nextOutputIndex = 0;
  let textItem: TMessageItem | null = null;
  const toolItems = new Map<number, TFunctionItem>();
  const order: TOutputItem[] = [];
  let usage: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  } | null = null;

  const ev = (type: string, payload: Record<string, unknown>): Uint8Array =>
    enc.encode(
      `event: ${type}\ndata: ${JSON.stringify({
        type,
        sequence_number: seq++,
        ...payload,
      })}\n\n`,
    );

  const responseObject = (status: string): Record<string, unknown> => ({
    id: responseId,
    object: "response",
    created_at: createdAt,
    status,
    model,
    output: order.map(itemJson),
    ...(usage !== null ? { usage } : {}),
  });

  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        // Read chunks until we have at least one event to emit (or the
        // upstream ends) — so we never enqueue an empty pull.
        for (;;) {
          const read = await reader.read();
          if (read.done) {
            const tail: Uint8Array[] = [];
            // An upstream that produced NO output item at all (no text, no
            // tool call) is a failure, not a completion — e.g. the chatgpt
            // backend's documented `output=[]` reply for an unavailable /
            // misconfigured model (observed on `gpt-5.3-codex-spark`). Emit
            // `response.failed` with a clear reason instead of a silent
            // `response.completed` that leaves Codex showing a blank turn.
            // (`response.created` is emitted first so the client sees a
            // well-formed stream even when zero chunks arrived.)
            if (order.length === 0) {
              if (!createdEmitted) {
                controller.enqueue(
                  ev("response.created", {
                    response: responseObject("in_progress"),
                  }),
                );
              }
              controller.enqueue(
                ev("response.failed", {
                  response: {
                    ...responseObject("failed"),
                    error: {
                      code: "empty_output",
                      message:
                        "The upstream model returned no output. The model id may be unavailable or misconfigured for this account.",
                    },
                  },
                }),
              );
              controller.close();
              return;
            }
            for (const it of order) {
              tail.push(
                ev("response.output_item.done", {
                  output_index: it.outputIndex,
                  item: itemJson(it),
                }),
              );
            }
            tail.push(
              ev("response.completed", {
                response: responseObject("completed"),
              }),
            );
            for (const b of tail) controller.enqueue(b);
            controller.close();
            return;
          }

          const chunk = read.value;
          const out: Uint8Array[] = [];

          if (!createdEmitted) {
            responseId = `resp_${chunk.id}`;
            createdAt = chunk.created;
            model = chunk.model;
            createdEmitted = true;
            out.push(
              ev("response.created", {
                response: responseObject("in_progress"),
              }),
            );
          }
          if (chunk.usage != null) {
            usage = {
              input_tokens: chunk.usage.prompt_tokens,
              output_tokens: chunk.usage.completion_tokens,
              total_tokens: chunk.usage.total_tokens,
            };
          }

          const delta = chunk.choices[0]?.delta;
          const content = delta?.content;
          if (typeof content === "string" && content.length > 0) {
            if (textItem === null) {
              textItem = {
                kind: "message",
                outputIndex: nextOutputIndex++,
                id: `msg_${chunk.id}`,
                text: "",
              };
              order.push(textItem);
              out.push(
                ev("response.output_item.added", {
                  output_index: textItem.outputIndex,
                  item: {
                    type: "message",
                    id: textItem.id,
                    status: "in_progress",
                    role: "assistant",
                    content: [],
                  },
                }),
              );
            }
            textItem.text += content;
            out.push(
              ev("response.output_text.delta", {
                item_id: textItem.id,
                output_index: textItem.outputIndex,
                content_index: 0,
                delta: content,
              }),
            );
          }

          for (const tc of delta?.tool_calls ?? []) {
            let item = toolItems.get(tc.index);
            if (item === undefined) {
              const callId = tc.id ?? `call_${tc.index}`;
              item = {
                kind: "function_call",
                outputIndex: nextOutputIndex++,
                id: `fc_${callId}`,
                callId,
                name: tc.function?.name ?? "",
                args: "",
              };
              toolItems.set(tc.index, item);
              order.push(item);
              out.push(
                ev("response.output_item.added", {
                  output_index: item.outputIndex,
                  item: {
                    type: "function_call",
                    id: item.id,
                    call_id: item.callId,
                    name: item.name,
                    arguments: "",
                  },
                }),
              );
            }
            if (
              tc.function?.name != null &&
              tc.function.name.length > 0 &&
              item.name === ""
            ) {
              item.name = tc.function.name;
            }
            const argsDelta = tc.function?.arguments;
            if (typeof argsDelta === "string" && argsDelta.length > 0) {
              item.args += argsDelta;
              out.push(
                ev("response.function_call_arguments.delta", {
                  item_id: item.id,
                  output_index: item.outputIndex,
                  delta: argsDelta,
                }),
              );
            }
          }

          if (out.length > 0) {
            for (const b of out) controller.enqueue(b);
            return;
          }
          // else: nothing emittable (e.g. a role-only opener) — read again.
        }
      } catch (err) {
        controller.error(err);
      }
    },
    cancel() {
      reader.cancel().catch(() => {});
    },
  });
};
