import type {
  TChatCompletionResponse,
  TResponsesOutputItem,
  TResponsesResponse,
} from "@openllmsh/protocol";
import { extractMessageText } from "../../lib/canonical/message";
import {
  reasoningItemsFromUnknown,
  reasoningItemToResponsesInput,
} from "../messages/reasoning-signature";

/**
 * Outbound adapter (non-streaming): canonical ChatCompletion response →
 * OpenAI **Responses API** response. The inverse direction of the inbound
 * request adapter; lets a Codex client read a `/v1/responses` reply.
 *
 * Output-item order mirrors the Responses API: reasoning item(s) first, then
 * the assistant message (if it has text), then any function calls. A `length`
 * finish maps to `status: "incomplete"`, everything else to `"completed"`.
 */
export const toResponsesResponse = (
  resp: TChatCompletionResponse,
): TResponsesResponse => {
  const choice = resp.choices[0];
  const msg = choice?.message;
  const output: TResponsesOutputItem[] = [];

  if (msg?.reasoning_items != null) {
    for (const r of reasoningItemsFromUnknown(msg.reasoning_items)) {
      output.push(reasoningItemToResponsesInput(r));
    }
  }

  const text = msg != null ? extractMessageText(msg.content) : "";
  if (text.length > 0) {
    output.push({
      type: "message",
      id: `msg_${resp.id}`,
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text }],
    });
  }

  if (msg?.tool_calls != null) {
    for (const tc of msg.tool_calls) {
      output.push({
        type: "function_call",
        id: `fc_${tc.id}`,
        call_id: tc.id,
        name: tc.function.name,
        arguments: tc.function.arguments,
        status: "completed",
      });
    }
  }

  return {
    id: resp.id,
    object: "response",
    created_at: resp.created,
    status: choice?.finish_reason === "length" ? "incomplete" : "completed",
    model: resp.model,
    output,
    usage: {
      input_tokens: resp.usage.prompt_tokens,
      output_tokens: resp.usage.completion_tokens,
      total_tokens: resp.usage.total_tokens,
    },
  };
};
