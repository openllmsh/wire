import type { TChatCompletionRequest } from "@openllmsh/protocol";

/**
 * The canonical content parts the inbound adapters PRODUCE — the
 * OpenAI-native subset of `ContentPart` in `packages/protocol/chat.ts`
 * (no `input_audio`: nothing inbound maps to it). Shared by the
 * `adapters/messages` and `adapters/responses` request adapters so the
 * two can't drift apart.
 */
export type TCanonicalContentPart =
  | { readonly type: "text"; readonly text: string }
  | {
      readonly type: "image_url";
      readonly image_url: {
        readonly url: string;
        readonly detail?: "auto" | "low" | "high";
      };
    }
  | {
      readonly type: "file";
      readonly file: {
        readonly file_data?: string;
        readonly file_id?: string;
        readonly filename?: string;
      };
    };

/**
 * True iff any `messages[].content[]` part is `type === "image_url"`.
 * Walks every role (user / assistant / tool) — the quoted serde error
 * is `messages[4]`. String content, null assistant content, and `file`
 * parts are not images.
 */
export const requestHasImageContent = (req: TChatCompletionRequest): boolean =>
  req.messages.some((m) => {
    if (typeof m.content === "string") return false;
    return (m.content ?? []).some((part) => part.type === "image_url");
  });
