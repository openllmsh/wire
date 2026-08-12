/**
 * Recovery for a replayed Anthropic `thinking`/`redacted_thinking` block whose
 * `signature` the upstream rejects. Anthropic thinking signatures are bound to
 * the serving model (and account). A multi-turn fallback that hops
 * `claude_code` models (e.g. fable → opus → sonnet) replays a prior hop's
 * thinking into a hop that can't verify the signature → pre-stream HTTP 400:
 *   `messages.N.content.M: Invalid \`signature\` in \`thinking\` block`
 *
 * Anthropic's own model-switch guidance is to strip `thinking` and
 * `redacted_thinking` from prior assistant turns (blocks are model-tied).
 * Stripping and retrying the same hop once lets the model re-reason instead of
 * knocking out the entire claude_code arm of the chain — the Anthropic-wire
 * analog of {@link ./encrypted-content.ts}'s Responses `encrypted_content`
 * strip-retry (issue #420).
 *
 * ## Strip scope (cache + tool_use)
 *
 * - **Deterministic + minimal.** Only `thinking` / `redacted_thinking` content
 *   blocks are removed. Message order, system/tools/user content, and all other
 *   assistant blocks (`text`, `tool_use`, …) are left untouched so the stripped
 *   prefix is stable across same-model turns (Anthropic prompt-cache matches an
 *   exact per-model prefix).
 * - **Historical only by default.** Thinking on the *active tool-use
 *   continuation* assistant turn is PRESERVED: Anthropic requires the final
 *   assistant turn's thinking block when that turn carries `tool_use` and the
 *   next user turn answers with `tool_result`. Blindly stripping that block
 *   breaks the tool loop (`thinking`/`redacted_thinking` blocks in the latest
 *   assistant message cannot be modified / must round-trip).
 * - **Limitation (documented, not papered over).** If the preserved final-turn
 *   thinking itself carries a foreign-model signature, the same 400 will fire
 *   again after strip (we deliberately did not touch it). Cross-model
 *   mid-tool-loop continuation cannot be made valid: keeping the block 400s on
 *   signature, stripping it 400s on tool-use integrity. The reactive retry then
 *   surfaces the original error and the walker advances / fails normally. Same-
 *   model tool_use continuation is unaffected (signature still verifies).
 * - **No proactive strip.** Some signature forms embed their minting model
 *   (e.g. Anthropic CAIS), so origin is not always opaque — but the origin
 *   model is not the deciding factor: only the TARGET model's acceptance is,
 *   and that is knowable only from its response. Same-family signatures are
 *   frequently accepted, so proactively stripping on every model hop would
 *   bust prompt cache on healthy same-model multi-turn for no reason. The
 *   reactive 400 → strip → retry path strips only when the upstream actually
 *   rejects, which is the correctness backstop.
 */

/** Anthropic 400 body for an undecryptable / foreign-model thinking signature.
 *  Matches the documented phrasing and tolerates the `messages.N.content.M: `
 *  prefix Anthropic attaches; also accepts a bare `Invalid \`signature\` in
 *  \`thinking\` block` substring. */
const THINKING_SIGNATURE_FAILURE_RE =
  /Invalid\s*`signature`\s*in\s*`thinking`\s*block/i;

export const isThinkingSignatureError = (rawBody: string): boolean =>
  rawBody.length > 0 && THINKING_SIGNATURE_FAILURE_RE.test(rawBody);

const isThinkingBlock = (block: unknown): boolean => {
  if (block === null || typeof block !== "object") return false;
  const type = (block as { type?: unknown }).type;
  return type === "thinking" || type === "redacted_thinking";
};

const isToolUseBlock = (block: unknown): boolean =>
  block !== null &&
  typeof block === "object" &&
  (block as { type?: unknown }).type === "tool_use";

const isToolResultBlock = (block: unknown): boolean =>
  block !== null &&
  typeof block === "object" &&
  (block as { type?: unknown }).type === "tool_result";

const messageContent = (msg: unknown): unknown[] | null => {
  if (msg === null || typeof msg !== "object") return null;
  const content = (msg as { content?: unknown }).content;
  return Array.isArray(content) ? content : null;
};

const isAssistantWithThinking = (msg: unknown): boolean => {
  if (msg === null || typeof msg !== "object") return false;
  if ((msg as { role?: unknown }).role !== "assistant") return false;
  const content = messageContent(msg);
  return content !== null && content.some(isThinkingBlock);
};

/**
 * True when `messages[i]` is the pending tool-use continuation Anthropic
 * requires thinking to round-trip for. That pair must occupy the FINAL TWO
 * messages: `messages[len-2]` is an assistant turn carrying `tool_use` and
 * `messages[len-1]` is a user turn carrying `tool_result`. Only index
 * `len-2` is active — every earlier assistant-with-thinking (including a
 * completed tool loop that is merely the latest tool_use/tool_result pair
 * but is followed by more turns) is historical and safe to strip.
 */
const isActiveToolUseContinuation = (
  messages: ReadonlyArray<unknown>,
  index: number,
): boolean => {
  // Active only when this message is the penultimate one (final two messages).
  if (messages.length < 2 || index !== messages.length - 2) return false;

  const cur = messages[index];
  const next = messages[index + 1];
  if (cur === null || typeof cur !== "object") return false;
  if (next === null || typeof next !== "object") return false;
  if ((cur as { role?: unknown }).role !== "assistant") return false;
  if ((next as { role?: unknown }).role !== "user") return false;

  const curContent = messageContent(cur);
  const nextContent = messageContent(next);
  if (curContent === null || nextContent === null) return false;
  return (
    curContent.some(isToolUseBlock) && nextContent.some(isToolResultBlock)
  );
};

/** True when a built Anthropic Messages body carries at least one assistant
 *  `thinking`/`redacted_thinking` block that the strip transform would remove
 *  (i.e. not solely the preserved active tool-use continuation). A retry is
 *  only worth attempting when stripping would change the request. */
export const messagesBodyHasStrippableThinking = (body: unknown): boolean => {
  if (body === null || typeof body !== "object") return false;
  const messages = (body as { messages?: unknown }).messages;
  if (!Array.isArray(messages)) return false;
  return messages.some(
    (msg, i) =>
      isAssistantWithThinking(msg) && !isActiveToolUseContinuation(messages, i),
  );
};

/** Return a copy of a built Anthropic Messages body with `thinking` and
 *  `redacted_thinking` blocks removed from every HISTORICAL assistant message.
 *  The active tool-use continuation assistant turn (tool_use answered by a
 *  trailing tool_result) is left intact — see file header. Other content
 *  blocks and non-assistant messages are untouched; order is preserved. */
export const stripMessagesThinkingBlocks = (body: unknown): unknown => {
  if (body === null || typeof body !== "object") return body;
  const b = body as { messages?: unknown };
  if (!Array.isArray(b.messages)) return body;

  const messages = b.messages.map((msg, i) => {
    if (msg === null || typeof msg !== "object") return msg;
    if ((msg as { role?: unknown }).role !== "assistant") return msg;
    if (isActiveToolUseContinuation(b.messages as ReadonlyArray<unknown>, i)) {
      return msg;
    }
    const content = messageContent(msg);
    if (content === null || !content.some(isThinkingBlock)) return msg;
    // Drop only thinking / redacted_thinking; keep text, tool_use, etc. in order.
    // Empty content after strip is left as `[]` — rare (assistant-only-thinking
    // turns); inventing a placeholder would be non-deterministic / semantic.
    return {
      ...(msg as Record<string, unknown>),
      content: content.filter((block) => !isThinkingBlock(block)),
    };
  });

  return { ...(body as Record<string, unknown>), messages };
};
