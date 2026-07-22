import { GATEWAY_PROMPT_PREFIX } from "@openllmsh/protocol";
import { CHATGPT_DEFAULT_INSTRUCTIONS } from "./chatgpt/common";

/**
 * ─── System-prompt injection — the ONE home ────────────────────────────
 *
 * Every system-prompt / preamble injection the gateway performs lives in
 * this module, so there is a single obvious place to find and change what
 * we prepend to model calls:
 *
 * 1. {@link applyCodexDefaultInstructions} — the Codex identity preamble
 *    floor for bare clients on the ChatGPT/Codex wire.
 * 2. {@link injectGatewayPromptPrefix} — OUR gateway policy prefix
 *    (`GATEWAY_PROMPT_PREFIX` in `@openllmsh/protocol/prompt-prefix.ts` —
 *    edit the TEXT there), injected into every upstream chat body EXCEPT
 *    the genuine Claude Code subscription hop (which is forwarded verbatim
 *    — see `buildUpstreamBody` in `./upstream-request`).
 *
 * All injections are idempotent. The composition runs at the tail of
 * `buildUpstreamBody` (`./upstream-request`) — the single choke point the
 * cloud runner and the daemon walker share.
 */

/** Mirrors `TUpstreamWire` in `./upstream-request` (not imported — that
 *  module imports THIS one). */
type TWire = "anthropic" | "chatgpt" | "openai";

// ─── Claude Code OAuth preamble (vendor-required, client-supplied) ─────

/**
 * Anthropic's OAuth (subscription) path GATES inference on the request's FIRST
 * system block being EXACTLY this string — a `text` block, verbatim, in first
 * position. A request that lacks it comes back `429 {type:"rate_limit_error",
 * message:"Error"}` — a spoof-guard masquerading as a rate limit (confirmed
 * live 2026-07-17 against claude-sonnet-4-5). The genuine Claude Code CLI
 * ALWAYS sends it; the handrolled OAuth-Anthropic hop only ever serves that
 * genuine CLI (every other originator takes the bridge), so its request is
 * forwarded VERBATIM and we no longer synthesise or reorder the preamble.
 * Kept for `injectAnthropic`'s slotting check on the BYOK path. */
export const CLAUDE_CODE_SYSTEM_PREAMBLE =
  "You are Claude Code, Anthropic's official CLI for Claude.";

type TSystemBlock = { type: "text"; text: string };

/** Is this the Claude Code preamble already, as the FIRST system block? */
const firstBlockIsPreamble = (system: unknown): boolean => {
  if (typeof system === "string") return system === CLAUDE_CODE_SYSTEM_PREAMBLE;
  if (Array.isArray(system)) {
    const first = system[0] as { type?: unknown; text?: unknown } | undefined;
    return first?.type === "text" && first.text === CLAUDE_CODE_SYSTEM_PREAMBLE;
  }
  return false;
};

// ─── Codex default-instructions floor ──────────────────────────────────

/**
 * The Codex preamble is a Codex IDENTITY the ChatGPT backend historically
 * needed. Inject it ONLY when (a) the caller wants it (`codexInstructions !==
 * false`; `false` suppresses it for xAI Grok) AND (b) the client sent NO
 * instructions of its own. A real Codex client always sends its own (newer)
 * preamble — layering ours on top is a DOUBLE preamble that wastes ~2 KB of
 * input every turn and is unnecessary (current models produce output with no
 * preamble at all — audit 2026-07-14-codex-upstream-wire §6 F10). So we trust
 * the client's instructions verbatim and only add ours as a floor for a bare
 * client.
 */
export const applyCodexDefaultInstructions = (
  instructions: string,
  codexInstructions: boolean | undefined,
): string =>
  codexInstructions !== false && instructions.length === 0
    ? CHATGPT_DEFAULT_INSTRUCTIONS
    : instructions;

// ─── 3. Gateway policy prefix ──────────────────────────────────────────

/**
 * Inject `GATEWAY_PROMPT_PREFIX` into an UPSTREAM-shaped body, per wire.
 * Idempotent: a body already carrying the prefix verbatim is untouched.
 *
 * Wire placement:
 * - `anthropic` — a `system` text block. Inserted AFTER a leading Claude
 *   Code OAuth preamble when present (that preamble MUST stay the first
 *   block — see {@link ensureClaudeCodeSystemPreamble}), else first.
 * - `chatgpt`  — appended to the top-level `instructions` string (the
 *   Codex wire rejects `role:"system"` turns; appending keeps the client
 *   or Codex preamble leading, which the backend expects).
 * - `openai`   — a leading `role:"system"` message.
 */
export const injectGatewayPromptPrefix = (
  upstreamWire: TWire,
  body: unknown,
): unknown => {
  if (typeof body !== "object" || body === null) return body;
  const record = body as Record<string, unknown>;
  if (upstreamWire === "anthropic") return injectAnthropic(record);
  if (upstreamWire === "chatgpt") return injectChatGpt(record);
  return injectOpenAi(record);
};

const injectAnthropic = (record: Record<string, unknown>): unknown => {
  const system = record.system;
  const blocks: TSystemBlock[] =
    typeof system === "string" && system.length > 0
      ? [{ type: "text", text: system }]
      : Array.isArray(system)
        ? (system as TSystemBlock[])
        : [];
  if (blocks.some((b) => b?.text === GATEWAY_PROMPT_PREFIX)) return record;
  const prefix: TSystemBlock = { type: "text", text: GATEWAY_PROMPT_PREFIX };
  // The Claude Code OAuth preamble gates inference on being the FIRST
  // block — slot the prefix right after it, never ahead of it.
  const next = firstBlockIsPreamble(blocks)
    ? [blocks[0] as TSystemBlock, prefix, ...blocks.slice(1)]
    : [prefix, ...blocks];
  return { ...record, system: next };
};

const injectChatGpt = (record: Record<string, unknown>): unknown => {
  const instructions =
    typeof record.instructions === "string" ? record.instructions : "";
  if (instructions.includes(GATEWAY_PROMPT_PREFIX)) return record;
  return {
    ...record,
    instructions:
      instructions.length > 0
        ? `${instructions}\n\n${GATEWAY_PROMPT_PREFIX}`
        : GATEWAY_PROMPT_PREFIX,
  };
};

const injectOpenAi = (record: Record<string, unknown>): unknown => {
  const messages = Array.isArray(record.messages) ? record.messages : [];
  const alreadyThere = messages.some(
    (m) =>
      (m as { role?: unknown; content?: unknown })?.role === "system" &&
      (m as { content?: unknown }).content === GATEWAY_PROMPT_PREFIX,
  );
  if (alreadyThere) return record;
  return {
    ...record,
    messages: [{ role: "system", content: GATEWAY_PROMPT_PREFIX }, ...messages],
  };
};
