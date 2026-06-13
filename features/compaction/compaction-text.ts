import type { TAnthropicResponse } from "@quantidexyz/openllmp";

/**
 * Claude Code compaction (`Xm()` in bundled CLI) rejects summaries when the
 * extracted visible text is empty or **shorter than 10 characters** after
 * trim — see https://github.com/anthropics/claude-code/issues/11095
 */
export const COMPACTION_MIN_VISIBLE_TEXT_CHARS = 10;

export type TAnthropicAssistantVisibleTextOpts = {
  /** Join between consecutive `text` blocks. Default `""` (matches Xm / #11095). */
  readonly betweenBlocks?: string;
};

export const anthropicAssistantVisibleTextTrimmed = (
  resp: TAnthropicResponse,
  opts: TAnthropicAssistantVisibleTextOpts = {},
): string => {
  const betweenBlocks = opts.betweenBlocks ?? "";
  const trimParts = betweenBlocks !== "";
  const texts = resp.content
    .filter((b): b is { type: "text"; text: string } => b.type === "text")
    .map((b) => (trimParts ? b.text.trim() : b.text));
  return texts.join(betweenBlocks).trim();
};

const FALLBACK_NO_SUMMARY =
  "OpenLLM: no visible summary text was returned for this compaction; you may retry /compact or continue the session.";

/**
 * Ensures assistant `text` blocks meet Claude Code’s minimum length so
 * `/compact` does not fail at ~30–40% with “did not contain valid text content”.
 */
export const ensureCompactionSafeVisibleText = (text: string): string => {
  const t = text.trim();
  if (t.length >= COMPACTION_MIN_VISIBLE_TEXT_CHARS) {
    return t;
  }
  if (t.length === 0) {
    return FALLBACK_NO_SUMMARY;
  }
  return `${t}${".".repeat(COMPACTION_MIN_VISIBLE_TEXT_CHARS - t.length)}`;
};
