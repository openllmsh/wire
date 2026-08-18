import type {
  TAnthropicStreamEvent,
  TChatCompletionChunk,
} from "@openllmsh/protocol";
import { ensureCompactionSafeVisibleText } from "../../features/compaction/compaction-text";
import { encodeSseEvent } from "../../lib/streaming/sse";
import { upstreamErrorFrom } from "../../lib/streaming/upstream-error";
import {
  anthropicStopReasonFrom,
  visibleAnswerAfterThought,
} from "./anthropic-map";
import { plainTextFromReasoningItems } from "./reasoning-from-items";
import {
  encodeReasoningSignature,
  reasoningItemsFromUnknown,
} from "./reasoning-signature";

type TDeltaFoldMode = "incremental" | "snapshot";

// runtime-only: stateful translation buffer.
export type TMessagesStreamState = {
  startEmitted: boolean;
  messageId: string;
  model: string;
  inputTokens: number;
  /**
   * Last-seen `usage.completion_tokens`. The synthetic EOF tail
   * carries no provider usage of its own; without this, a clean
   * `end_turn` terminal `message_delta` would reconcile to
   * `output_tokens: 0` and last-win the live trailer (Claude Code
   * treats `0` as authoritative, unlike input which is `> 0`-guarded).
   */
  outputTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  /** Anthropic content_index for the text block. null = not yet opened. */
  textBlockIndex: number | null;
  /** True while the text block is open (between content_block_start and stop). */
  textBlockOpen: boolean;
  /** Responses `reasoning_summary_text.delta` → Anthropic thinking block. */
  thinkingBlockIndex: number | null;
  thinkingBlockOpen: boolean;
  /** OpenAI tool_calls[i].index → Anthropic content_index. */
  toolCallToContentIndex: Map<number, number>;
  /**
   * Tool calls seen but not yet OPENED because their `name` is still empty.
   * Anthropic requires `tool_use.name` at `content_block_start`, and a client
   * (Claude Code) drops a `tool_use` whose name is `""` — so it never runs the
   * tool. The OpenAI Responses wire (grok/chatgpt) can split a tool call's
   * `name` / `id` / argument fragments across separate events, so the first
   * fragment may carry no name. We buffer `id`/`name`/`args` per OpenAI index
   * here and open the block the moment a non-empty name arrives (flushing the
   * buffered args), or belatedly at finish with a synthesized id. Mirrors
   * CLIProxyAPI's accumulate-until-(name)+belated-open (`openai_claude_response.go`).
   */
  pendingToolCalls: Map<number, { id: string; name: string; args: string }>;
  /** True once any tool_use block has been OPENED (announced) for this message. */
  emittedToolUse: boolean;
  /** Anthropic content_index → still-open flag. */
  openToolContentIndexes: Set<number>;
  /** Next free Anthropic content_index. */
  nextContentIndex: number;
  finalStopReason: ReturnType<typeof anthropicStopReasonFrom>;
  /**
   * Concatenation of all `reasoning_content` / thinking deltas. On
   * terminal chunk, if there was no non-empty `text_delta`, we mirror
   * this into a synthetic `text` block so Claude Code `/compact` sees
   * valid user-visible text (same rule as `toAnthropicMessagesResponse`).
   */
  thinkingAccumulated: string;
  /** True after at least one non-empty `content` delta became `text_delta`. */
  emittedNonemptyTextDelta: boolean;
  /** Set when `message_stop` is emitted — detects truncated upstream streams. */
  messageStopEmitted: boolean;
  /** Concatenation of all visible `text_delta`s (content + dumped reasoning). */
  textAccumulated: string;
  /**
   * Snapshot ledger for `delta.content` only. Must not share
   * `textAccumulated`: dumped `reasoning_content` rides that field, and
   * comparing answer snapshots against it would suppress a real answer
   * that happens to prefix the dumped thought.
   */
  contentSnapshotAccumulated: string;
  /**
   * True when the most recently processed chunk carried real provider
   * usage (`prompt_tokens > 0`). Sticky-ever-seen is wrong: a later
   * content chunk after a usage frame is a tear, not a clean close.
   * Live-estimate frames keep prompt_tokens at 0 and stay false.
   */
  lastChunkHadProviderUsage: boolean;
  /**
   * How many leading chars of `thinkingAccumulated` were already emitted as
   * `thinking_delta` events. Keeps streaming aligned when `reasoning_items`
   * arrives as a snapshot (final chunk only).
   */
  thinkingDeltaEmittedLen: number;
  /**
   * Encoded `thinking.signature` carrying the upstream's `reasoning`
   * item(s) (Codex/Responses `encrypted_content`). Set when a chunk
   * carries `reasoning_items`; flushed onto the thinking block right
   * before it closes so Claude Code replays it next turn. Without this
   * the model loses chain-of-thought state and loops forever.
   */
  pendingReasoningSignature: string | null;
  /**
   * Responses hops (Grok / Codex) tag summary deltas with
   * `reasoning_items: []` (and later a real encrypted item). The
   * summary must ride a signed `thinking` block — dumping it as
   * visible `text` and then opening an empty thinking block produces
   * two identical ⏺ prompts around "Thought for 1s".
   */
  signedReasoningChannel: boolean;
  /** True after a signed thinking block was sealed this turn. */
  sealedSignedThinking: boolean;
  /** Provider-executed hosted searches emitted so far — drives the terminal
   *  `message_delta.usage.server_tool_use.web_search_requests`. */
  serverSearchCount: number;
};

export const newMessagesStreamState = (): TMessagesStreamState => ({
  startEmitted: false,
  messageId: "",
  model: "",
  inputTokens: 0,
  outputTokens: 0,
  cacheReadTokens: 0,
  cacheCreationTokens: 0,
  textBlockIndex: null,
  textBlockOpen: false,
  thinkingBlockIndex: null,
  thinkingBlockOpen: false,
  toolCallToContentIndex: new Map(),
  pendingToolCalls: new Map(),
  emittedToolUse: false,
  openToolContentIndexes: new Set(),
  nextContentIndex: 0,
  finalStopReason: null,
  thinkingAccumulated: "",
  emittedNonemptyTextDelta: false,
  messageStopEmitted: false,
  textAccumulated: "",
  contentSnapshotAccumulated: "",
  lastChunkHadProviderUsage: false,
  thinkingDeltaEmittedLen: 0,
  pendingReasoningSignature: null,
  signedReasoningChannel: false,
  sealedSignedThinking: false,
  serverSearchCount: 0,
});

/**
 * Dedup / accumulate a text-bearing delta.
 *
 * Mode is channel identity, not string shape: `"The"` + `"There"` and a
 * growing snapshot that happens to start with the accumulated prefix are
 * the same `startsWith` condition. Incremental tokens (Kimi
 * `reasoning_content`, unmarked `delta.content`) always append.
 * Responses summaries (`reasoning_items` present, including `[]`)
 * replay the full prefix — emit only the tail.
 *
 * Identical `incoming === accumulated` is always a same-snapshot repeat
 * and is skipped in both modes.
 */
const foldDelta = (
  incoming: string,
  accumulated: string,
  mode: TDeltaFoldMode,
): { readonly next: string; readonly emit: string } => {
  if (incoming.length === 0) return { next: accumulated, emit: "" };
  if (incoming === accumulated) return { next: accumulated, emit: "" };
  if (mode === "snapshot") {
    if (incoming.startsWith(accumulated)) {
      return { next: incoming, emit: incoming.slice(accumulated.length) };
    }
    if (accumulated.startsWith(incoming)) {
      return { next: accumulated, emit: "" };
    }
    return { next: incoming, emit: incoming };
  }
  return { next: accumulated + incoming, emit: incoming };
};

/**
 * Emit the reasoning `signature_delta` onto the thinking block so
 * Claude Code replays it verbatim next turn. Anthropic requires
 * `signature_delta` to be the thinking block's LAST delta, before
 * `content_block_stop` and before any tool_use block opens — so this
 * runs the moment encrypted `reasoning_items` arrive (Codex/Grok send
 * them on the reasoning item's `output_item.done`, ahead of the
 * function call). The human-readable summary rides the same thinking
 * block; dumping it as `text` first is what produced the double ⏺.
 */
const emitReasoningSignature = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
): void => {
  if (
    state.pendingReasoningSignature === null ||
    state.openToolContentIndexes.size > 0
  ) {
    return;
  }
  closeTextBlock(state, out);
  closeAllToolBlocks(state, out);
  const idx = openThinkingBlock(state, out);
  const held = state.thinkingAccumulated.slice(state.thinkingDeltaEmittedLen);
  if (held.length > 0) {
    out.push({
      type: "content_block_delta",
      index: idx,
      delta: { type: "thinking_delta", thinking: held },
    });
    state.thinkingDeltaEmittedLen = state.thinkingAccumulated.length;
  }
  out.push({
    type: "content_block_delta",
    index: idx,
    delta: {
      type: "signature_delta",
      signature: state.pendingReasoningSignature,
    },
  });
  state.pendingReasoningSignature = null;
  state.sealedSignedThinking = true;
  closeThinkingBlock(state, out);
};

const closeThinkingBlock = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
): void => {
  if (state.thinkingBlockOpen && state.thinkingBlockIndex !== null) {
    out.push({
      type: "content_block_stop",
      index: state.thinkingBlockIndex,
    });
    state.thinkingBlockOpen = false;
  }
};

const openThinkingBlock = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
): number => {
  // A stopped thinking index must not be reopened (Anthropic indices
  // are unique + monotonic). A later reasoning item (Grok interleaves
  // summaries with tool calls) gets a fresh block.
  if (state.thinkingBlockIndex === null || !state.thinkingBlockOpen) {
    state.thinkingBlockIndex = state.nextContentIndex;
    state.nextContentIndex += 1;
  }
  if (!state.thinkingBlockOpen) {
    out.push({
      type: "content_block_start",
      index: state.thinkingBlockIndex,
      content_block: { type: "thinking", thinking: "" },
    });
    state.thinkingBlockOpen = true;
  }
  return state.thinkingBlockIndex;
};

const closeTextBlock = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
): void => {
  if (state.textBlockOpen && state.textBlockIndex !== null) {
    out.push({
      type: "content_block_stop",
      index: state.textBlockIndex,
    });
    state.textBlockOpen = false;
  }
};

const closeAllToolBlocks = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
): void => {
  for (const idx of state.openToolContentIndexes) {
    out.push({ type: "content_block_stop", index: idx });
  }
  state.openToolContentIndexes.clear();
  // Fresh tool_call deltas must open new content blocks. If reasoning or
  // another branch stops blocks mid-stream but leaves stale tc.index →
  // Anthropic index mappings, later `input_json_delta` targets a block we
  // already emitted `content_block_stop` for — Claude Code sees prose +
  // truncated pseudo-tools (e.g. literal `<tool_call>` tail).
  state.toolCallToContentIndex.clear();
};

/**
 * Open a tool_use content block for a now-named buffered tool call and flush
 * any argument fragments accumulated while its name was still empty. The block
 * stays open (subsequent fragments stream straight through). `id` is
 * synthesized when the upstream never supplied one, so the block is always
 * executable. Anthropic content blocks are strictly sequential, so any open
 * text / tool / thinking block is closed first.
 */
const openBufferedToolBlock = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
  openAiIndex: number,
  pending: { id: string; name: string; args: string },
): number => {
  closeTextBlock(state, out);
  closeAllToolBlocks(state, out);
  closeThinkingBlock(state, out);
  const contentIndex = state.nextContentIndex;
  state.nextContentIndex += 1;
  state.toolCallToContentIndex.set(openAiIndex, contentIndex);
  state.pendingToolCalls.delete(openAiIndex);
  state.emittedToolUse = true;
  state.openToolContentIndexes.add(contentIndex);
  out.push({
    type: "content_block_start",
    index: contentIndex,
    content_block: {
      type: "tool_use",
      id:
        pending.id !== ""
          ? pending.id
          : `toolu_${state.messageId}_${openAiIndex}`,
      name: pending.name,
      input: {},
    },
  });
  if (pending.args.length > 0) {
    out.push({
      type: "content_block_delta",
      index: contentIndex,
      delta: { type: "input_json_delta", partial_json: pending.args },
    });
  }
  return contentIndex;
};

/**
 * Belatedly open every still-pending tool call that accumulated a name but was
 * never opened (its `content_block_start` awaited a non-empty name that only
 * arrived on the terminal event, or the block was reset mid-stream). Called at
 * finish so a named-but-unopened tool still reaches the client. Pending calls
 * that never got a name are dropped (an unnamed tool is unexecutable) — mirrors
 * CLIProxyAPI's belated-emit that skips `accumulator.Name == ""`.
 */
const flushPendingToolBlocks = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
): void => {
  const indexes = [...state.pendingToolCalls.keys()].sort((a, b) => a - b);
  for (const idx of indexes) {
    const pending = state.pendingToolCalls.get(idx);
    if (pending === undefined || pending.name === "") {
      state.pendingToolCalls.delete(idx);
      continue;
    }
    const contentIndex = openBufferedToolBlock(state, out, idx, pending);
    out.push({ type: "content_block_stop", index: contentIndex });
    state.openToolContentIndexes.delete(contentIndex);
  }
};

const openTextBlock = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
): number => {
  closeThinkingBlock(state, out);
  // A previously-stopped content index can never be reopened (Anthropic
  // streaming indices are unique + monotonic). If the text block was
  // closed — e.g. a signed thinking block or a tool block was emitted
  // in between — allocate a FRESH index instead of resurrecting the
  // stopped one.
  if (state.textBlockIndex === null || !state.textBlockOpen) {
    state.textBlockIndex = state.nextContentIndex;
    state.nextContentIndex += 1;
  }
  if (!state.textBlockOpen) {
    out.push({
      type: "content_block_start",
      index: state.textBlockIndex,
      content_block: { type: "text", text: "" },
    });
    state.textBlockOpen = true;
  }
  return state.textBlockIndex;
};

const emitVisibleText = (
  state: TMessagesStreamState,
  out: TAnthropicStreamEvent[],
  text: string,
): void => {
  if (text.length === 0) return;
  const idx = openTextBlock(state, out);
  out.push({
    type: "content_block_delta",
    index: idx,
    delta: { type: "text_delta", text },
  });
  state.emittedNonemptyTextDelta = true;
  state.textAccumulated += text;
};

/**
 * Translate one OpenAI ChatCompletion chunk into zero or more
 * Anthropic SSE events. Handles both text deltas and tool_call deltas:
 * each new tool_call opens a new content_block(tool_use); subsequent
 * deltas for that tool_call emit input_json_delta events.
 */
export const chunkToMessagesEvents = (
  chunk: TChatCompletionChunk,
  state: TMessagesStreamState,
): TAnthropicStreamEvent[] => {
  const out: TAnthropicStreamEvent[] = [];

  if (!state.startEmitted) {
    state.startEmitted = true;
    state.messageId = chunk.id;
    state.model = chunk.model;
    out.push({
      type: "message_start",
      message: {
        id: state.messageId,
        type: "message",
        role: "assistant",
        model: state.model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    });
  }

  const choice = chunk.choices[0];
  const deltaReasoning = choice?.delta.reasoning_content ?? null;
  const deltaText = choice?.delta.content ?? null;
  const deltaToolCalls = choice?.delta.tool_calls ?? null;

  // Responses hops tag summary deltas with `reasoning_items` (empty
  // array = channel marker; later chunks carry the encrypted item).
  // Sticky: once we know the hop is signed, later unsigned-looking
  // fragments stay on the thinking ledger.
  if (
    choice?.delta.reasoning_items !== undefined &&
    choice.delta.reasoning_items !== null
  ) {
    state.signedReasoningChannel = true;
  }

  if (
    deltaReasoning !== null &&
    deltaReasoning !== undefined &&
    deltaReasoning.length > 0
  ) {
    // Unsigned reasoning (Kimi / DashScope) has no replay-safe
    // signature, so it must be visible `text` — Anthropic hard-rejects
    // a signature-less thinking block on replay. Signed Responses hops
    // (Grok / Codex) hold the same text for the thinking block opened
    // by `emitReasoningSignature`; dumping it first is the double ⏺.
    const reasoningSnap = foldDelta(
      deltaReasoning,
      state.thinkingAccumulated,
      state.signedReasoningChannel ? "snapshot" : "incremental",
    );
    state.thinkingAccumulated = reasoningSnap.next;
    if (
      reasoningSnap.emit.length > 0 &&
      state.openToolContentIndexes.size === 0 &&
      !state.signedReasoningChannel
    ) {
      emitVisibleText(state, out, reasoningSnap.emit);
      state.thinkingDeltaEmittedLen = state.thinkingAccumulated.length;
    }
  }

  const reasoningItems = reasoningItemsFromUnknown(
    choice?.delta.reasoning_items,
  );
  if (reasoningItems.length > 0) {
    const sig = encodeReasoningSignature(reasoningItems);
    if (sig !== null) {
      // A distinct new signature while one is still pending (Grok interleaves
      // multiple reasoning items) — seal the prior item as its own thinking
      // block before adopting the new one, so neither signature is lost.
      if (
        state.pendingReasoningSignature !== null &&
        state.pendingReasoningSignature !== sig
      ) {
        emitReasoningSignature(state, out);
      }
      state.pendingReasoningSignature = sig;
    }
  }

  const fromReasoningItems = plainTextFromReasoningItems(
    choice?.delta.reasoning_items,
  );
  // Growing `reasoning_items` snapshot of the same summary. Fold the
  // tail into the ledger. Do not dump as `text` here — signed hops
  // emit it as `thinking_delta` with the signature; unsigned hops
  // flush at content / finish (compaction-safe).
  // Reasoning item summaries are authoritative even when they are not a
  // prefix of the folded streamed delta (e.g. snapshot collapse) — accept a
  // longer snapshot whenever it grows.
  if (fromReasoningItems.length > state.thinkingAccumulated.length) {
    state.thinkingAccumulated = fromReasoningItems;
  }

  // Seal each signed reasoning item as soon as its encrypted item arrives.
  // Codex can deliver that item before its human summary. In that case the
  // replay-safe result is a signature-only thinking block, not a placeholder;
  // a later summary cannot be attached after `signature_delta` has sealed the
  // block. This keeps the signature last and before a subsequent tool_use.
  emitReasoningSignature(state, out);

  // Provider-executed hosted searches (Codex `webSearch` items on a chatgpt
  // hop) → self-contained server_tool_use + web_search_tool_result block
  // pairs. Blocks must stay strictly sequential, so anything open closes
  // first; codex interleaves commentary text → search → answer text, and
  // `openTextBlock` allocates a fresh index for the post-search text. The
  // result content is an empty list — Codex never exposes result items; the
  // findings ride the grounded answer text (see the JSON adapter's note).
  const deltaSearches = choice?.delta.server_search_calls ?? null;
  if (deltaSearches !== null && deltaSearches !== undefined) {
    // Boundary: seal any pending signature onto the thinking block before it
    // is closed for the server-search blocks.
    emitReasoningSignature(state, out);
    for (const search of deltaSearches) {
      closeTextBlock(state, out);
      closeAllToolBlocks(state, out);
      closeThinkingBlock(state, out);
      const useIndex = state.nextContentIndex;
      state.nextContentIndex += 1;
      out.push({
        type: "content_block_start",
        index: useIndex,
        content_block: {
          type: "server_tool_use",
          id: search.id,
          name: "web_search",
          input: {},
        },
      });
      out.push({
        type: "content_block_delta",
        index: useIndex,
        delta: {
          type: "input_json_delta",
          partial_json: JSON.stringify({ query: search.query }),
        },
      });
      out.push({ type: "content_block_stop", index: useIndex });
      const resultIndex = state.nextContentIndex;
      state.nextContentIndex += 1;
      out.push({
        type: "content_block_start",
        index: resultIndex,
        content_block: {
          type: "web_search_tool_result",
          tool_use_id: search.id,
          content: (search.results ?? []).map((r) => ({
            type: "web_search_result",
            url: r.url,
            title: r.title ?? r.url,
          })),
        },
      });
      out.push({ type: "content_block_stop", index: resultIndex });
      state.serverSearchCount += 1;
    }
  }

  // Text delta → open text block (if not already) + content_block_delta.
  // Content is incremental: exact-match skip, else append. A dedicated
  // ledger so dumped reasoning cannot swallow a later answer prefix.
  // Signed hops also drop content that merely restates the thought
  // (field sample: same sentence as a second ⏺ after "Thought for 1s").
  if (deltaText !== null && deltaText !== undefined && deltaText.length > 0) {
    // Boundary: the answer text starts — seal any pending signature before the
    // visible text block opens.
    emitReasoningSignature(state, out);
    const textSnap = foldDelta(
      deltaText,
      state.contentSnapshotAccumulated,
      "incremental",
    );
    state.contentSnapshotAccumulated = textSnap.next;
    let emit = textSnap.emit;
    if (state.signedReasoningChannel && state.thinkingAccumulated.length > 0) {
      const desired = visibleAnswerAfterThought(
        state.contentSnapshotAccumulated,
        state.thinkingAccumulated,
      );
      if (desired.startsWith(state.textAccumulated)) {
        emit = desired.slice(state.textAccumulated.length);
      } else if (state.textAccumulated.startsWith(desired)) {
        emit = "";
      } else {
        emit = desired;
      }
    }
    emitVisibleText(state, out, emit);
  }

  // Tool-call deltas → open tool_use blocks + input_json_delta events.
  if (deltaToolCalls !== null && deltaToolCalls !== undefined) {
    // Boundary: seal any pending signature BEFORE the first tool_use block
    // opens. Empty summaries produce a signature-only thinking block.
    emitReasoningSignature(state, out);
    for (const tc of deltaToolCalls) {
      const contentIndex = state.toolCallToContentIndex.get(tc.index);
      if (contentIndex !== undefined) {
        // Block already opened — stream this argument fragment straight through.
        const argFragment = tc.function?.arguments ?? "";
        if (argFragment.length > 0) {
          out.push({
            type: "content_block_delta",
            index: contentIndex,
            delta: { type: "input_json_delta", partial_json: argFragment },
          });
        }
        continue;
      }
      // Not opened yet — accumulate id / name / args in the pending buffer.
      // We CANNOT open a tool_use block until we have a non-empty `name`
      // (Anthropic requires it at `content_block_start`, and Claude Code
      // silently drops a nameless tool_use — the sub-agent never spawns). The
      // Responses wire may deliver name/id/args across separate events, so the
      // first fragment can be nameless.
      const pending = state.pendingToolCalls.get(tc.index) ?? {
        id: "",
        name: "",
        args: "",
      };
      if (tc.id != null && tc.id !== "") pending.id = tc.id;
      const fragmentName = tc.function?.name;
      if (fragmentName != null && fragmentName !== "") {
        pending.name = fragmentName;
      }
      pending.args += tc.function?.arguments ?? "";
      state.pendingToolCalls.set(tc.index, pending);
      // The moment we know the name, open the block and flush buffered args.
      if (pending.name !== "") {
        openBufferedToolBlock(state, out, tc.index, pending);
      }
    }
  }

  state.lastChunkHadProviderUsage =
    chunk.usage != null && chunk.usage.prompt_tokens > 0;
  if (chunk.usage != null) {
    state.inputTokens = chunk.usage.prompt_tokens;
    state.outputTokens = chunk.usage.completion_tokens;
    state.cacheReadTokens =
      chunk.usage.prompt_tokens_details?.cached_tokens ?? 0;
    state.cacheCreationTokens =
      chunk.usage.prompt_tokens_details?.cache_creation_tokens ?? 0;
  }

  const turnEnds =
    choice?.finish_reason !== null && choice?.finish_reason !== undefined;

  // Live token feed: a usage-bearing chunk that does NOT end the turn —
  // the running estimate synthesized by `withLiveUsageEstimate`, or an
  // upstream that reports incremental usage — emits a standalone
  // `message_delta` so a CLI's token counter climbs mid-stream instead
  // of staying at zero until completion. No `stop_reason`, no
  // `message_stop`: the turn is still open and the terminal
  // `message_delta` below reconciles to the provider's exact totals.
  if (
    chunk.usage != null &&
    !turnEnds &&
    state.startEmitted &&
    !state.messageStopEmitted
  ) {
    out.push({
      type: "message_delta",
      delta: { stop_reason: null, stop_sequence: null },
      usage: {
        output_tokens: chunk.usage.completion_tokens,
        input_tokens: state.inputTokens,
        ...(state.cacheCreationTokens > 0
          ? { cache_creation_input_tokens: state.cacheCreationTokens }
          : {}),
        ...(state.cacheReadTokens > 0
          ? { cache_read_input_tokens: state.cacheReadTokens }
          : {}),
      },
    });
  }

  if (choice?.finish_reason !== null && choice?.finish_reason !== undefined) {
    let finalStopReason = anthropicStopReasonFrom(choice.finish_reason);
    // Seal any pending reasoning signature before flushing terminally buffered
    // tool blocks. An empty summary produces a signature-only thinking block;
    // the call is a no-op when nothing is pending or a tool block is open.
    emitReasoningSignature(state, out);
    // A tool call whose name only arrived on the terminal event (or never
    // streamed a `content_block_start` because its name stayed empty until
    // now) is opened + closed here so a named-but-unopened `Task`/`Agent`
    // call still reaches the client. Runs BEFORE the stop-reason override
    // below (it sets `emittedToolUse` when it opens a block).
    flushPendingToolBlocks(state, out);
    // If the upstream emitted tool_call deltas during the stream but
    // wrongly settled on `finish_reason: "stop"`, override to
    // `tool_use`. Without this, Claude Code receives a
    // `stop_reason: "end_turn"` alongside `tool_use` blocks and never
    // calls the tool back — observed on chatgpt.com Responses API.
    if (
      state.emittedToolUse &&
      (finalStopReason === null || finalStopReason === "end_turn")
    ) {
      finalStopReason = "tool_use";
    }
    state.finalStopReason = finalStopReason;
    closeThinkingBlock(state, out);
    if (state.textBlockOpen && state.textBlockIndex !== null) {
      const trimmed = state.textAccumulated.trim();
      if (trimmed.length > 0) {
        const safe = ensureCompactionSafeVisibleText(state.textAccumulated);
        if (safe.length > trimmed.length && safe.startsWith(trimmed)) {
          out.push({
            type: "content_block_delta",
            index: state.textBlockIndex,
            delta: { type: "text_delta", text: safe.slice(trimmed.length) },
          });
        }
      }
    }
    closeTextBlock(state, out);
    closeAllToolBlocks(state, out);
    // A signature received while a tool was open is deferred with its
    // reasoning text, then emitted only after the tool block is sealed.
    emitReasoningSignature(state, out);
    closeThinkingBlock(state, out);
    const deferredReasoning = state.thinkingAccumulated.slice(
      state.thinkingDeltaEmittedLen,
    );
    // Unsigned leftover (Kimi / items without encrypted_content) still
    // becomes visible text. Leftover on a sealed signed hop is more
    // summary after the thinking block closed — dumping it is the
    // "polished answer, then the thought again" field sample.
    if (deferredReasoning.length > 0 && !state.sealedSignedThinking) {
      emitVisibleText(
        state,
        out,
        ensureCompactionSafeVisibleText(deferredReasoning),
      );
      state.thinkingDeltaEmittedLen = state.thinkingAccumulated.length;
      closeTextBlock(state, out);
    } else if (
      !state.emittedNonemptyTextDelta &&
      state.sealedSignedThinking &&
      state.thinkingAccumulated.length === 0
    ) {
      // Signed thinking with no human summary still needs a compact-safe
      // text block. Do not clone a non-empty sealed summary (second ⏺).
      emitVisibleText(state, out, ensureCompactionSafeVisibleText(""));
      closeTextBlock(state, out);
    }

    const outputTokens = chunk.usage?.completion_tokens ?? state.outputTokens;
    out.push({
      type: "message_delta",
      delta: {
        stop_reason: state.finalStopReason,
        stop_sequence: null,
      },
      usage: {
        output_tokens: outputTokens,
        input_tokens: state.inputTokens,
        ...(state.cacheCreationTokens > 0
          ? { cache_creation_input_tokens: state.cacheCreationTokens }
          : {}),
        ...(state.cacheReadTokens > 0
          ? { cache_read_input_tokens: state.cacheReadTokens }
          : {}),
        ...(state.serverSearchCount > 0
          ? {
              server_tool_use: {
                web_search_requests: state.serverSearchCount,
              },
            }
          : {}),
      },
    });
    out.push({ type: "message_stop" });
    state.messageStopEmitted = true;
  }
  return out;
};

/**
 * Encode an Anthropic stream event as SSE bytes. Anthropic uses an
 * `event: <name>\n` prefix line (not just `data:` like OpenAI), so we
 * can't reuse `encodeSseEvent` directly.
 */
export const encodeAnthropicSseEvent = (
  event: TAnthropicStreamEvent,
): Uint8Array => {
  const lines = `event: ${event.type}\n`;
  const body = encodeSseEvent(event);
  const prefix = new TextEncoder().encode(lines);
  const out = new Uint8Array(prefix.byteLength + body.byteLength);
  out.set(prefix, 0);
  out.set(body, prefix.byteLength);
  return out;
};

/**
 * Pipe a stream of OpenAI ChatCompletion chunks into Anthropic-format
 * SSE bytes. Used by the `/v1/messages` handler when the runner
 * returned a streaming outcome.
 */
export const chunksToMessagesSseBytes = (
  chunks: ReadableStream<TChatCompletionChunk>,
): ReadableStream<Uint8Array> => {
  const reader = chunks.getReader();
  const state = newMessagesStreamState();
  const buffer: Uint8Array[] = [];
  // One-chunk lookahead. The OpenAI streaming spec delivers token
  // counts in a SEPARATE trailing chunk (`choices: []`, `usage` set,
  // emitted under `stream_options.include_usage`) that arrives AFTER
  // the `finish_reason` chunk. The terminal `message_delta` is built
  // from the finish chunk, so without folding that trailing usage in
  // first the CLI feed gets `usage:{input_tokens:0,output_tokens:0}` —
  // i.e. no token counter once you go through the proxy. Providers
  // that already put usage on the finish chunk are untouched.
  let pending: TChatCompletionChunk | null = null;
  const readChunk = async (): Promise<
    { value: TChatCompletionChunk; done: false } | { done: true }
  > => {
    if (pending !== null) {
      const v = pending;
      pending = null;
      return { value: v, done: false };
    }
    const r = await reader.read();
    return r.done ? { done: true } : { value: r.value, done: false };
  };
  const isUsageOnly = (c: TChatCompletionChunk): boolean =>
    c.choices.length === 0 && c.usage != null;
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        for (;;) {
          if (buffer.length > 0) {
            const next = buffer.shift();
            if (next !== undefined) controller.enqueue(next);
            return;
          }
          const read = await readChunk();
          if (read.done) {
            if (state.startEmitted && !state.messageStopEmitted) {
              // Upstream ended WITHOUT a finish_reason. Mid-tool / mid-
              // text tears (no real usage trailer) stay `length` →
              // Anthropic `max_tokens` so Claude Code does not treat a
              // truncated tool_use as executable. A usage trailer with
              // prompt_tokens > 0 and no open tool is a clean
              // OpenAI-compat close (DashScope / Kimi often omit
              // finish_reason) → `stop` / `end_turn`. Live-estimate
              // frames keep prompt_tokens at 0 and do not flip this.
              // Only the LAST chunk's usage counts: a usage frame
              // followed by more content is a tear, not a clean close.
              const torn =
                state.openToolContentIndexes.size > 0 ||
                state.pendingToolCalls.size > 0;
              const finishReason =
                !torn && state.lastChunkHadProviderUsage ? "stop" : "length";
              const tail = chunkToMessagesEvents(
                {
                  id:
                    state.messageId !== ""
                      ? state.messageId
                      : "chatcmpl-truncated",
                  object: "chat.completion.chunk",
                  created: Math.floor(Date.now() / 1000),
                  model: state.model !== "" ? state.model : "unknown",
                  // The trailer was already consumed; fold last-seen
                  // usage onto this finish chunk so the terminal
                  // `message_delta` reconciles to the provider totals
                  // instead of `output_tokens: 0`.
                  usage: {
                    prompt_tokens: state.inputTokens,
                    completion_tokens: state.outputTokens,
                    total_tokens: state.inputTokens + state.outputTokens,
                  },
                  choices: [
                    {
                      index: 0,
                      delta: {},
                      finish_reason: finishReason,
                    },
                  ],
                },
                state,
              );
              for (const e of tail) {
                buffer.push(encodeAnthropicSseEvent(e));
              }
            }
            while (buffer.length > 0) {
              const next = buffer.shift();
              if (next !== undefined) controller.enqueue(next);
            }
            controller.close();
            return;
          }
          let value = read.value;
          // If this chunk ends the turn but has no usage yet, peek the
          // next one: a trailing usage-only chunk gets folded in so the
          // terminal `message_delta` carries real input/output/cache
          // tokens. Anything else is stashed and processed next.
          const endsTurn = value.choices.some(
            (c) => c.finish_reason != null && c.finish_reason !== undefined,
          );
          if (endsTurn && value.usage == null) {
            const la = await readChunk();
            if (!la.done) {
              if (isUsageOnly(la.value)) {
                value = { ...value, usage: la.value.usage };
              } else {
                pending = la.value;
              }
            }
          }
          const events = chunkToMessagesEvents(value, state);
          for (const e of events) buffer.push(encodeAnthropicSseEvent(e));
        }
      } catch (err) {
        // Upstream errored mid-stream — for example Anthropic's
        // `event: error` (overloaded_error/api_error). Surface as an
        // Anthropic-format error event so Claude Code sees a clean
        // failure rather than an abruptly closed stream (which is
        // exactly what manifested as "compaction fails at 20%").
        const { type, message } = upstreamErrorFrom(err);
        buffer.push(
          encodeAnthropicSseEvent({
            type: "error",
            error: { type, message },
          }),
        );
        const next = buffer.shift();
        if (next !== undefined) controller.enqueue(next);
        controller.close();
      }
    },
    cancel(reason) {
      reader.cancel(reason).catch(() => {});
    },
  });
};
