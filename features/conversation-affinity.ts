/**
 * Conversation affinity — keep one conversation on the backend that last
 * served it, because switching account, supplier or device cold-misses the
 * provider's prompt cache.
 *
 * Every routing layer ranks its candidates (chain hops, personal accounts,
 * team suppliers, supplier devices) from live data that drifts between turns.
 * Affinity sits on top of that ranking: the remembered candidate moves to the
 * front whenever it is still eligible, and the ranking only decides for a new
 * conversation or after the remembered one is excluded (cooling, exhausted,
 * offline, critical failure). Whatever serves next is remembered in its place.
 *
 * Pure: the key derivation, the reorder and an in-process TTL memory. Hosts
 * that need a cross-instance store wrap their own behind the same get/set.
 */

/** Longer than any provider's prompt-cache lifetime. */
export const CONVERSATION_AFFINITY_TTL_MS = 60 * 60 * 1000;
const DEFAULT_MAX_ENTRIES = 10_000;

type TJson = Record<string, unknown>;

const isRecord = (value: unknown): value is TJson =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const nonEmpty = (value: unknown): string | null =>
  typeof value === "string" && value.trim() !== "" ? value : null;

/** 64-bit FNV-1a (two 32-bit lanes) — a routing key, not a secret. */
const hash64 = (text: string): string => {
  let a = 0x811c9dc5;
  let b = 0x01000193 ^ 0x9e3779b9;
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    a = Math.imul(a ^ code, 0x01000193);
    b = Math.imul(b ^ code, 0x01000193 ^ 0x5bd1e995);
  }
  return (
    (a >>> 0).toString(16).padStart(8, "0") +
    (b >>> 0).toString(16).padStart(8, "0")
  );
};

/** The turns of an inbound body, whichever surface it arrived on. */
const turnsOf = (body: TJson): ReadonlyArray<unknown> => {
  if (Array.isArray(body.messages)) return body.messages;
  if (Array.isArray(body.input)) return body.input;
  return [];
};

/**
 * A stable key for the conversation `body` belongs to, or null when the body
 * carries nothing to key on. A client-supplied cache/session id wins (OpenAI
 * `prompt_cache_key`, Anthropic `metadata.user_id`, which Claude Code scopes
 * per session); otherwise the hash of the conversation's fixed prefix — the
 * system prompt and the turns up to and including the first user turn, which
 * never change as the conversation grows.
 */
export const conversationAffinityKey = (body: unknown): string | null => {
  if (!isRecord(body)) return null;
  const cacheKey = nonEmpty(body.prompt_cache_key);
  if (cacheKey !== null) return `k:${hash64(cacheKey)}`;
  const metadata = isRecord(body.metadata) ? body.metadata : null;
  const sessionId = nonEmpty(metadata?.user_id);
  if (sessionId !== null) return `s:${hash64(sessionId)}`;
  const turns = turnsOf(body);
  const firstUser = turns.findIndex(
    (turn) => isRecord(turn) && turn.role === "user",
  );
  // A bare string `input` (Responses) is a one-shot with no history to keep.
  if (firstUser === -1) return null;
  const prefix = JSON.stringify([
    body.system ?? null,
    body.instructions ?? null,
    turns.slice(0, firstUser + 1),
  ]);
  return `p:${hash64(prefix)}`;
};

/** `items` with the remembered one first; unchanged when it isn't present. */
export const preferRemembered = <T>(
  items: ReadonlyArray<T>,
  isRemembered: (item: T) => boolean,
): ReadonlyArray<T> => {
  const index = items.findIndex(isRemembered);
  if (index <= 0) return items;
  return [
    items[index] as T,
    ...items.slice(0, index),
    ...items.slice(index + 1),
  ];
};

export type TAffinityMemory = {
  readonly get: (key: string) => string | undefined;
  readonly set: (key: string, value: string) => void;
};

/** Bounded in-process memory; the oldest entry is evicted first. */
export const createAffinityMemory = (options?: {
  readonly ttlMs?: number;
  readonly maxEntries?: number;
  readonly now?: () => number;
}): TAffinityMemory => {
  const ttlMs = options?.ttlMs ?? CONVERSATION_AFFINITY_TTL_MS;
  const maxEntries = options?.maxEntries ?? DEFAULT_MAX_ENTRIES;
  const now = options?.now ?? Date.now;
  const entries = new Map<string, { value: string; expiresAt: number }>();
  return {
    get: (key) => {
      const entry = entries.get(key);
      if (entry === undefined) return undefined;
      if (entry.expiresAt <= now()) {
        entries.delete(key);
        return undefined;
      }
      return entry.value;
    },
    set: (key, value) => {
      entries.delete(key);
      entries.set(key, { value, expiresAt: now() + ttlMs });
      while (entries.size > maxEntries) {
        const oldest = entries.keys().next().value;
        if (oldest === undefined) break;
        entries.delete(oldest);
      }
    },
  };
};

/** Memory slots, one per routing decision a conversation pins. */
export const affinitySlot = {
  /** The chain hop (model id) that served. */
  hop: (conversation: string): string => `${conversation}|hop`,
  /** The personal account index that served `provider`. */
  account: (conversation: string, provider: string): string =>
    `${conversation}|account|${provider}`,
  /** The team share that served `teamModelId`. */
  share: (conversation: string, teamModelId: string): string =>
    `${conversation}|share|${teamModelId}`,
  /** The supplier device that served `shareId`. */
  device: (conversation: string, shareId: string): string =>
    `${conversation}|device|${shareId}`,
} as const;
