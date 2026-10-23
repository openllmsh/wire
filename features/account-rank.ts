import type {
  TProviderUsageSnapshot,
  TSubscriptionMeterMatch,
} from "@openllmsh/protocol";
import { matchesSubscriptionMeter } from "@openllmsh/protocol";
import { quotaGateDecision } from "./quota-gate";

/**
 * Order the accounts of ONE subscription provider for a single hop
 * (`docs/plan/multi-account-subscriptions.md` §6):
 *
 *   1. the pinned account, when it is usable;
 *   2. accounts WITH quota data, least remaining first (drain the nearly
 *      empty one before touching a fuller one), ties by index;
 *   3. accounts WITHOUT usable quota data, by index (they rely on cooldown);
 *   4. exhausted accounts are left out — except that the caller's FINAL
 *      account of the FINAL hop may still dial (never-drop-all).
 *
 * A pinned account that is cooled or exhausted falls back into its normal
 * place in the order.
 */

export type TRankableAccount = {
  readonly index: number;
  readonly snapshot: TProviderUsageSnapshot | null;
  /** A local per-account cooldown is live for this hop's model. */
  readonly cooling: boolean;
};

export type TRankedAccount = {
  readonly index: number;
  /** Remaining headroom on the binding constraint, 0..100; null = unknown. */
  readonly remaining: number | null;
};

type TQuotaSnapshot = Extract<
  TProviderUsageSnapshot,
  { readonly kind: "quota" }
>;

/**
 * Headroom on the TIGHTEST constraint that can reject this hop: the most-used
 * provider window (5h, weekly, …) and, when the hop has a catalog meter, the
 * matching extra pool. 10% on 5h but 100% weekly is exhausted.
 */
export const remainingQuotaPercent = (
  snapshot: TQuotaSnapshot,
  meter: TSubscriptionMeterMatch | undefined,
): number => {
  let used = 0;
  for (const window of snapshot.windows)
    used = Math.max(used, window.percent_used);
  for (const pool of snapshot.extra_pools ?? []) {
    if (matchesSubscriptionMeter(meter, pool.meter_id)) {
      used = Math.max(used, pool.percent_used);
    }
  }
  return Math.max(0, Math.min(100, 100 - used));
};

/**
 * A snapshot whose numbers may rank an account. Stale reads are only trusted
 * within the gate's freshness cap — beyond it the account is "unknown" (it
 * still serves, ordered by index, and cooldown catches a dead one).
 */
const trustedQuota = (
  snapshot: TProviderUsageSnapshot | null,
  staleCapMs: number,
  now: number,
): TQuotaSnapshot | null => {
  if (snapshot?.kind !== "quota") return null;
  if (!snapshot.stale) return snapshot;
  return snapshot.as_of_ms !== undefined &&
    now - snapshot.as_of_ms <= staleCapMs
    ? snapshot
    : null;
};

export const rankProviderAccounts = (params: {
  readonly accounts: readonly TRankableAccount[];
  readonly meter: TSubscriptionMeterMatch | undefined;
  readonly pinnedIndex: number | null;
  readonly staleCapMs: number;
  readonly now: number;
}): readonly TRankedAccount[] => {
  const known: TRankedAccount[] = [];
  const unknown: TRankedAccount[] = [];
  for (const account of params.accounts) {
    if (account.cooling) continue;
    const gate = quotaGateDecision({
      snapshot: account.snapshot,
      meter: params.meter,
      finalHop: false,
      staleCapMs: params.staleCapMs,
      now: params.now,
    });
    if (gate.kind === "skip") continue;
    const quota = trustedQuota(account.snapshot, params.staleCapMs, params.now);
    if (quota === null) {
      unknown.push({ index: account.index, remaining: null });
      continue;
    }
    const remaining = remainingQuotaPercent(quota, params.meter);
    // A trusted 0% headroom the gate let through (e.g. status not yet
    // `rejected`) is still exhausted for ordering purposes.
    if (remaining <= 0) continue;
    known.push({ index: account.index, remaining });
  }
  known.sort(
    (a, b) => (a.remaining ?? 0) - (b.remaining ?? 0) || a.index - b.index,
  );
  unknown.sort((a, b) => a.index - b.index);
  const ordered = [...known, ...unknown];
  if (params.pinnedIndex === null) return ordered;
  const pinned = ordered.find((a) => a.index === params.pinnedIndex);
  return pinned === undefined
    ? ordered
    : [pinned, ...ordered.filter((a) => a !== pinned)];
};

/**
 * Never-drop-all pick when every signed-in account is cooled or exhausted on
 * the FINAL hop: still dial one account so the caller gets a real upstream
 * error instead of a synthetic "all cooled" envelope.
 *
 * Deterministic order:
 *   1. the pinned account, when it is among the signed-in set;
 *   2. else the signed-in account with the most remaining *trusted* quota
 *      (ties by lowest index);
 *   3. else the lowest index.
 */
export const pickFinalFallback = (params: {
  readonly accounts: readonly TRankableAccount[];
  readonly meter: TSubscriptionMeterMatch | undefined;
  readonly pinnedIndex: number | null;
  readonly staleCapMs: number;
  readonly now: number;
}): number | undefined => {
  if (params.accounts.length === 0) return undefined;
  if (
    params.pinnedIndex !== null &&
    params.accounts.some((a) => a.index === params.pinnedIndex)
  ) {
    return params.pinnedIndex;
  }
  let best: { readonly index: number; readonly remaining: number } | null =
    null;
  for (const account of params.accounts) {
    const quota = trustedQuota(account.snapshot, params.staleCapMs, params.now);
    if (quota === null) continue;
    const remaining = remainingQuotaPercent(quota, params.meter);
    if (
      best === null ||
      remaining > best.remaining ||
      (remaining === best.remaining && account.index < best.index)
    ) {
      best = { index: account.index, remaining };
    }
  }
  if (best !== null) return best.index;
  let lowest = params.accounts[0]?.index;
  for (const account of params.accounts) {
    if (lowest === undefined || account.index < lowest) lowest = account.index;
  }
  return lowest;
};
