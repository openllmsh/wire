import type {
  TProviderUsageSnapshot,
  TSubscriptionMeterMatch,
} from "@openllmsh/protocol";
import { matchesSubscriptionMeter } from "@openllmsh/protocol";

/** Maximum age for a stale quota snapshot to remain eligible for routing. */
export const GATE_STALE_CAP_MS = 10 * 60_000;

export type TQuotaGateDecision =
  | { readonly kind: "allow" }
  | { readonly kind: "skip"; readonly reason: string };

type TQuotaPool = Extract<
  TProviderUsageSnapshot,
  { readonly kind: "quota" }
>["windows"][number];

/**
 * May a STALE snapshot's exhausted `pool` still gate routing?
 *
 * A pool whose reset instant is KNOWN and still ahead is self-validating: the
 * window cannot have refilled before it resets, so the read's age is
 * irrelevant — it gates until that instant, then stops. Age-capping this case
 * made the gate useless for precisely the provider that needs it: an exhausted
 * account serves no successful request, so nothing re-samples its usage, so its
 * snapshot sits permanently past the cap and the dead hop is dialled on every
 * request (grok weekly at 100% with a 6-day-out reset, read 20 min earlier →
 * dialled → HTTP 402 "usage balance exhausted").
 *
 * With NO reset instant there is nothing to validate against, so the freshness
 * cap remains the only evidence and an aged read is not trusted.
 */
const stalePoolIsGateable = (
  snapshot: Extract<TProviderUsageSnapshot, { readonly kind: "quota" }>,
  pool: TQuotaPool | undefined,
  staleCapMs: number,
  now: number,
): boolean => {
  if (!snapshot.stale) return true;
  if (pool?.reset_at_ms !== null && pool?.reset_at_ms !== undefined) {
    return now < pool.reset_at_ms;
  }
  return (
    snapshot.as_of_ms !== undefined && now - snapshot.as_of_ms <= staleCapMs
  );
};

export const quotaGateDecision = (params: {
  readonly snapshot: TProviderUsageSnapshot | null;
  readonly meter: TSubscriptionMeterMatch | undefined;
  readonly finalHop: boolean;
  readonly staleCapMs: number;
  readonly now: number;
}): TQuotaGateDecision => {
  if (params.finalHop || params.snapshot?.kind !== "quota") {
    return { kind: "allow" };
  }

  const { snapshot } = params;
  // A rejected quota snapshot derives from exhausted shared provider windows.
  // For fresh snapshots that vendor verdict is current. For stale snapshots,
  // assess EACH exhausted window independently: an old 5-hour window that has
  // reset cannot erase the still-future 7-day exhaustion in the same snapshot.
  const exhaustedProviderWindow = snapshot.windows.find(
    (window) =>
      window.percent_used >= 100 &&
      stalePoolIsGateable(snapshot, window, params.staleCapMs, params.now),
  );
  if (snapshot.status === "rejected" && exhaustedProviderWindow !== undefined) {
    return { kind: "skip", reason: "quota_skip: provider window exhausted" };
  }

  const exhaustedPool = snapshot.extra_pools?.find(
    (pool) =>
      matchesSubscriptionMeter(params.meter, pool.meter_id) &&
      pool.percent_used >= 100 &&
      stalePoolIsGateable(snapshot, pool, params.staleCapMs, params.now),
  );
  if (exhaustedPool !== undefined) {
    return { kind: "skip", reason: "quota_skip: model meter exhausted" };
  }

  return { kind: "allow" };
};
