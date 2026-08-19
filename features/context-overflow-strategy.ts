import type { TContextOverflowStrategy as TProtocolContextOverflowStrategy } from "@openllmsh/protocol";

/**
 * Shared context-overflow routing strategy. The cloud and daemon resolve an
 * absent or unrecognised persisted value to the historical hop behaviour.
 */

export type TContextOverflowStrategy = TProtocolContextOverflowStrategy;

export const DEFAULT_CONTEXT_OVERFLOW_STRATEGY: TContextOverflowStrategy =
  "hop_to_larger_context";

export const resolveContextOverflowStrategy = (
  raw: string | undefined | null,
): TContextOverflowStrategy =>
  raw === "compact_in_place"
    ? "compact_in_place"
    : DEFAULT_CONTEXT_OVERFLOW_STRATEGY;

/** Hop mode both demotes on overflow and pre-skips known-too-small hops. */
export const isHopToLargerContext = (
  strategy: TContextOverflowStrategy,
): boolean => strategy === "hop_to_larger_context";

export const shouldDemoteOnContextOverflow = isHopToLargerContext;
export const shouldSkipHopForSize = isHopToLargerContext;
