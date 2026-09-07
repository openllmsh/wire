/**
 * Bounded in-process memory of upstreams that do not expose
 * `/v1/messages/count_tokens`. Client-facing preflight must still answer
 * `{ input_tokens }` via the local BPE; it must not re-dial an endpoint
 * already classified as unsupported.
 *
 * Skipping *auth/decrypt* is only possible when the count URL is known
 * without a credential (cloud catalog Anthropic `defaultBaseUrl`). The
 * daemon's captured inference URL is learned from the signed-plan
 * delegate, so `acquireUpstream` still runs; the memory then skips the
 * count HTTP. Identity is never taken from an unsigned caller URL.
 *
 * Scope is provider + endpoint identity (origin + pathname only).
 * Secrets, userinfo, query strings, and API keys never enter the key.
 * Custom endpoints stay isolated from each other and from catalog
 * providers. Model id is NOT part of the key: a missing count *route*
 * is an endpoint property; a missing *model* is not, and must not
 * poison the endpoint.
 */

export const COUNT_TOKENS_CAPABILITY_LIMIT = 256;

const unsupported = new Map<string, true>();

export type TCountTokensCapabilityScope = {
  readonly provider: string;
  readonly endpointIdentity: string;
};

/**
 * Stable identity for a count URL: origin + pathname. Unparseable input
 * (and anything that is not http/https) yields `""` — never the raw
 * string, which may carry query tokens or userinfo.
 */
export const countTokensEndpointIdentity = (countUrl: string): string => {
  try {
    const u = new URL(countUrl);
    if (u.protocol !== "http:" && u.protocol !== "https:") return "";
    return `${u.origin}${u.pathname}`;
  } catch {
    return "";
  }
};

const scopeKey = (scope: TCountTokensCapabilityScope): string =>
  `${scope.provider}\0${scope.endpointIdentity}`;

export const isCountTokensUnsupported = (
  scope: TCountTokensCapabilityScope,
): boolean => {
  if (scope.endpointIdentity.length === 0) return false;
  const k = scopeKey(scope);
  if (!unsupported.has(k)) return false;
  unsupported.delete(k);
  unsupported.set(k, true);
  return true;
};

export const rememberCountTokensUnsupported = (
  scope: TCountTokensCapabilityScope,
): void => {
  if (scope.endpointIdentity.length === 0) return;
  const k = scopeKey(scope);
  if (unsupported.has(k)) {
    unsupported.delete(k);
    unsupported.set(k, true);
    return;
  }
  if (unsupported.size >= COUNT_TOKENS_CAPABILITY_LIMIT) {
    const oldest = unsupported.keys().next().value;
    if (oldest !== undefined) unsupported.delete(oldest);
  }
  unsupported.set(k, true);
};

const modelMissing = (t: string): boolean =>
  /\bmodel\b.{0,120}\b(not found|unknown|does not exist|unsupported|invalid)\b/.test(
    t,
  ) ||
  /\b(not found|unknown|does not exist|unsupported|invalid).{0,120}\bmodel\b/.test(
    t,
  );

const countRouteMissing = (t: string): boolean =>
  /\b(unknown|no such|missing)\s+(endpoint|route|path|url)\b/.test(t) ||
  /\b(endpoint|route|path|url)\s+(is\s+)?(not found|unknown|does not exist|missing)\b/.test(
    t,
  ) ||
  /\bcount_tokens\s+(endpoint|route|path)\s+(not found|does not exist|unknown|not implemented|missing)\b/.test(
    t,
  ) ||
  (t.includes("not implemented") &&
    (t.includes("count_tokens") || t.includes("count tokens"))) ||
  t.includes("method not allowed");

/**
 * Classify an already-attempted count_tokens HTTP response as "this
 * endpoint does not provide the route" vs everything else.
 *
 * 501 / 405 are unambiguous. 404 is only unsupported when the body
 * names a missing *route/endpoint/method* — not a missing *model*,
 * even if the same sentence mentions `count_tokens`.
 * Auth, rate-limit, and server errors must never be remembered.
 */
export const countTokensUnsupportedFromUpstream = (
  status: number,
  bodyText: string,
): boolean => {
  if (status === 401 || status === 403 || status === 429) return false;
  if (status === 501 || status === 405) return true;
  if (status >= 500) return false;
  if (status !== 404) return false;
  const t = bodyText.toLowerCase();
  if (modelMissing(t)) return false;
  return countRouteMissing(t);
};

/** TEST-ONLY: drop remembered unsupported endpoints. */
export const __resetCountTokensCapabilityForTest = (): void => {
  unsupported.clear();
};
