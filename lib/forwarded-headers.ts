/**
 * Inbound header allow-list (C1) — the pure policy, shared by the cloud
 * gateway (`@openllm/api`/`@openllm/core`) AND the coreless daemon walker
 * (`@openllmsh/daemon`).
 *
 * Every header in this set survives the proxy hop verbatim — the
 * gateway used to silently strip them, breaking SDK ergonomics and
 * provider-side request correlation. Concretely:
 *
 *   - `user-agent` / `x-app` — SDK identity strings the Anthropic /
 *     OpenAI dashboards group requests by. Without forwarding, every
 *     request appears as the gateway's UA and per-app analytics
 *     collapse to one giant bucket.
 *   - `anthropic-dangerous-direct-browser-access` — Anthropic-specific
 *     CORS-bypass opt-in; clients that set it expect it to reach
 *     api.anthropic.com or they error out.
 *   - `x-stainless-*` — Stainless-generated SDKs (Anthropic, OpenAI)
 *     stamp ~10 of these for runtime/lang/version diagnostics. We
 *     match the prefix because the exact suffix set evolves.
 *   - `anthropic-version` — the Anthropic API version the client
 *     pinned. Without forwarding, every request lands as our
 *     hardcoded default and a client that pinned a newer version
 *     can't tell their pin had no effect. (C2 makes the runner
 *     actually honour it; C1 just carries it through.)
 *   - `openai-organization` / `openai-project` — billing / scoping
 *     headers that OpenAI dashboards respect.
 *   - `openai-beta` — feature-flag opt-ins for preview APIs.
 *   - `accept-language` / `accept-encoding` — locale + transport
 *     preferences. accept-encoding in particular: an inbound `gzip`
 *     should let the upstream send gzip back so we don't pay for an
 *     uncompressed round-trip the client would have accepted.
 *
 * NOT in this set (B2 adds them with a `custom:`-only gate, which stays
 * in `@openllm/api` since it's a gateway concept):
 *   - `idempotency-key` — the client's gateway-side key must not leak
 *     to first-party providers.
 *   - `x-openllm-pin-model` — meaningful only within the openllm
 *     gateway stack.
 *
 * Auth (`authorization` / `x-api-key`) and `content-type` are
 * IDENTIFIED EXPLICITLY by the caller (the provider's `authHeaders`, or
 * the daemon's injected subscription bearer); if a client somehow inserts
 * one of those, the caller's value wins on collision (spread AFTER). Same
 * for `anthropic-beta` — it's composed from inbound + body-derived +
 * OAuth-derived betas, so the inbound is consumed via a separate field,
 * not forwarded as a raw header.
 */
const FORWARD_EXACT: ReadonlySet<string> = new Set([
  "user-agent",
  "x-app",
  "anthropic-dangerous-direct-browser-access",
  "anthropic-version",
  "openai-organization",
  "openai-project",
  "openai-beta",
  "accept-language",
  "accept-encoding",
]);

const FORWARD_PREFIX: ReadonlyArray<string> = ["x-stainless-"];

/**
 * Walks a `Headers` object and returns a flat lowercase map of every
 * header the allow-list matches. Returned record is safe to spread into
 * a fetch `headers` init — caller auth / content-type override on
 * collision because they're spread AFTER.
 */
export const forwardableHeadersFrom = (
  headers: Headers,
): Record<string, string> => {
  const out: Record<string, string> = {};
  headers.forEach((value, key) => {
    const lk = key.toLowerCase();
    if (FORWARD_EXACT.has(lk)) {
      out[lk] = value;
      return;
    }
    for (const prefix of FORWARD_PREFIX) {
      if (lk.startsWith(prefix)) {
        out[lk] = value;
        return;
      }
    }
  });
  return out;
};

/** `Request`-level convenience over {@link forwardableHeadersFrom}. */
export const captureForwardableHeaders = (
  req: Request,
): Record<string, string> => forwardableHeadersFrom(req.headers);

/**
 * Originator passthrough DENYLIST (the daemon's policy).
 *
 * Where the cloud gateway curates an allow-list (it's a multi-tenant BYOK proxy
 * that must NOT leak arbitrary/gateway-internal headers to first-party
 * providers), the local daemon is a transparent reverse proxy in front of the
 * user's OWN subscription CLI: it should carry the ORIGINATOR's headers verbatim
 * so a genuine vendor-CLI request reaches the vendor byte-for-byte, and an
 * unsupported one is rejected by the vendor (terms compliance — we don't launder
 * it). So instead of naming every header to KEEP (which needs updating as SDKs
 * and vendors add headers), we name only the small, stable set to DROP and pass
 * everything else through.
 *
 * Dropped, and why:
 *   - `authorization` / `x-api-key` — the inbound gateway key; the daemon
 *     replaces it with the subscription bearer.
 *   - `host` / `content-length` / `content-type` / `content-encoding` /
 *     `accept-encoding` — transport + body framing the daemon re-derives (it
 *     re-serializes the body and may re-encode the response), plus the wire
 *     builder owns `content-type`.
 *   - hop-by-hop (`connection`, `keep-alive`, `transfer-encoding`, `te`,
 *     `trailer`, `upgrade`, `proxy-authorization`, `proxy-connection`) — never
 *     forwarded across a proxy hop.
 *   - BROWSER-HOP headers: `origin` / `referer` + the `sec-*` client hints
 *     (`sec-fetch-*`, `sec-ch-ua*`, `sec-websocket-*`). These describe the
 *     client→daemon hop, not the daemon→vendor hop, and forwarding them is
 *     actively harmful: an `Origin` makes Anthropic treat the request as a
 *     browser CORS call and reject it ("CORS requests must set
 *     'anthropic-dangerous-direct-browser-access'"). The daemon is a
 *     server-to-server proxy, not a browser.
 *   - `cookie` — the client's cookies are not the vendor's.
 *   - `anthropic-beta` — NOT a raw passthrough: it's composed from inbound +
 *     body-derived + OAuth betas by the wire builder (`inboundBeta`), which
 *     layers the authoritative value on top.
 *
 * NOT dropped (deliberately, vs the old capture-fixture denylist): per-request
 * volatiles like `session-id` / `x-client-request-id` / `x-codex-*`. Those were
 * stripped when we REPLAYED a captured fixture (a stale captured value would
 * pin every request); now we forward the ORIGINATOR's live value, which is
 * correct.
 */
const ORIGINATOR_DENY_EXACT: ReadonlySet<string> = new Set([
  "authorization",
  "x-api-key",
  // The per-boot local caller token is a loopback-only credential — it must
  // never reach a vendor upstream even when a first-party client presents it.
  "x-openllm-local-token",
  "host",
  "content-length",
  "content-type",
  "content-encoding",
  "accept-encoding",
  "transfer-encoding",
  "connection",
  "keep-alive",
  "te",
  "trailer",
  "upgrade",
  "proxy-authorization",
  "proxy-connection",
  "cookie",
  "anthropic-beta",
  // Browser-hop headers (client→daemon only). `origin` in particular makes
  // Anthropic gate the request behind `anthropic-dangerous-direct-browser-access`.
  "origin",
  "referer",
]);

/** Browser-set header families never meant for the daemon→vendor hop:
 *  `sec-fetch-*`, `sec-ch-ua*`, `sec-websocket-*` (codex opens `/responses` as a
 *  WS upgrade; the daemon serves plain HTTP POST). */
const ORIGINATOR_DENY_PREFIX: ReadonlyArray<string> = ["sec-"];

/**
 * Pass through EVERY inbound header except the stable denylist above. Returned
 * record is safe to spread into a fetch `headers` init as the BASE — the daemon
 * spreads the injected bearer / credential-account headers and the wire-derived
 * headers AFTER, so they win on collision.
 */
export const originatorHeadersFrom = (
  headers: Headers,
): Record<string, string> => {
  const out: Record<string, string> = {};
  headers.forEach((value, key) => {
    const lk = key.toLowerCase();
    if (ORIGINATOR_DENY_EXACT.has(lk)) return;
    if (ORIGINATOR_DENY_PREFIX.some((p) => lk.startsWith(p))) return;
    out[lk] = value;
  });
  return out;
};
