# `packages/wire` Architecture

> `@openllmsh/wire` — the **wire-format transforms** shared by the cloud
> proxy pipeline (`packages/core`) and the local subscription daemon
> (`packages/daemon`). Depends on `@openllmsh/protocol` and `effect`'s
> `Schema` decoder only: **no Effect-DI, no provider registry, no `fetch`, no
> HTTP framework, no DB.** It provides pure request/response transforms and
> self-contained stream transforms over protocol wire types.
>
> Extracted from `packages/core` per
> [`docs/proposals/coreless-daemon-passthrough.md`](../../docs/proposals/coreless-daemon-passthrough.md)
> §4 + §7.1. Referenced from the root
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md) §3.

---

## Why this package exists

The daemon must adapt a request to each hop's wire, encode the response
back, and synthesize usage — **without** linking the whole `core`
pipeline (the runner + provider graph + Effect-DI layers). Those transforms
were extracted from `core` into a protocol- and Schema-only package so the
daemon can import them directly. The gateway re-imports the shared client
adapters, while `core` re-imports provider-wire transforms, SSE utilities, and
policy helpers; provider registration and runner orchestration remain outside
this package. Shared transform behavior is pinned by the existing
`tests/{inbound,outbound}-*` and matrix suites.

## Modules

```
wire/
  index.ts                              root barrel for shared adapters, features, canonical,
                                        streaming, refusal, encrypted-content, and thinking helpers
  adapters/
    messages/                           Anthropic Messages ↔ canonical request/response/SSE;
                                        Anthropic mapping and reasoning-signature/item helpers
    responses/                          OpenAI Responses ↔ canonical request/response/SSE
  providers/
    upstream-request.ts                 one request body/header recipe for every client/upstream wire pair
    anthropic/                          request/response/SSE transforms, adaptive thinking, beta headers
    chatgpt/                            Codex request/SSE transforms, auth/endpoints, native web search
    grok|kimi/web-search.ts             native web-search request transforms
  features/
    compaction/{compact-request,
      compaction-text}.ts               last-resort cache-aware request compaction and visible-text safety
    context-{demote,overflow-strategy,
      skip}.ts                          context ladder selection, overflow policy, and fit targets
    max-tokens-backfill.ts              output-token cap normalization
    quota-gate.ts                       subscription-quota admission decision
  lib/
    canonical/{content-part,message,
      encoding-select,token-estimate}.ts image detection, canonical utilities, and BPE token estimates
    token-ruler/                        lazy Claude/o200k BPE counters and vocabularies
    streaming/{sse,provider-decode,
      response-stream,accumulate,peek,
      strip-tool-calls,upstream-error}.ts SSE framing/liveness, generic provider decode,
                                        canonical chunk conversion/collection, peeking, and errors
    {encrypted-content,error-class,
      forwarded-headers,refusal,
      thinking-signature,tool-schema}.ts request/error/header and provider-compatibility helpers
```

The root barrel intentionally does not expose every provider or streaming helper;
package export subpaths expose `adapters/messages`, `adapters/responses`,
`providers/anthropic`, `providers/chatgpt`, and individual source modules.

## Layering rules

- Depends only on `@openllmsh/protocol` plus `effect`'s `Schema` decoder. It
  has no `@openllm/core`, `@openllm/api`, `db`, or `vault` dependency; it does
  not use Effect `Layer`, `Context`, or `Effect`, nor Next/Vercel, `fetch`, or
  a provider registry.
- Transforms do not perform network or persistence I/O and have no DI. The
  lazy BPE ruler cache and debug-only stream logging are local implementation
  details; request/response transforms run identically in the cloud pipeline
  and compiled daemon.
- `wire` owns client adapters (Messages and Responses), upstream request
  composition, provider-specific Anthropic/ChatGPT wire transforms, canonical
  SSE framing, generic provider-SSE decoding, and canonical-chunk encoding.
  `core` retains `providerEventStream`, which binds a `TChatProviderSpec` to
  the same SSE/Schema mechanics, plus provider registration, fallback and
  runner orchestration. The daemon remains `core`-free: its walker composes
  wire transforms with local subscription delegation and transport.
- Compaction helpers in `features/compaction` are last-resort, immutable
  request rewriting after the fallback chain is exhausted. Context sizing,
  overflow strategy, output-cap backfill, and quota decisions are wire policy
  helpers; executing fallback or a compaction attempt is owned by `core` or the
  daemon walker.
