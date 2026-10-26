/**
 * Scrub credential material from upstream text before it leaves the gateway:
 * error envelopes returned to a caller and debug logs. Removes each exact
 * secret, its masked-tail and prefix forms, and common key/Bearer patterns.
 * Pure and dependency-free so `core` (runner debug logs), `api` (team
 * credential hops) and the daemon share one implementation.
 */

const SK_LIKE = /\b(?:sk|rk|key)[-_][A-Za-z0-9_-]{8,}\b/gi;
const BEARER_LIKE = /\bBearer\s+[A-Za-z0-9._\-+/=]{8,}/gi;
const LAST4_HINT = /\b(?:\*{2,}|•{2,}|\.{2,}|x{2,})[A-Za-z0-9]{4}\b/gi;

const escapeRegExp = (value: string): string =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * Remove exact secrets and common masked/prefix forms from a string.
 * `secrets` should include the full api_key and any base_url that must not
 * leak. Returns a new string; never throws.
 */
export const scrubSecretText = (
  text: string,
  secrets: ReadonlyArray<string> = [],
): string => {
  let out = text;
  for (const secret of secrets) {
    if (secret.length === 0) continue;
    out = out.split(secret).join("[redacted]");
    if (secret.length >= 4) {
      const last4 = secret.slice(-4);
      const last4Re = new RegExp(
        `(?:\\*{2,}|\\u2022{2,}|\\.{2,}|x{2,}|…)${escapeRegExp(last4)}`,
        "gi",
      );
      out = out.replace(last4Re, "[redacted]");
      // Also scrub a bare "...abcd" / "sk-...abcd" style suffix when the
      // upstream echoes only the tail.
      const tailRe = new RegExp(
        `\\b[A-Za-z0-9_-]*${escapeRegExp(last4)}\\b`,
        "g",
      );
      // Only rewrite tokens that are longer than the last4 alone (avoid
      // nuking unrelated 4-char words).
      out = out.replace(tailRe, (match) =>
        match.length > 4 ? "[redacted]" : match,
      );
    }
    if (secret.length >= 8) {
      const prefix = secret.slice(0, 8);
      const prefixRe = new RegExp(
        `${escapeRegExp(prefix)}[A-Za-z0-9._\\-+/=]*`,
        "g",
      );
      out = out.replace(prefixRe, "[redacted]");
    }
  }
  out = out.replace(SK_LIKE, "[redacted]");
  out = out.replace(BEARER_LIKE, "Bearer [redacted]");
  out = out.replace(LAST4_HINT, "[redacted]");
  return out;
};

export type TScrubbableErrorEnvelope = {
  readonly error: {
    readonly message: string;
    readonly type: string;
    readonly code?: string | null;
    readonly param?: string | null;
  };
};

/** Return a copy of an OpenAI-shaped error envelope with secrets scrubbed. */
export const scrubSecretErrorEnvelope = <T extends TScrubbableErrorEnvelope>(
  envelope: T,
  secrets: ReadonlyArray<string>,
): T => {
  const err = envelope.error;
  return {
    ...envelope,
    error: {
      ...err,
      message: scrubSecretText(err.message, secrets),
      type: scrubSecretText(err.type, secrets),
      ...(err.code !== undefined && err.code !== null
        ? { code: scrubSecretText(String(err.code), secrets) }
        : err.code === null
          ? { code: null }
          : {}),
      ...(err.param !== undefined && err.param !== null
        ? { param: scrubSecretText(String(err.param), secrets) }
        : err.param === null
          ? { param: null }
          : {}),
    },
  };
};
