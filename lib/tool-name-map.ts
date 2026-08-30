export type TToolNameConstraints = {
  /** A single allowed character. */
  readonly charset: RegExp;
  readonly maxLen: number;
  /** Substitute used for each disallowed character. */
  readonly replacement?: string;
  /** Fallback when sanitization would otherwise produce an empty identifier. */
  readonly fallback?: string;
};

export type TToolNameMaps = {
  /** Original client name → unique upstream name. */
  readonly outbound: ReadonlyMap<string, string>;
  /** Unique upstream name → original client name. */
  readonly inbound: ReadonlyMap<string, string>;
};

type TValidatedToolNameConstraints = {
  readonly charset: RegExp;
  readonly maxLen: number;
  readonly replacement: string;
  readonly fallback: string;
};

const EMPTY_TOOL_NAME_MAP: ReadonlyMap<string, string> = new Map();

export const EMPTY_TOOL_NAME_MAPS: TToolNameMaps = {
  outbound: EMPTY_TOOL_NAME_MAP,
  inbound: EMPTY_TOOL_NAME_MAP,
};

/** Stable 64-bit digest, kept pure for browser and daemon wire transforms. */
const stableHash = (value: string): string => {
  let h1 = 0x811c9dc5;
  let h2 = 0x1000193;
  for (let i = 0; i < value.length; i++) {
    const char = value.charCodeAt(i);
    h1 = Math.imul(h1 ^ char, 0x01000193) >>> 0;
    h2 = Math.imul(h2 ^ char, 0x85ebca77) >>> 0;
  }
  return h1.toString(16).padStart(8, "0") + h2.toString(16).padStart(8, "0");
};

const matchesCharset = (char: string, charset: RegExp): boolean => {
  charset.lastIndex = 0;
  const matches = charset.test(char);
  charset.lastIndex = 0;
  return matches;
};

const isAllowed = (value: string, charset: RegExp): boolean =>
  value.length > 0 &&
  Array.from(value).every((char) => matchesCharset(char, charset));

const validateConstraints = (
  constraints: TToolNameConstraints,
): TValidatedToolNameConstraints => {
  if (!(constraints.charset instanceof RegExp)) {
    throw new Error("Tool identifier constraints require a RegExp charset");
  }
  if (!Number.isSafeInteger(constraints.maxLen) || constraints.maxLen <= 0) {
    throw new Error(
      "Tool identifier constraints require a positive integer maxLen",
    );
  }
  const replacement = constraints.replacement ?? "_";
  const fallback = constraints.fallback ?? "tool";
  if (!isAllowed(replacement, constraints.charset)) {
    throw new Error("Tool identifier replacement must match the charset");
  }
  if (!isAllowed(fallback, constraints.charset)) {
    throw new Error("Tool identifier fallback must match the charset");
  }
  return { ...constraints, replacement, fallback };
};

const hashSuffix = (
  value: string,
  constraints: TValidatedToolNameConstraints,
): string | undefined => {
  const suffix = `_${stableHash(value)}`;
  return suffix.length <= constraints.maxLen &&
    isAllowed(suffix, constraints.charset)
    ? suffix
    : undefined;
};

const truncate = (value: string, maxLen: number): string =>
  Array.from(value).slice(0, maxLen).join("");

/** Sanitize and clamp an identifier without splitting Unicode code points. */
export const sanitizeToolIdentifier = (
  value: string,
  constraints: TToolNameConstraints,
): string => {
  const validated = validateConstraints(constraints);
  const sanitized = Array.from(value)
    .map((char) =>
      matchesCharset(char, validated.charset) ? char : validated.replacement,
    )
    .join("")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
  const nonEmpty = sanitized.length > 0 ? sanitized : validated.fallback;
  if (Array.from(nonEmpty).length <= validated.maxLen) return nonEmpty;
  const suffix = hashSuffix(value, validated);
  if (suffix !== undefined) {
    return `${truncate(nonEmpty, validated.maxLen - suffix.length)}${suffix}`;
  }
  return truncate(nonEmpty, validated.maxLen);
};

const identifierAlphabet = (
  names: ReadonlyArray<string>,
  constraints: TValidatedToolNameConstraints,
): readonly string[] => {
  const candidates = new Set<string>([
    ...Array.from(constraints.replacement),
    ...Array.from(constraints.fallback),
  ]);
  for (const name of names) {
    for (const char of Array.from(name)) {
      if (matchesCharset(char, constraints.charset)) candidates.add(char);
    }
  }
  for (let code = 0x21; code <= 0x7e; code++) {
    const char = String.fromCharCode(code);
    if (matchesCharset(char, constraints.charset)) candidates.add(char);
  }
  return [...candidates].sort();
};

const firstAvailableIdentifier = (
  used: ReadonlySet<string>,
  alphabet: readonly string[],
  maxLen: number,
): string | undefined => {
  let current = [""];
  for (let length = 1; length <= maxLen; length++) {
    const next: string[] = [];
    for (const prefix of current) {
      for (const char of alphabet) {
        const candidate = `${prefix}${char}`;
        if (!used.has(candidate)) return candidate;
        next.push(candidate);
      }
    }
    current = next;
  }
  return undefined;
};

const uniqueName = (
  base: string,
  original: string,
  used: ReadonlySet<string>,
  constraints: TValidatedToolNameConstraints,
  alphabet: readonly string[],
): string => {
  if (!used.has(base)) return base;
  const suffix = hashSuffix(original, constraints);
  if (suffix !== undefined) {
    const candidate = `${truncate(base, constraints.maxLen - suffix.length)}${suffix}`;
    if (!used.has(candidate)) return candidate;
  }
  const fallback = firstAvailableIdentifier(used, alphabet, constraints.maxLen);
  if (fallback !== undefined) return fallback;
  throw new Error("Tool identifier namespace exhausted");
};

const distinctNames = (names: ReadonlyArray<string>): string[] => {
  const distinct: string[] = [];
  const seen = new Set<string>();
  for (const name of names) {
    if (seen.has(name)) continue;
    seen.add(name);
    distinct.push(name);
  }
  return distinct;
};

/**
 * Create request-scoped paired maps (original client name → unique upstream name).
 * A non-empty map is returned only when an identifier is actually lossy, so a set
 * of already-valid names yields an empty map and a byte-identical request.
 *
 * Duplicate and empty declarations are the client's problem, NOT ours: we do NOT
 * throw (an uncaught throw in the hop would 500 rather than let the upstream return
 * a clean 400, and it would diverge from the chatgpt path, which dedups). Duplicates
 * reuse the first assignment; an empty name falls back through
 * `sanitizeToolIdentifier`. Distinct originals are kept distinct until the finite
 * representable namespace is exhausted, at which point construction fails.
 */
export const buildToolNameMaps = (
  names: ReadonlyArray<string>,
  constraints: TToolNameConstraints,
): TToolNameMaps => {
  const validated = validateConstraints(constraints);
  const distinct = distinctNames(names);
  if (
    distinct.every(
      (name) =>
        isAllowed(name, validated.charset) &&
        Array.from(name).length <= validated.maxLen,
    )
  ) {
    return EMPTY_TOOL_NAME_MAPS;
  }

  const outbound = new Map<string, string>();
  const inbound = new Map<string, string>();
  const used = new Set<string>();
  const alphabet = identifierAlphabet(distinct, validated);
  for (const original of distinct) {
    const assigned = uniqueName(
      sanitizeToolIdentifier(original, validated),
      original,
      used,
      validated,
      alphabet,
    );
    outbound.set(original, assigned);
    used.add(assigned);
    if (assigned !== original) inbound.set(assigned, original);
  }
  return { outbound, inbound };
};

/**
 * Request-scoped collision-safe map for tool-call IDs (original → unique upstream
 * id). Unlike tool NAMES, an id legitimately REPEATS across an assistant
 * `tool_call` and its matching `tool_result`, so duplicates are DEDUPED here (not
 * rejected) and every DISTINCT original id is guaranteed a distinct sanitized id.
 * Without this, two client ids differing only in disallowed characters (`call#1`
 * vs `call@1`, or two >maxLen ids sharing a prefix) would collapse to one, and the
 * upstream could not pair `tool_result` → `tool_use`. Both emission sites look the
 * id up by its ORIGINAL value, so pairing is preserved. Forward-only: the upstream
 * mints its own ids on the response, so no reverse map is needed. Returns an empty
 * map when every distinct id already satisfies the constraints (no rewrite).
 */
export const buildToolCallIdMap = (
  ids: ReadonlyArray<string>,
  constraints: TToolNameConstraints,
): ReadonlyMap<string, string> => {
  const validated = validateConstraints(constraints);
  const distinct = distinctNames(ids);
  if (
    distinct.every(
      (id) =>
        isAllowed(id, validated.charset) &&
        Array.from(id).length <= validated.maxLen,
    )
  ) {
    return EMPTY_TOOL_NAME_MAP;
  }
  const outbound = new Map<string, string>();
  const used = new Set<string>();
  const alphabet = identifierAlphabet(distinct, validated);
  for (const original of distinct) {
    const assigned = uniqueName(
      sanitizeToolIdentifier(original, validated),
      original,
      used,
      validated,
      alphabet,
    );
    outbound.set(original, assigned);
    used.add(assigned);
  }
  return outbound;
};
