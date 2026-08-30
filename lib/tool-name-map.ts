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

const isAllowed = (value: string, charset: RegExp): boolean =>
  value.length > 0 && Array.from(value).every((char) => charset.test(char));

/** Sanitize and clamp an identifier without splitting Unicode code points. */
export const sanitizeToolIdentifier = (
  value: string,
  constraints: TToolNameConstraints,
): string => {
  const replacement = constraints.replacement ?? "_";
  const fallback = constraints.fallback ?? "tool";
  const sanitized = Array.from(value)
    .map((char) => (constraints.charset.test(char) ? char : replacement))
    .join("")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
  const nonEmpty = sanitized.length > 0 ? sanitized : fallback;
  const codePoints = Array.from(nonEmpty);
  if (codePoints.length <= constraints.maxLen) return nonEmpty;
  const suffix = `_${stableHash(value)}`;
  return `${codePoints.slice(0, constraints.maxLen - suffix.length).join("")}${suffix}`;
};

const uniqueName = (
  base: string,
  original: string,
  used: ReadonlySet<string>,
  maxLen: number,
): string => {
  if (!used.has(base)) return base;
  const suffix = `_${stableHash(original)}`;
  let candidate = `${Array.from(base)
    .slice(0, maxLen - suffix.length)
    .join("")}${suffix}`;
  // A hash collision is extremely unlikely, but retain deterministic uniqueness.
  for (let retry = 1; used.has(candidate); retry++) {
    const retrySuffix = `_${stableHash(`${original}:${retry}`)}`;
    candidate = `${Array.from(base)
      .slice(0, maxLen - retrySuffix.length)
      .join("")}${retrySuffix}`;
  }
  return candidate;
};

/**
 * Create request-scoped paired maps. Empty and duplicate declarations are adapter
 * errors; a non-empty map is returned only when an identifier is lossy.
 */
export const buildToolNameMaps = (
  names: ReadonlyArray<string>,
  constraints: TToolNameConstraints,
): TToolNameMaps => {
  const seen = new Set<string>();
  for (const name of names) {
    if (name.length === 0)
      throw new Error("Tool function name must not be empty");
    if (seen.has(name))
      throw new Error(`Duplicate tool function name: ${name}`);
    seen.add(name);
  }
  if (
    names.every(
      (name) =>
        isAllowed(name, constraints.charset) &&
        Array.from(name).length <= constraints.maxLen,
    )
  ) {
    return EMPTY_TOOL_NAME_MAPS;
  }

  const outbound = new Map<string, string>();
  const inbound = new Map<string, string>();
  const used = new Set<string>();
  for (const original of names) {
    const assigned = uniqueName(
      sanitizeToolIdentifier(original, constraints),
      original,
      used,
      constraints.maxLen,
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
  const distinct: string[] = [];
  const seen = new Set<string>();
  for (const id of ids) {
    if (seen.has(id)) continue;
    seen.add(id);
    distinct.push(id);
  }
  if (
    distinct.every(
      (id) =>
        isAllowed(id, constraints.charset) &&
        Array.from(id).length <= constraints.maxLen,
    )
  ) {
    return EMPTY_TOOL_NAME_MAP;
  }
  const outbound = new Map<string, string>();
  const used = new Set<string>();
  for (const original of distinct) {
    const assigned = uniqueName(
      sanitizeToolIdentifier(original, constraints),
      original,
      used,
      constraints.maxLen,
    );
    outbound.set(original, assigned);
    used.add(assigned);
  }
  return outbound;
};
