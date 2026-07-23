/**
 * Recursively remove JSON-Schema keywords a provider rejects from a tool
 * parameter schema (xAI 400s on `minContains`/`maxContains` — its partner
 * client strips exactly these before every request; see
 * docs/audit/2026-07-14-grok-upstream-wire-openclaw-comparison.md §F7).
 *
 * Values under name-map containers (`properties` / `patternProperties` /
 * `$defs` / `definitions`) are themselves schemas, but their KEYS are
 * user-chosen property/definition names — a property named `minContains`
 * must survive — so those maps are descended value-by-value without key
 * filtering. Everything else inside a parameters schema is schema-shaped
 * and is filtered + descended.
 */

const NAME_MAP_CONTAINERS: ReadonlySet<string> = new Set([
  "properties",
  "patternProperties",
  "$defs",
  "definitions",
]);

export const stripSchemaKeywords = (
  schema: unknown,
  keywords: ReadonlyArray<string>,
): unknown => {
  if (schema === null || typeof schema !== "object") return schema;
  if (Array.isArray(schema)) {
    return schema.map((entry) => stripSchemaKeywords(entry, keywords));
  }
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(schema)) {
    if (keywords.includes(key)) continue;
    if (
      NAME_MAP_CONTAINERS.has(key) &&
      value !== null &&
      typeof value === "object" &&
      !Array.isArray(value)
    ) {
      out[key] = Object.fromEntries(
        Object.entries(value).map(([name, sub]) => [
          name,
          stripSchemaKeywords(sub, keywords),
        ]),
      );
      continue;
    }
    out[key] = stripSchemaKeywords(value, keywords);
  }
  return out;
};

/**
 * Rewrite a tool `parameters` schema so every LOCAL definitions `$ref` points at
 * `#/$defs/…`, the only base Kimi/Moonshot's endpoint accepts (it 400s otherwise:
 * "not a valid moonshot flavored json schema … references must start with
 * #/$defs/"; see docs/proposals/kimi-tool-schema-ref-normalization.md).
 *
 * Two normalizations, both semantics-preserving:
 *   1. `$ref` string values `#/definitions/<name>` → `#/$defs/<name>`
 *      (already-`#/$defs/…` and non-local/external refs — `http…`, a live
 *      `#/properties/…` pointer — are left untouched; rewriting those would
 *      change meaning).
 *   2. A `definitions` map is merged into `$defs`. On a name collision
 *      (`definitions.Foo` AND `$defs.Foo` both present) the existing `$defs`
 *      entry is kept and the legacy one is re-homed under a fresh unique name,
 *      with only the `#/definitions/Foo` refs repointed there — so NEITHER
 *      schema is dropped.
 *
 * Recursion descends ONLY into schema-valued positions: name-map containers
 * (`properties`/`patternProperties`/`$defs`/`definitions` — values are schemas,
 * keys are user names) and everything else EXCEPT the instance/annotation
 * keywords (`const`/`default`/`examples`/`enum`), whose values are arbitrary
 * instance data and are copied verbatim (a `$ref`-looking string inside a
 * `const` must not be rewritten). `$ref` rewriting only ever touches a `$ref`
 * whose VALUE is a string.
 */
const DEFINITIONS_REF_PREFIX = "#/definitions/";

// Keywords whose VALUES are arbitrary instance/annotation data, not subschemas —
// never descended, so schema-shaped data inside them is preserved verbatim.
const INSTANCE_VALUE_KEYWORDS: ReadonlySet<string> = new Set([
  "const",
  "default",
  "examples",
  "enum",
]);

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === "object" && !Array.isArray(v);

// JSON Pointer token escaping (RFC 6901): `~1`→`/`, `~0`→`~`. The definition
// NAME is one token; a `$ref` may address INTO it (`#/definitions/Foo/props/x`),
// so only the first token is the name — the rest is an untouched suffix.
const decodePointerToken = (t: string): string =>
  t.replace(/~1/g, "/").replace(/~0/g, "~");
const encodePointerToken = (t: string): string =>
  t.replace(/~/g, "~0").replace(/\//g, "~1");

/** Rewrite one `$ref` string's legacy-definitions base to `#/$defs/`, honoring a
 *  per-name remap for collision-relocated definitions. Only the first pointer
 *  token (the definition name) is remapped; any deeper pointer suffix into that
 *  definition is preserved verbatim, and JSON-Pointer escaping is decoded for
 *  the remap lookup and re-encoded on output. */
const rewriteRef = (
  ref: string,
  remap: ReadonlyMap<string, string>,
): string => {
  if (!ref.startsWith(DEFINITIONS_REF_PREFIX)) return ref;
  const rest = ref.slice(DEFINITIONS_REF_PREFIX.length);
  const slash = rest.indexOf("/");
  const nameToken = slash === -1 ? rest : rest.slice(0, slash);
  const suffix = slash === -1 ? "" : rest.slice(slash); // includes leading "/"
  const name = decodePointerToken(nameToken);
  const target = remap.get(name) ?? name;
  return `#/$defs/${encodePointerToken(target)}${suffix}`;
};

// Transform ONE (key, value) of a schema object per the ref-normalization rules.
// The single source of truth shared by both `normalizeRefsInner` and
// `normalizeSchemaRefs` so they can't drift (the root function only additionally
// special-cases the `definitions`/`$defs` merge).
const normalizeKeyValue = (
  key: string,
  value: unknown,
  remap: ReadonlyMap<string, string>,
): unknown => {
  if (key === "$ref" && typeof value === "string") {
    return rewriteRef(value, remap);
  }
  // Instance/annotation data — copy verbatim, never descend.
  if (INSTANCE_VALUE_KEYWORDS.has(key)) return value;
  // Name-map container — keys are user names, values are schemas.
  if (NAME_MAP_CONTAINERS.has(key) && isPlainObject(value)) {
    return normalizeMap(value, remap);
  }
  return normalizeRefsInner(value, remap);
};

// Normalize every value of a name-map container / definitions map. `fromEntries`
// DEFINES own data properties, so a user key literally named `__proto__` stays
// data and can't pollute the object's prototype (the `out[key] = …` assignment
// form would instead trip the `__proto__` setter).
const normalizeMap = (
  map: Record<string, unknown>,
  remap: ReadonlyMap<string, string>,
): Record<string, unknown> =>
  Object.fromEntries(
    Object.entries(map).map(([name, sub]) => [
      name,
      normalizeRefsInner(sub, remap),
    ]),
  );

const normalizeRefsInner = (
  schema: unknown,
  remap: ReadonlyMap<string, string>,
): unknown => {
  if (!isPlainObject(schema)) {
    return Array.isArray(schema)
      ? schema.map((entry) => normalizeRefsInner(entry, remap))
      : schema;
  }
  return Object.fromEntries(
    Object.entries(schema).map(([key, value]) => [
      key,
      normalizeKeyValue(key, value, remap),
    ]),
  );
};

export const normalizeSchemaRefs = (schema: unknown): unknown => {
  if (!isPlainObject(schema)) return normalizeRefsInner(schema, new Map());

  // Plan the `definitions` → `$defs` merge FIRST so ref rewriting can honor
  // collision relocations in a single pass. Only the ROOT `definitions`/`$defs`
  // participate (JSON-Schema resolves `#/definitions/*` / `#/$defs/*` from the
  // document root); nested maps are descended as ordinary schemas.
  const rootDefinitions = isPlainObject(schema.definitions)
    ? schema.definitions
    : null;
  const rootDefs = isPlainObject(schema.$defs) ? schema.$defs : null;
  const remap = new Map<string, string>();
  // `fromEntries` keeps a literal `__proto__` definition name as own data.
  const mergedEntries: Array<[string, unknown]> = Object.entries(
    rootDefs ?? {},
  );
  const taken = new Set(mergedEntries.map(([name]) => name));

  if (rootDefinitions !== null) {
    for (const [name, sub] of Object.entries(rootDefinitions)) {
      let target = name;
      if (taken.has(name)) {
        // Collision: keep the existing `$defs.<name>`, re-home the legacy one.
        let i = 2;
        while (taken.has(`${name}_${i}`)) i += 1;
        target = `${name}_${i}`;
        remap.set(name, target);
      }
      taken.add(target);
      // `sub` is normalized below (once the remap is complete).
      mergedEntries.push([target, sub]);
    }
  }

  const hasDefs = rootDefinitions !== null || rootDefs !== null;
  const out: Record<string, unknown> = Object.fromEntries(
    Object.entries(schema)
      .filter(([key]) => key !== "definitions" && key !== "$defs")
      .map(([key, value]) => [key, normalizeKeyValue(key, value, remap)]),
  );
  if (hasDefs) {
    out.$defs = Object.fromEntries(
      mergedEntries.map(([name, sub]) => [
        name,
        normalizeRefsInner(sub, remap),
      ]),
    );
  }
  return out;
};
