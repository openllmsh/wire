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
