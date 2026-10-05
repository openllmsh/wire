/**
 * Responses-API `namespace` tools ↔ canonical flat function names.
 *
 * Codex groups tools — every MCP server, sub-agent controls, … — as
 * `{type:"namespace", name:"mcp__openllm", tools:[{type:"function", …}]}` and
 * only routes a call back to the tool when the returned `function_call` carries
 * BOTH `namespace` and `name` (a flat `mcp__openllm__tool` is rejected as
 * "unsupported call"). Canonical requests have no namespace concept, so the
 * Responses adapters flatten each member to `<namespace>--<name>` and split it
 * again on the way out. Codex identifiers use `_`, never `--`, so the join is
 * unambiguous; a plain top-level function never contains the separator.
 */

export const RESPONSES_NAMESPACE_SEPARATOR = "--";

/** Canonical function name for a namespace member. */
export const joinNamespacedToolName = (
  namespace: string,
  name: string,
): string => `${namespace}${RESPONSES_NAMESPACE_SEPARATOR}${name}`;

/** Split a canonical name back into its Responses `namespace` + `name`. */
export const splitNamespacedToolName = (
  joined: string,
): { readonly namespace?: string; readonly name: string } => {
  const at = joined.indexOf(RESPONSES_NAMESPACE_SEPARATOR);
  if (at <= 0 || at + RESPONSES_NAMESPACE_SEPARATOR.length >= joined.length) {
    return { name: joined };
  }
  return {
    namespace: joined.slice(0, at),
    name: joined.slice(at + RESPONSES_NAMESPACE_SEPARATOR.length),
  };
};

/** `{ name }` or `{ namespace, name }` — spread into a Responses `function_call`. */
export const responsesCallIdentity = (
  joined: string,
): { readonly name: string; readonly namespace?: string } => {
  const split = splitNamespacedToolName(joined);
  return split.namespace === undefined
    ? { name: split.name }
    : { namespace: split.namespace, name: split.name };
};

/** Canonical name for an incoming Responses item that may carry `namespace`. */
export const canonicalToolNameOf = (
  name: string,
  namespace: unknown,
): string =>
  typeof namespace === "string" && namespace.length > 0
    ? joinNamespacedToolName(namespace, name)
    : name;
