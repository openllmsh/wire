const ANTHROPIC_WEB_SEARCH_TOOL_TYPE = /^web_search_\d{8}$/;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/**
 * Removes ambiguous empty domain lists from native Anthropic web-search tools.
 *
 * This only shapes same-wire inbound tool descriptors; canonical Anthropic
 * encoding owns cross-wire tool construction.
 */
export const normaliseAnthropicNativeTools = (
  body: Record<string, unknown>,
): Record<string, unknown> => {
  if (!Array.isArray(body.tools)) return body;

  let changed = false;
  const tools = body.tools.map((tool) => {
    if (
      !isRecord(tool) ||
      typeof tool.type !== "string" ||
      !ANTHROPIC_WEB_SEARCH_TOOL_TYPE.test(tool.type) ||
      ((!Array.isArray(tool.blocked_domains) ||
        tool.blocked_domains.length > 0) &&
        (!Array.isArray(tool.allowed_domains) ||
          tool.allowed_domains.length > 0))
    ) {
      return tool;
    }

    const normalised = { ...tool };
    if (
      Array.isArray(normalised.blocked_domains) &&
      normalised.blocked_domains.length === 0
    ) {
      delete normalised.blocked_domains;
    }
    if (
      Array.isArray(normalised.allowed_domains) &&
      normalised.allowed_domains.length === 0
    ) {
      delete normalised.allowed_domains;
    }
    changed = true;
    return normalised;
  });

  return changed ? { ...body, tools } : body;
};
