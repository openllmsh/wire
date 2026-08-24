import type { TModelCaps } from "@openllmsh/protocol";

export type { TModelCaps };

/**
 * Catalog-declared, model-family request constraints. These run after wire
 * encoding so the resolved model card can make the final provider-specific
 * adjustment without teaching a wire format about model names.
 */

const otherMaxTokensField = (
  field: NonNullable<TModelCaps["maxTokensField"]>,
): "max_tokens" | "max_completion_tokens" =>
  field === "max_tokens" ? "max_completion_tokens" : "max_tokens";

/**
 * Apply the resolved model card's final outbound-body constraints without
 * mutating the wire encoder's result.
 */
export const applyModelCaps = (
  body: Record<string, unknown>,
  caps: TModelCaps | undefined,
): Record<string, unknown> => {
  if (caps === undefined) return body;

  const next = { ...body };
  for (const key of caps.deniedParams ?? []) {
    delete next[key];
  }

  if (caps.maxTokensField !== undefined) {
    const otherField = otherMaxTokensField(caps.maxTokensField);
    if (otherField in next) {
      if (!(caps.maxTokensField in next)) {
        next[caps.maxTokensField] = next[otherField];
      }
      delete next[otherField];
    }
  }

  if (caps.temperature === "only-1" && next.temperature !== 1) {
    delete next.temperature;
  } else if (caps.temperature === "clamp-0-1") {
    const temperature = next.temperature;
    if (typeof temperature === "number") {
      next.temperature = Math.min(1, Math.max(0, temperature));
    }
  } else if (caps.temperature === "drop") {
    delete next.temperature;
  }

  return next;
};
