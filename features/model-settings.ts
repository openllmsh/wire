import type {
  TChatCompletionRequest,
  TModelSettings,
} from "@openllmsh/protocol";
import { applyReasoningEffortFloor } from "@openllmsh/protocol";

/** Apply to a fresh hop-local canonical view, never the shared inbound request. */
export const applyModelSettings = (
  request: TChatCompletionRequest,
  settings: TModelSettings | undefined,
): TChatCompletionRequest => {
  let next = request;
  if (request.service_tier == null && settings?.service_tier !== undefined) {
    next = { ...next, service_tier: settings.service_tier };
  }
  const effort = applyReasoningEffortFloor(
    next.reasoning_effort,
    settings?.min_reasoning_effort,
  );
  if (effort !== next.reasoning_effort) {
    next = { ...next, reasoning_effort: effort };
  }
  return next;
};
