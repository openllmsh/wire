import type {
  TChatCompletionRequest,
  TModelSettings,
} from "@openllmsh/protocol";

/** Apply to a fresh hop-local canonical view, never the shared inbound request. */
export const applyModelSettings = (
  request: TChatCompletionRequest,
  settings: TModelSettings | undefined,
): TChatCompletionRequest =>
  request.service_tier != null || settings?.service_tier === undefined
    ? request
    : { ...request, service_tier: settings.service_tier };
