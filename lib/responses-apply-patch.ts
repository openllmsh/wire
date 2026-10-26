/**
 * OpenAI's hosted Responses `apply_patch` tool ↔ a canonical function tool.
 *
 * Clients built on the OpenAI Agents SDK (Pomerado, the SDK's sandbox agents)
 * declare `{type:"apply_patch"}` and expect `apply_patch_call` output items
 * carrying `{type, path, diff}` operations. No upstream but OpenAI's own API
 * honours the hosted tool (chatgpt.com 400s `Unsupported tool type:
 * apply_patch`), and canonical requests model only function tools. So the
 * inbound adapter declares it as a function under a reserved name, and the
 * outbound encoders turn calls to that name back into `apply_patch_call` items.
 * The reserved name never collides with a client's own `apply_patch` function
 * (older Codex builds declare one with a different argument shape).
 */

export const APPLY_PATCH_FUNCTION_NAME = "openllm_apply_patch";

/** The Responses function definition the hosted tool is declared as. */
export const applyPatchFunctionTool = {
  type: "function",
  name: APPLY_PATCH_FUNCTION_NAME,
  description:
    "Create, update or delete one file. For create_file and update_file, diff is a V4A diff: lines prefixed '+' add, '-' remove and ' ' keep, with '@@' context headers for updates. A create_file diff contains only '+' lines. delete_file takes no diff.",
  parameters: {
    type: "object",
    properties: {
      type: {
        type: "string",
        enum: ["create_file", "update_file", "delete_file"],
      },
      path: { type: "string" },
      diff: { type: "string" },
    },
    required: ["type", "path"],
    additionalProperties: false,
  },
} as const;

export const isHostedApplyPatchTool = (t: unknown): boolean =>
  typeof t === "object" &&
  t !== null &&
  (t as { type?: unknown }).type === "apply_patch";

export type TApplyPatchOperation = {
  readonly type: "create_file" | "update_file" | "delete_file";
  readonly path: string;
  readonly diff?: string;
};

const OPERATION_TYPES = new Set(["create_file", "update_file", "delete_file"]);

/** The operation in a call's JSON arguments, or undefined if malformed. */
export const applyPatchOperationOf = (
  args: string,
): TApplyPatchOperation | undefined => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(args);
  } catch {
    return undefined;
  }
  if (typeof parsed !== "object" || parsed === null) return undefined;
  const { type, path, diff } = parsed as Record<string, unknown>;
  if (typeof type !== "string" || !OPERATION_TYPES.has(type)) return undefined;
  if (typeof path !== "string") return undefined;
  return {
    type: type as TApplyPatchOperation["type"],
    path,
    ...(typeof diff === "string" ? { diff } : {}),
  };
};

/**
 * A finished call to the reserved function as a Responses `apply_patch_call`
 * item, or undefined when its arguments are not a valid operation (the caller
 * then emits a plain `function_call`, which the client reports as an error).
 */
export const toApplyPatchCallItem = (
  id: string,
  callId: string,
  args: string,
):
  | {
      readonly type: "apply_patch_call";
      readonly id: string;
      readonly call_id: string;
      readonly status: "completed";
      readonly operation: TApplyPatchOperation;
    }
  | undefined => {
  const operation = applyPatchOperationOf(args);
  return operation === undefined
    ? undefined
    : {
        type: "apply_patch_call",
        id: id.replace(/^fc_/, "apc_"),
        call_id: callId,
        status: "completed",
        operation,
      };
};
