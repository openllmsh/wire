import type {
  TChatGptProviderOptions,
  TErrorEnvelope,
} from "@quantidexyz/openllmp";

// The Codex backend the daemon's delegated bearer token addresses. The
// OAuth/device-code endpoints + client id are gone — subscription sign-in
// is the official Codex CLI's job (daemon-delegated), not ours.
export const CHATGPT_API_BASE = "https://chatgpt.com/backend-api/codex";

/**
 * The Codex preamble, copied verbatim from
 * `common_utils.py:25-105`. The chatgpt.com endpoint (especially gpt-5.4)
 * requires this preamble in `instructions` — without it the server
 * replies with `response.completed` carrying `output=[]`.
 */
export const CHATGPT_DEFAULT_INSTRUCTIONS = `You are Codex, based on GPT-5. You are running as a coding agent in the Codex CLI on a user's computer.

## General

- When searching for text or files, prefer using \`rg\` or \`rg --files\` respectively because \`rg\` is much faster than alternatives like \`grep\`. (If the \`rg\` command is not found, then use alternatives.)

## Editing constraints

- Default to ASCII when editing or creating files. Only introduce non-ASCII or other Unicode characters when there is a clear justification and the file already uses them.
- Add succinct code comments that explain what is going on if code is not self-explanatory. You should not add comments like "Assigns the value to the variable", but a brief comment might be useful ahead of a complex code block that the user would otherwise have to spend time parsing out. Usage of these comments should be rare.
- Try to use apply_patch for single file edits, but it is fine to explore other options to make the edit if it does not work well. Do not use apply_patch for changes that are auto-generated (i.e. generating package.json or running a lint or format command like gofmt) or when scripting is more efficient (such as search and replacing a string across a codebase).
- You may be in a dirty git worktree.
    * NEVER revert existing changes you did not make unless explicitly requested, since these changes were made by the user.
    * If asked to make a commit or code edits and there are unrelated changes to your work or changes that you didn't make in those files, don't revert those changes.
    * If the changes are in files you've touched recently, you should read carefully and understand how you can work with the changes rather than reverting them.
    * If the changes are in unrelated files, just ignore them and don't revert them.
- Do not amend a commit unless explicitly requested to do so.
- While you are working, you might notice unexpected changes that you didn't make. If this happens, STOP IMMEDIATELY and ask the user how they would like to proceed.
- **NEVER** use destructive commands like \`git reset --hard\` or \`git checkout --\` unless specifically requested or approved by the user.`;

const safeHeaderValue = (value: string): string => {
  let out = "";
  for (let i = 0; i < value.length; i++) {
    const ch = value.charCodeAt(i);
    out += ch >= 32 && ch <= 126 ? value[i] : "_";
  }
  return out;
};

/**
 * Auth + identification headers for the Codex backend.
 *
 * `originator` + `user-agent` are NOT synthesized — they come from the
 * local daemon, which reads them from the official Codex CLI's own
 * identity (see `packages/daemon/src/delegation/chatgpt.ts`) and passes
 * them via `options`. We emit them only when supplied, so the gateway
 * never forges a `codex_cli_rs` fingerprint; the request carries the
 * genuine client's identity.
 */
export const chatGptAuthHeaders = (
  accessToken: string,
  options: TChatGptProviderOptions,
): Record<string, string> => {
  const headers: Record<string, string> = {
    authorization: `Bearer ${accessToken}`,
    "content-type": "application/json",
    accept: "text/event-stream",
  };
  if (options.originator !== undefined && options.originator.length > 0) {
    headers.originator = safeHeaderValue(options.originator);
  }
  if (options.userAgent !== undefined && options.userAgent.length > 0) {
    headers["user-agent"] = safeHeaderValue(options.userAgent);
  }
  if (options.sessionId !== undefined && options.sessionId.length > 0) {
    headers.session_id = safeHeaderValue(options.sessionId);
  }
  if (options.accountId !== undefined && options.accountId.length > 0) {
    headers["ChatGPT-Account-Id"] = safeHeaderValue(options.accountId);
  }
  return headers;
};

/** Endpoint: every chatgpt invocation lands on /backend-api/codex/responses. */
export const chatGptEndpoint = (params: {
  readonly stream: boolean;
  readonly baseUrl?: string;
}): string => {
  const base = (params.baseUrl ?? CHATGPT_API_BASE).replace(/\/+$/, "");
  return `${base}/responses`;
};

/** ChatGPT error envelope — matches the OpenAI shape it usually returns. */
export const chatGptErrorEnvelope = (
  status: number,
  raw: unknown,
): TErrorEnvelope => {
  const fallback = `ChatGPT upstream returned ${status}`;
  if (raw === null || typeof raw === "undefined") {
    return { error: { message: fallback, type: "chatgpt_error" } };
  }
  if (typeof raw === "string") {
    return {
      error: {
        message: raw.length > 0 ? raw : fallback,
        type: "chatgpt_error",
      },
    };
  }
  if (typeof raw !== "object") {
    return {
      error: { message: `${fallback}: ${String(raw)}`, type: "chatgpt_error" },
    };
  }
  const obj = raw as Record<string, unknown>;
  const responseObj = obj.response;
  const inner =
    obj.error ??
    (responseObj !== null && typeof responseObj === "object"
      ? (responseObj as Record<string, unknown>).error
      : undefined);
  if (inner !== null && typeof inner === "object") {
    const i = inner as Record<string, unknown>;
    return {
      error: {
        message: typeof i.message === "string" ? i.message : fallback,
        type: typeof i.type === "string" ? i.type : "chatgpt_error",
        ...(typeof i.code === "string" ? { code: i.code } : {}),
      },
    };
  }
  // Last-resort: surface the raw JSON body so the user can see why
  // upstream rejected the request rather than just a status code.
  return {
    error: {
      message: `${fallback}: ${JSON.stringify(raw)}`,
      type: "chatgpt_error",
    },
  };
};
