# gah-plugin-toolcall-guard

A **GA-Harness** worker extension plugin (`v0.2.0`) that detects **plain-text / XML / DSML / special-token tool calls** when the LLM fails to emit a structured API `tool_calls` payload, **sanitizes the polluted assistant turn in conversation history** so the model does not self-imitate on retry, surfaces a structured step warning in the Web UI, and automatically prompts the model to regenerate using real structured `tool_calls`.

## Why

During long multi-step sessions (50k–100k+ tokens) or across certain OpenAI-compatible / DeepSeek / Qwen / Kimi / GLM / Minimax / Claude / Llama / Mistral gateways, models occasionally drift into writing tool invocations directly inside the assistant text body instead of structured `tool_calls` — for example:

- **XML tool + parameter blocks** (even when truncated at `max_tokens` without closing `</tool>` or exceeding 8 KB, e.g. large `file_write` / `code_run` scripts):
  ```xml
  <summary>Run verification script</summary>
  <code_run>
  <script>
  import pathlib
  ...
  ```
- **DeepSeek DSML / BPE tokens**: `<｜DSML｜function_calls>`, `<｜DSML｜invoke name="code_run">`, `<｜tool▁calls▁begin｜>`
- **Kimi / GLM / Qwen / Claude / Minimax / Llama / Mistral control markers**: `<|tool_calls_section_begin|>`, `<|tool_call|>`, `<tool_call>{...}</tool_call>`, `<antml:function_calls>`, `<minimax:tool_call>`, `<function=file_read>`, `[TOOL_CALLS]`
- **Raw JSON tool call objects** or **sole-fenced XML pseudo-calls** (` ```xml\n<code_run>...</code_run>\n``` `)

Without `toolcall-guard`, GenericAgent sees `no_tool`, treats the turn as a final plain-text reply, and terminates the task prematurely while dumping raw XML/code into the chat. Even if retried naively, leaving the broken XML inside `backend.history` causes strong in-context self-imitation on subsequent turns.

## Key Capabilities (v0.2.0)

1. **Schema-Aware Multi-Pattern Detection** (`detect_text_tool_call`):
   - Reads active tool names **and** parameter schemas (`script`, `path`, `old_content`, `new_content`, etc.) dynamically from `agent.tools_handler`.
   - Catches `<tool><param>...` blocks **even when truncated mid-stream without a closing `</tool>` tag** or with payloads up to 200 KB.
   - Catches DeepSeek DSML & `▁` BPE tokens, Kimi/GLM/Qwen/Anthropic/Minimax/Llama/Mistral wrappers, raw JSON tool objects, user-configured `extra_markers`, and sole-fenced XML tool calls.
   - Masks `<think>`, `<thinking>`, `<thought>`, `<summary>`, inline backticks, and prose-accompanied markdown code fences to avoid false positives when explaining syntax to users.
2. **History Sanitization Against Self-Imitation** (`sanitize_history = true`):
   - Truncates the leaked plain-text tool markup from the trailing assistant message in `llmclient.backend.history` (across OpenAI, Anthropic, Responses API, and Cascade multi-session backends) while preserving `<summary>`, `thinking` blocks + cryptographic `signature`, and strict `user`/`assistant` role alternation.
   - Resets `llmclient.last_tools = ""` on text-protocol `ToolClient` sessions so tool schemas are freshly re-anchored.
3. **Tool-Specific Retry Guidance & Clean Exhaustion**:
   - Returns `StepOutcome(..., next_prompt=...)` with concrete instructions (e.g. reminding the model that `code_run` can also execute a ```python fenced block when structured JSON escaping is error-prone).
   - Emits structured `tool.call` + `tool.result` (`toolcall_guard`) events so Web UI step logs record every interception clearly, and emits a visible exhaustion notice if `max_retries` is reached.

## Settings

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `max_retries` | integer (`0`–`10`) | `3` | Maximum consecutive regeneration attempts per turn (`0` disables interception). |
| `sanitize_history` | boolean | `true` | Strip plain-text tool call markup from the polluted assistant turn in LLM history before retrying. |
| `extra_markers` | string | `""` | Optional comma-separated custom substrings that also trigger interception. |

## Installation

Copy this folder to `$GAH_DATA_DIR/plugins/toolcall-guard` (on Windows Desktop: `C:\Program Files\ga-harness-desktop\data\plugins\toolcall-guard`), then rescan & enable via the Plugins panel or `plugin_control`. Takes effect starting from the next worker run/turn.
