"""toolcall-guard: catch plain-text / XML / DSML / token-leaked tool calls and force regeneration.

In long multi-step sessions (e.g. 50k-100k+ context), LLMs sometimes drift into emitting
tool calls as raw body text instead of structured API ``tool_calls`` — such as DeepSeek
``<｜DSML｜function_calls>`` / ``<｜tool▁calls▁begin｜>``, XML parameter blocks like
``<code_run>\\n<script>...`` (often truncated without closing tags), Qwen/Hermes
``<tool_call>...</tool_call>``, Kimi ``<|tool_calls_section_begin|>``, Anthropic
``<function_calls><invoke>``, Llama ``<function=name>``, Mistral ``[TOOL_CALLS]``, or
raw JSON ``{"name": "...", "arguments": {...}}``.

When ``response.tool_calls`` is empty, GenericAgent normally routes the turn to
``do_no_tool`` and terminates the agent loop immediately — mistaking the broken tool
attempt for a final answer.

This plugin wraps ``handler.do_no_tool``:
1. Detects plain-text tool call patterns (aware of active tool names AND their JSON Schema
   parameter names, including unclosed/truncated XML and large payloads) while ignoring
   ``<thinking>``, ``<summary>``, inline code, and normal prose/code examples.
2. Resets retry streak automatically whenever intervening turns execute real tools or a
   new run begins (fixing cumulative lifetime streak exhaustion on long tasks).
3. Sanitizes the polluted tail of the assistant message already appended to
   ``llmclient.backend.history`` (preserving ``thinking`` signatures and ``<summary>``)
   so the model does not imitate its own broken XML/text format on retry.
4. Emits structured ``tool.call`` / ``tool.result`` diagnostic events (when running inside
   a GA-Harness worker) and returns a ``StepOutcome`` ``next_prompt`` instructing the
   model to re-emit the step via structured ``tool_calls``.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import importlib
import json
import re
from typing import Iterable, Mapping

_THINK_RE = re.compile(r"<think(?:ing)?\b[^>]*>[\s\S]*?</think(?:ing)?>", re.I)
_SUMMARY_RE = re.compile(r"<summary\b[^>]*>[\s\S]*?</summary>", re.I)
_FENCED_CODE_RE = re.compile(r"(?:```|~~~)[^\n`~]*\n[\s\S]*?(?:```|~~~)|```[^`\n]+```")
_INLINE_CODE_RE = re.compile(r"`[^`\n]{1,400}`")
_SOLE_FENCE_RE = re.compile(
    r"^\s*(?P<fence>```|~~~)(?P<lang>[^\n`~]*)\n(?P<body>[\s\S]*?)\n?(?P=fence)\s*$"
)

_OMITTED_MARKER = "[plain-text tool call omitted; use structured tool_calls]"

# Common parameter / child tag names used when LLMs hallucinate XML tool calls,
# merged with live JSON Schema property names extracted from active tools.
_COMMON_PARAM_TAGS = frozenset({
    "parameter", "param", "arg", "args", "arguments", "input", "parameters",
    "script", "code", "command", "cmd", "path", "file_path", "filepath",
    "content", "old_content", "new_content", "query", "url", "prompt",
    "images", "action", "name", "question", "key_info", "cwd", "timeout",
    "type", "mode", "start", "count", "switch_tab_id", "tabs_only",
    "text_only", "no_monitor", "save_to_file", "candidates", "model",
    "platform", "max_pixels", "show_linenos", "inline_eval",
})

_DEDICATED_TOOL_FENCE_LANGS = frozenset({
    "tool_call", "tool_calls", "function_call", "function_calls", "tool_use", "dsml",
})
_XML_LIKE_FENCE_LANGS = frozenset({"", "xml", "html", "text", "plaintext", "txt"})

# Model-family special control tokens / wrappers (independent of active tool names).
_SPECIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "dsml",
        re.compile(
            r"<[｜|]\s*DSML\s*[｜|]\s*(?:function_calls|invoke|parameter)\b[^>]*>",
            re.I,
        ),
    ),
    (
        "special-token",
        re.compile(
            r"<[｜|](?:"
            r"tool_calls?_section_begin|tool_calls?_begin|tool_call_argument_begin|"
            r"tool[\u2581_]calls?[\u2581_]begin|"
            r"tool_calls?|plugin|action_start"
            r")[｜|]>|\[TOOL_CALLS\]",
            re.I,
        ),
    ),
    (
        "xml-wrapper",
        re.compile(
            # Closed wrapper whose body contains structured content ({, <, =, or newline)
            r"<\s*(?:antml:)?(?:tool_call|function_calls|minimax:tool_call|tool_use)\b[^>]*>"
            r"(?=[\s\S]*?[\{<=\n])[\s\S]*?"
            r"<\s*/\s*(?:antml:)?(?:tool_call|function_calls|minimax:tool_call|tool_use)\s*>"
            r"|"
            # Or unclosed/truncated wrapper at line start followed by JSON/XML payload
            r"^\s*<\s*(?:antml:)?(?:tool_call|function_calls|minimax:tool_call|tool_use)\b[^>]*>"
            r"\s*(?:\{|<\s*[a-zA-Z_])"
            r"|"
            r"<\s*antml:invoke\b[^>]*>"
            r"|"
            r"^\s*<\s*invoke\s+name\s*=\s*[\"'][^\"']+[\"'][^>]*>"
            r"|"
            r"<\s*function\s*=\s*[\"']?[a-zA-Z_][\w.-]*[\"']?\s*>\s*(?:\{|<|\n|\Z)"
            r"|"
            r"<\s*/?\s*(?:antml:)?(?:tool_call|function_calls|minimax:tool_call|tool_use)\s*[\"':=]*\s*>\s*\Z",
            re.I | re.M,
        ),
    ),
)

_JSON_ARG_KEYS = ("arguments", "parameters", "input", "args")


@dataclass(frozen=True)
class Detection:
    kind: str
    tool: str
    snippet: str
    raw_offset: int = -1


def _mask_keep_newlines(match: re.Match[str]) -> str:
    return re.sub(r"[^\n]", " ", match.group(0))


def _mask_header_tags(text: str) -> str:
    s = _THINK_RE.sub(_mask_keep_newlines, text)
    s = _SUMMARY_RE.sub(_mask_keep_newlines, s)
    return s


def _mask_non_prose(text: str) -> str:
    """Mask thinking, summary, fenced code blocks, and inline code while preserving offsets & newlines."""
    s = _mask_header_tags(text)
    s = _FENCED_CODE_RE.sub(_mask_keep_newlines, s)
    s = _INLINE_CODE_RE.sub(_mask_keep_newlines, s)
    return s


def _compact(snippet: str, limit: int = 140) -> str:
    one_line = " ".join(snippet.strip().split())
    return one_line if len(one_line) <= limit else one_line[: limit - 3] + "..."


def _extract_schema_tool(item: object) -> tuple[str, set[str]]:
    """Return ``(tool_name, set_of_param_names)`` from an OpenAI/GA tool schema dict."""
    if not isinstance(item, dict):
        return "", set()
    fn = item.get("function") if isinstance(item.get("function"), dict) else item
    name = fn.get("name") if isinstance(fn, dict) else None
    if not isinstance(name, str) or not name.strip() or name == "no_tool":
        return "", set()
    params: set[str] = set()
    schema = fn.get("parameters") if isinstance(fn, dict) else None
    if isinstance(schema, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            for k in props:
                if isinstance(k, str) and k:
                    params.add(k.lower())
    return name.strip(), params


def _active_tool_spec(agent, handler) -> tuple[set[str], dict[str, set[str]]]:
    """Collect active tool names and their schema parameter names from handler/agent."""
    names: set[str] = set()
    params_by_tool: dict[str, set[str]] = {}

    for tool_list in (getattr(handler, "tools", None), getattr(agent, "all_tools", None)):
        if not isinstance(tool_list, list):
            continue
        for item in tool_list:
            tname, tparams = _extract_schema_tool(item)
            if tname:
                names.add(tname)
                if tparams:
                    params_by_tool.setdefault(tname, set()).update(tparams)

    extra = getattr(handler, "_extra_tools", None)
    if isinstance(extra, dict):
        for k in extra:
            if isinstance(k, str) and k and k != "no_tool":
                names.add(k)

    for attr in dir(handler):
        if attr.startswith("do_") and len(attr) > 3 and attr != "do_no_tool":
            names.add(attr[3:])

    return names, params_by_tool


def _active_tool_names(agent, handler) -> set[str]:
    names, _ = _active_tool_spec(agent, handler)
    return names


def _infer_tool_from_tail(tail: str, names: set[str]) -> str:
    """Best-effort extraction of an active tool name inside a matched wrapper/token region."""
    if not names or not tail:
        return ""
    window = tail[:600]
    alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    patterns = (
        rf'\bname\s*[:=]\s*["\'](?P<tool>{alt})["\']',
        rf'<\s*function\s*=\s*["\']?(?P<tool>{alt})["\']?\s*>',
        rf'<\s*(?P<tool>{alt})\b',
    )
    for pat in patterns:
        m = re.search(pat, window)
        if m:
            return m.group("tool")
    return ""


def _scan_xml_and_special(
    source_text: str,
    scan_text: str,
    names: set[str],
    params_by_tool: Mapping[str, Iterable[str]],
    base_offset: int = 0,
) -> list[Detection]:
    hits: list[Detection] = []

    for kind, regex in _SPECIAL_PATTERNS:
        m = regex.search(scan_text)
        if m:
            lead_ws = len(m.group(0)) - len(m.group(0).lstrip())
            abs_off = base_offset + m.start() + lead_ws
            inferred = _infer_tool_from_tail(source_text[abs_off:], names)
            hits.append(
                Detection(
                    kind=kind,
                    tool=inferred,
                    snippet=_compact(source_text[abs_off : abs_off + max(len(m.group(0)) - lead_ws, 120)]),
                    raw_offset=abs_off,
                )
            )

    if not names:
        return hits

    alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    lower_param_map: dict[str, set[str]] = {
        k: {p.lower() for p in v} for k, v in params_by_tool.items()
    }

    # 1) <tool_name>\n<param_tag>... — catches unclosed/truncated XML & >8k scripts (e.g. <code_run><script>...)
    param_xml_re = re.compile(
        rf"<\s*(?P<tool>{alt})\s*>\s*<\s*(?P<param>[a-zA-Z_][\w-]*)\b[^>]*>"
    )
    for m in param_xml_re.finditer(scan_text):
        tname = m.group("tool")
        pname = m.group("param").lower()
        allowed = _COMMON_PARAM_TAGS | lower_param_map.get(tname, set())
        if pname in allowed:
            abs_off = base_offset + m.start()
            hits.append(
                Detection(
                    kind="tool-param-xml",
                    tool=tname,
                    snippet=_compact(source_text[abs_off : abs_off + 160]),
                    raw_offset=abs_off,
                )
            )
            break

    # 2) Closed <tool>...</tool> (up to 200k chars), self-closing <tool/>, <tool attr="val">,
    #    <function=tool>, <invoke name="tool">, or line-anchored <tool>\n{...
    closed_re = re.compile(
        rf"<\s*(?P<tool>{alt})\s*>(?P<body>[\s\S]{{0,200000}}?)<\s*/\s*(?P=tool)\s*>"
    )
    for m in closed_re.finditer(scan_text):
        body = m.group("body")
        # Empty <tool></tool> or structured multiline/XML/JSON/key=value body (avoids inline prose "<a> and </a>")
        if not body.strip() or any(ch in body for ch in ("\n", "<", "{", "=")):
            abs_off = base_offset + m.start()
            hits.append(
                Detection(
                    kind="tool-xml",
                    tool=m.group("tool"),
                    snippet=_compact(source_text[abs_off : abs_off + min(len(m.group(0)), 160)]),
                    raw_offset=abs_off,
                )
            )
            break

    other_tool_xml_re = re.compile(
        rf"<\s*(?P<t1>{alt})\s*/>"
        rf"|"
        rf"<\s*(?P<t2>{alt})(?:\s+[a-zA-Z_][\w-]*\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s/>]+))+\s*/?>"
        rf"|"
        rf"^\s*<\s*(?P<t3>{alt})\s*>\s*(?:\n\s*)?\{{"
        rf"|"
        rf"<\s*function\s*=\s*[\"']?(?P<t4>{alt})[\"']?\s*>"
        rf"|"
        rf"<\s*(?:antml:)?(?:tool_call|invoke|tool_use|function_call)\b[^>]*\bname\s*=\s*[\"'](?P<t5>{alt})[\"']",
        re.M,
    )
    m = other_tool_xml_re.search(scan_text)
    if m:
        tname = next((v for k, v in m.groupdict().items() if v), "")
        lead_ws = len(m.group(0)) - len(m.group(0).lstrip())
        abs_off = base_offset + m.start() + lead_ws
        hits.append(
            Detection(
                kind="tool-xml",
                tool=tname,
                snippet=_compact(source_text[abs_off : abs_off + 160]),
                raw_offset=abs_off,
            )
        )

    # 3) Dangling / relay-mangled / stream-truncated tool tags (e.g. trailing "<code_run>" or "<code_run\">"):
    #    - d1: malformed quote/colon/equals right after tool name anywhere (<code_run">, <code_run'>, <code_run")
    #    - d2: standalone <tool_name> line immediately followed by another '<' tag or end-of-text
    #    - d3..d5: trailing <tool_name>, </tool_name>, or mid-tag truncated <tool_name / <invoke name="tool_name at EOF
    dangling_tool_xml_re = re.compile(
        rf"<\s*/?\s*(?P<d1>{alt})\s*[\"':=]+[^>\n]{{0,120}}>?"
        rf"|"
        rf"^\s*<\s*/?\s*(?P<d2>{alt})\s*>[ \t]*(?:\r?\n\s*(?=<)|\Z)"
        rf"|"
        rf"<\s*/?\s*(?P<d3>{alt})\s*>\s*\Z"
        rf"|"
        rf"(?:^|\n)\s*<(?P<d4>{alt})(?:\s+[a-zA-Z_][\w-]*\s*=\s*[^>\n]{{0,200}})?\s*\Z"
        rf"|"
        rf"<\s*(?:(?:antml:)?(?:tool_call|invoke|tool_use|function_call)\b[^>\n]{{0,120}}\bname\s*=\s*[\"']?|function\s*=\s*[\"']?)(?P<d5>{alt})[\"']?[^>\n]{{0,120}}\s*\Z",
        re.M,
    )
    m_dang = dangling_tool_xml_re.search(scan_text)
    if m_dang:
        # Ensure EOF-anchored matches (d3/d4/d5, or d2 at \Z) are truly at the tail of source_text
        # and not merely followed by masked inline-code/fenced prose.
        is_malformed = bool(m_dang.group("d1"))
        after_in_source = source_text[base_offset + m_dang.end() :] if base_offset == 0 else ""
        if is_malformed or not after_in_source.strip() or after_in_source.lstrip().startswith("<"):
            tname = next((v for k, v in m_dang.groupdict().items() if v), "")
            lead_ws = len(m_dang.group(0)) - len(m_dang.group(0).lstrip())
            abs_off = base_offset + m_dang.start() + lead_ws
            hits.append(
                Detection(
                    kind="tool-dangling-xml",
                    tool=tname,
                    snippet=_compact(source_text[abs_off : abs_off + 160]),
                    raw_offset=abs_off,
                )
            )

    return hits


def _match_json_tool_obj(obj: object, names: set[str]) -> str:
    """If ``obj`` looks like a structured JSON tool call for an active tool, return tool name."""
    if isinstance(obj, dict):
        # Direct {"name"|"tool"|"tool_name"|"function": "<tool>", "arguments"|...: {...|str}}
        for key in ("name", "tool", "tool_name", "function"):
            cand = obj.get(key)
            if isinstance(cand, str) and cand in names:
                if any(k in obj and isinstance(obj[k], (dict, str)) for k in _JSON_ARG_KEYS):
                    return cand
        # OpenAI {"type": "function", "function": {"name": "<tool>", "arguments": ...}}
        fn = obj.get("function")
        if isinstance(fn, dict):
            cand = fn.get("name")
            if isinstance(cand, str) and cand in names:
                if any(k in fn and isinstance(fn[k], (dict, str)) for k in _JSON_ARG_KEYS):
                    return cand
        # Envelope {"tool_calls": [...]}
        tcs = obj.get("tool_calls")
        if isinstance(tcs, list):
            for item in tcs[:5]:
                matched = _match_json_tool_obj(item, names)
                if matched:
                    return matched
    elif isinstance(obj, list) and 1 <= len(obj) <= 10:
        for item in obj:
            matched = _match_json_tool_obj(item, names)
            if matched:
                return matched
    return ""


def _scan_json_calls(source_text: str, scan_text: str, names: set[str]) -> Detection | None:
    if not names or ("{" not in scan_text):
        return None
    decoder = json.JSONDecoder()
    idx = 0
    checks = 0
    while checks < 40:
        brace = scan_text.find("{", idx)
        if brace < 0:
            break
        idx = brace + 1
        checks += 1
        try:
            obj, end = decoder.raw_decode(scan_text, brace)
        except json.JSONDecodeError:
            continue
        tname = _match_json_tool_obj(obj, names)
        if tname:
            return Detection(
                kind="json-call",
                tool=tname,
                snippet=_compact(source_text[brace:end]),
                raw_offset=brace,
            )
    return None


def _pick_earliest(hits: list[Detection]) -> Detection | None:
    if not hits:
        return None
    hits_sorted = sorted(hits, key=lambda h: (h.raw_offset if h.raw_offset >= 0 else 10**9))
    best = hits_sorted[0]
    if not best.tool:
        for other in hits_sorted[1:]:
            if other.tool:
                return Detection(
                    kind=best.kind,
                    tool=other.tool,
                    snippet=best.snippet,
                    raw_offset=best.raw_offset,
                )
    return best


def detect_text_tool_call(
    text: str,
    tool_names: Iterable[str] = (),
    *,
    extra_markers: Iterable[str] = (),
    tool_params: Mapping[str, Iterable[str]] | None = None,
) -> Detection | None:
    """Return a :class:`Detection` if ``text`` contains a plain-text tool call, else ``None``."""
    if not text or not text.strip():
        return None

    names = {t.strip() for t in tool_names if isinstance(t, str) and t.strip()}
    params_map = tool_params or {}

    masked = _mask_non_prose(text)
    if masked.strip():
        # 1. User-configured custom markers
        for raw_marker in extra_markers:
            marker = raw_marker.strip()
            if marker:
                pos = masked.find(marker)
                if pos >= 0:
                    return Detection(
                        kind="custom-marker",
                        tool=_infer_tool_from_tail(text[pos:], names),
                        snippet=_compact(text[pos : pos + max(len(marker), 100)]),
                        raw_offset=pos,
                    )

        # 2. Special protocol markers & active-tool XML patterns
        hits = _scan_xml_and_special(text, masked, names, params_map, base_offset=0)
        json_hit = _scan_json_calls(text, masked, names)
        if json_hit is not None:
            hits.append(json_hit)
        picked = _pick_earliest(hits)
        if picked is not None:
            return picked

    # 3. Sole-fence fallback: entire reply (outside <thinking>/<summary>) is a single fenced
    #    XML / protocol tool block with zero explanatory prose around it.
    after_headers = _mask_header_tags(text)
    m_sole = _SOLE_FENCE_RE.match(after_headers)
    if m_sole:
        lang = m_sole.group("lang").strip().lower()
        body = m_sole.group("body")
        fence_offset = after_headers.find(m_sole.group("fence"))
        body_offset = m_sole.start("body")
        if lang in _DEDICATED_TOOL_FENCE_LANGS:
            return Detection(
                kind="fenced-tool-call",
                tool=_infer_tool_from_tail(body, names),
                snippet=_compact(body),
                raw_offset=fence_offset,
            )
        if lang in _XML_LIKE_FENCE_LANGS:
            inner_hits = _scan_xml_and_special(text, body, names, params_map, base_offset=body_offset)
            picked = _pick_earliest(inner_hits)
            if picked is not None:
                return Detection(
                    kind=f"fenced-{picked.kind}",
                    tool=picked.tool,
                    snippet=picked.snippet,
                    raw_offset=fence_offset,
                )

    return None


def _scrub_text_tail(text: str, raw_offset: int) -> str:
    if 0 <= raw_offset <= len(text):
        prefix = text[:raw_offset].rstrip()
    else:
        # Fallback: preserve leading <summary>...</summary> if present, drop broken tail
        m = _SUMMARY_RE.search(text)
        prefix = text[: m.end()].rstrip() if m else ""
    return f"{prefix}\n\n{_OMITTED_MARKER}" if prefix else _OMITTED_MARKER


def _sanitize_polluted_history(agent, raw_offset: int, *, sanitize_history: bool = True) -> bool:
    """Remove leaked plain-text tool call tail from the just-appended assistant history entry.

    Preserves ``thinking`` blocks (and their cryptographic signatures), user/assistant
    alternation, and any leading ``<summary>`` or prose before ``raw_offset``.
    """
    llmclient = getattr(agent, "llmclient", None)
    if llmclient is None:
        return False

    # Refresh tool schema injection on text-protocol ToolClient sessions if applicable
    if hasattr(llmclient, "last_tools"):
        try:
            llmclient.last_tools = ""
        except Exception:
            pass

    if not sanitize_history:
        return False

    backend = getattr(llmclient, "backend", None)
    if backend is None:
        return False

    targets = [backend]
    subs = getattr(backend, "_sessions", None)
    if isinstance(subs, (list, tuple)):
        for s in subs:
            if s is not None and s not in targets:
                targets.append(s)

    modified = False
    for session in targets:
        hist = getattr(session, "history", None)
        if not isinstance(hist, list) or not hist:
            continue
        lock = getattr(session, "lock", None)
        cm = lock if hasattr(lock, "__enter__") else nullcontext()
        with cm:
            if not hist:
                continue
            last = hist[-1]
            if not isinstance(last, dict) or last.get("role") != "assistant":
                continue
            content = last.get("content")
            if isinstance(content, list):
                # Scrub the last text block (never touch thinking / tool_use blocks)
                for block in reversed(content):
                    if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                        block["text"] = _scrub_text_tail(block["text"], raw_offset)
                        modified = True
                        break
            elif isinstance(content, str):
                last["content"] = _scrub_text_tail(content, raw_offset)
                modified = True

    return modified


def _build_retry_prompt(hit: Detection, attempt: int, max_retries: int) -> str:
    tool_hint = f" (`{hit.tool}`, pattern: {hit.kind})" if hit.tool else f" (pattern: {hit.kind})"
    code_run_tip = (
        "\n2. For `code_run`, either pass `script` in the structured tool call arguments, "
        "or emit a structured `code_run` tool call with `{}` arguments alongside a ```python fenced code block."
        "\n3. Only reply with normal text if the entire user task is genuinely complete and needs no tool execution."
        if hit.tool == "code_run"
        else "\n2. Only reply with normal text if the entire user task is genuinely complete and needs no tool execution."
    )
    return (
        f"[System Notice] Your previous reply wrote a tool call{tool_hint} as plain text/XML "
        f"in the message body instead of invoking the structured tool-calling API, so NO tool was executed.\n"
        f"Snippet: `{hit.snippet}`\n"
        f"Regenerate this step now:\n"
        f"1. If you still need to call `{hit.tool or 'a tool'}`, invoke it via the structured `tool_calls` / `tool_use` API "
        f"— NEVER write `<{hit.tool or 'tool_name'}>...</{hit.tool or 'tool_name'}>`, `<tool_call>`, `<｜DSML｜...>`, "
        f"or raw JSON tool objects in message text."
        f"{code_run_tip}\n"
        f"(Retry {attempt}/{max_retries})"
    )


def _emit_guard_event(ctx, turn: int, hit: Detection, attempt: int, limit: int, warn: str) -> None:
    emit = getattr(ctx, "emit", None)
    if not callable(emit) or not isinstance(turn, int) or turn <= 0:
        return
    tid = f"{turn}.guard.{attempt}"
    try:
        emit(
            "tool.call",
            {
                "id": tid,
                "turn": turn,
                "index": 0,
                "name": "toolcall_guard",
                "args": {
                    "kind": hit.kind,
                    "tool": hit.tool or None,
                    "attempt": attempt,
                    "max_retries": limit,
                },
            },
        )
        emit(
            "tool.result",
            {
                "id": tid,
                "turn": turn,
                "index": 0,
                "name": "toolcall_guard",
                "ok": False,
                "data": warn,
                "seconds": 0.0,
            },
        )
    except Exception:
        pass


def apply(ctx, config):
    agent = ctx.get("ga.agent")
    if agent is None:
        return

    handler = getattr(agent, "tools_handler", None)
    if handler is None or not hasattr(handler, "do_no_tool"):
        return

    orig_do_no_tool = handler.do_no_tool
    state = {"streak": 0, "last_turn": None, "turn": None}

    if ctx.has("ga.hooks"):
        def _on_agent_before(gactx, nxt):
            state["streak"] = 0
            state["last_turn"] = None
            state["turn"] = None
            return nxt(gactx)

        def _on_turn_before(gactx, nxt):
            if isinstance(gactx, dict) and isinstance(gactx.get("turn"), int):
                state["turn"] = gactx["turn"]
            return nxt(gactx)

        def _on_tool_before(gactx, nxt):
            if isinstance(gactx, dict) and gactx.get("tool_name") != "no_tool":
                state["streak"] = 0
            return nxt(gactx)

        ctx.on("ga.agent_before", _on_agent_before)
        ctx.on("ga.turn_before", _on_turn_before)
        ctx.on("ga.tool_before", _on_tool_before)

    def _max_retries() -> int:
        try:
            return max(0, min(10, int((config or {}).get("max_retries", 3))))
        except (TypeError, ValueError):
            return 3

    def _sanitize_enabled() -> bool:
        val = (config or {}).get("sanitize_history", True)
        return val if isinstance(val, bool) else True

    def _extra_markers() -> list[str]:
        raw = str((config or {}).get("extra_markers", "") or "")
        return [m.strip() for m in raw.split(",") if m.strip()]

    def guarded_no_tool(h, args, response):
        limit = _max_retries()
        raw = (
            response.get("content", "")
            if isinstance(response, dict)
            else getattr(response, "content", None)
        ) or ""
        cur_turn = state["turn"]
        if not isinstance(cur_turn, int):
            cur_turn = getattr(h, "current_turn", None)
        if not isinstance(cur_turn, int):
            cur_turn = getattr(h, "turn", None)

        if limit > 0 and raw:
            names, params_map = _active_tool_spec(agent, h)
            hit = detect_text_tool_call(
                raw,
                names,
                extra_markers=_extra_markers(),
                tool_params=params_map,
            )
            if hit is not None:
                # Reset streak if non-consecutive turn (intervening real tool calls succeeded or new run)
                if isinstance(cur_turn, int) and state["last_turn"] is not None:
                    if cur_turn != state["last_turn"] + 1:
                        state["streak"] = 0
                state["streak"] += 1
                if isinstance(cur_turn, int):
                    state["last_turn"] = cur_turn

                if state["streak"] <= limit:
                    StepOutcome = importlib.import_module("agent_loop").StepOutcome
                    scrubbed = _sanitize_polluted_history(
                        agent,
                        hit.raw_offset,
                        sanitize_history=_sanitize_enabled(),
                    )
                    warn = (
                        f"[toolcall-guard] Intercepted plain-text tool call "
                        f"({hit.kind}{': ' + hit.tool if hit.tool else ''}); "
                        f"{'sanitized history & ' if scrubbed else ''}"
                        f"requesting regeneration ({state['streak']}/{limit}) — {hit.snippet}"
                    )
                    _emit_guard_event(ctx, cur_turn, hit, state["streak"], limit, warn)
                    yield warn + "\n"
                    return StepOutcome(
                        {"guarded": True, "kind": hit.kind, "tool": hit.tool, "attempt": state["streak"]},
                        next_prompt=_build_retry_prompt(hit, state["streak"], limit),
                    )

                exhausted_msg = (
                    f"[toolcall-guard] Plain-text tool call persisted after {limit} retries "
                    f"({hit.kind}{': ' + hit.tool if hit.tool else ''}); allowing turn to end."
                )
                _emit_guard_event(ctx, cur_turn, hit, state["streak"], limit, exhausted_msg)
                yield exhausted_msg + "\n"

        state["streak"] = 0
        state["last_turn"] = None
        return (yield from orig_do_no_tool(args, response))

    handler.do_no_tool = lambda args, response: guarded_no_tool(handler, args, response)

    def dispose():
        handler.do_no_tool = orig_do_no_tool

    ctx.scope.add(dispose)
    return dispose
