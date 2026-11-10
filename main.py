"""toolcall-guard: catch tool calls that the model wrote as plain text.

Some models (seen with deepseek-v4.1-flash) occasionally emit a tool call as text in
the reply body instead of a structured tool call, e.g.

    <code_run><parameter name="script">...</parameter></code_run>
    <code_run> <｜DSML｜ parameter name="script" string="true">...
    <code_run script="...">...</code_run>

GA sees no tool call, routes the turn to ``do_no_tool`` and treats it as the final
reply, so the run ends as ``done`` although nothing was executed.

This plugin wraps ``GenericAgentHandler.do_no_tool`` (GA source is not modified; the
original is restored when the plugin is disabled). When the reply looks like a
text-form tool call it returns an error prompt so the model regenerates. The text is
never parsed into a real call and never executed.

config:
  max_retries   consecutive turns to ask for regeneration (default 2, 0 = off)
"""
from __future__ import annotations
import importlib
import re
import sys

_MISSING = object()

# fenced blocks and inline code are how a reply legitimately *shows* such markup
_CODE = re.compile(r"```[\s\S]*?```|`[^`\n]*`")
_DSML = re.compile(r"<\s*[|｜]\s*DSML\s*[|｜]", re.I)

PROMPT = (
    "[ERROR] Tool call format invalid; no tool was called. "
    "You wrote the tool call as plain text in the reply body, so nothing was executed. "
    "Regenerate using a proper structured tool call. "
    "If you only meant to show this markup to the user, put it inside a fenced code block."
)


def detect(content: str, names) -> str | None:
    """Return the kind of text-form tool call found in ``content``, or None."""
    if not content:
        return None
    text = _CODE.sub(" ", content)
    if _DSML.search(text):
        return "dsml"
    names = [n for n in names if n]
    if not names:
        return None
    alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    if re.search(r"<\s*invoke\s+name\s*=\s*[\"']?(?:%s)\b" % alt, text):
        return "invoke"
    for m in re.finditer(r"<\s*(%s)(\s+[\w:.-]+\s*=|\s*>)" % alt, text):
        if "=" in m.group(2):
            return "attr"
        rest = text[m.end():]
        if re.search(r"<\s*parameter\b", rest) or re.search(r"<\s*/\s*%s\s*>" % re.escape(m.group(1)), rest):
            return "tag"
    return None


def tool_names(mod, cls) -> list[str]:
    out = []
    for t in getattr(mod, "TOOLS_SCHEMA", None) or []:
        if isinstance(t, dict):
            out.append((t.get("function") or {}).get("name", ""))
    out += [a[3:] for a in dir(cls) if a.startswith("do_") and a != "do_no_tool"]
    return list(dict.fromkeys(n for n in out if n))


def _call(fn, *a):
    ret = fn(*a)
    if hasattr(ret, "__iter__") and not isinstance(ret, (str, bytes, dict, list)):
        ret = yield from ret
    return ret


def make_wrapper(orig, mod, cls, max_retries: int):
    def do_no_tool(self, args, response):
        kind = None
        if max_retries > 0:
            try:
                kind = detect(getattr(response, "content", "") or "", tool_names(mod, cls))
            except Exception as e:  # never break the run because of the guard
                sys.stderr.write(f"[toolcall-guard] detect error: {e}\n")
        if kind:
            turn = getattr(self, "current_turn", 0) or 0
            last_turn, n = getattr(self, "_tcg_state", (-9, 0))
            n = n + 1 if turn == last_turn + 1 else 1
            self._tcg_state = (turn, n)
            if n <= max_retries:
                StepOutcome = importlib.import_module("agent_loop").StepOutcome
                sys.stderr.write(f"[toolcall-guard] text-form tool call ({kind}) at turn {turn}, retry {n}/{max_retries}\n")
                yield f"[Warn] Tool call was written as plain text ({kind}); nothing was executed. Asking the model to regenerate ({n}/{max_retries})...\n"
                return StepOutcome({}, next_prompt=PROMPT)
        return (yield from _call(orig, self, args, response))
    do_no_tool.__wrapped__ = orig
    return do_no_tool


def apply(ctx, config):
    agent = ctx.get("ga.agent")
    if agent is None:
        return
    mod = sys.modules[type(agent).__module__]
    cls = getattr(mod, "GenericAgentHandler", None)
    if cls is None or not hasattr(cls, "do_no_tool"):
        sys.stderr.write("[toolcall-guard] GenericAgentHandler.do_no_tool not found; guard inactive\n")
        return
    try:
        max_retries = int(config.get("max_retries", 2))
    except (TypeError, ValueError):
        max_retries = 2
    saved = cls.__dict__.get("do_no_tool", _MISSING)
    wrapper = make_wrapper(cls.do_no_tool, mod, cls, max_retries)
    cls.do_no_tool = wrapper

    def dispose():
        if cls.__dict__.get("do_no_tool") is not wrapper:
            return  # someone else replaced it afterwards; leave theirs alone
        if saved is _MISSING:
            delattr(cls, "do_no_tool")
        else:
            setattr(cls, "do_no_tool", saved)
    ctx.scope.add(dispose)
