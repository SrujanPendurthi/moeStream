"""Parity eval (docs/specs/2026-10-15-parity-eval-design.md): moeStream against lagunaLab's references.

lagunaLab's replay harness is used read-only; outputs go to results/ here only.
"""
import json
import re

_CALL = re.compile(r"<tool_call>(.*?)(</tool_call>|$)", re.S)
_ARG = re.compile(r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>", re.S)


def parse_calls(text):
    """Every tool call in an action, Laguna's format
    <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>, as (name, {key: value});
    a call cut off before </tool_call> is None."""
    out = []
    for body, end in _CALL.findall(text):
        if not end:
            out.append(None)
            continue
        name = body.split("<", 1)[0].strip()
        out.append((name, dict(_ARG.findall(body))))
    return out


def _typed(value, kind):
    try:
        if kind == "integer":
            int(value)
        elif kind == "number":
            float(value)
        elif kind == "boolean":
            return value in ("true", "false")
        elif kind in ("array", "object"):
            return isinstance(json.loads(value), list if kind == "array" else dict)
        return True
    except ValueError:
        return False
    return True


def valid_call(call, tools):
    """(ok, reason): the name is one of the session's tools, required keys are present, no unknown
    keys, and each value converts to its schema type."""
    if call is None:
        return False, "unterminated call"
    name, args = call
    spec = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools or []}
    if name not in spec:
        return False, f"unknown tool {name!r}"
    props = spec[name].get("properties", {})
    missing = [k for k in spec[name].get("required", []) if k not in args]
    if missing:
        return False, f"missing required {missing}"
    extra = [k for k in args if k not in props]
    if extra:
        return False, f"unknown key {extra}"
    for k, v in args.items():
        if not _typed(v, props[k].get("type", "string")):
            return False, f"{k} is not a {props[k].get('type')}"
    return True, ""
