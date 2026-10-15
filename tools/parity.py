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


def _ulp(x):
    import math

    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x else 2.0 ** -133


def is_tie(logits, a, b, ulps=4):
    """Tokens a and b are the top two and their logits are within `ulps` bf16 ulps: an exact engine
    can flip between them on summation order alone (lagunaLab runbook: 0-4 ulps measured)."""
    import mlx.core as mx

    top2 = set(mx.argsort(-logits)[:2].tolist())
    if {a, b} != top2:
        return False
    la, lb = float(logits[a]), float(logits[b])
    return abs(la - lb) <= ulps * _ulp(max(abs(la), abs(lb)))


def first_diff(ref_ids, got_ids):
    """Index of the first differing token, or None if one is the other (equal)."""
    for i, (r, g) in enumerate(zip(ref_ids, got_ids)):
        if r != g:
            return i
    return None if len(ref_ids) == len(got_ids) else min(len(ref_ids), len(got_ids))


def classify(model, prefix_ids, ref_ids, got_ids):
    """"exact", "tie" (the first differing token was a near-tie under this model's own logits) or "real"."""
    import mlx.core as mx

    i = first_diff(ref_ids, got_ids)
    if i is None:
        return "exact"
    if i >= len(ref_ids) or i >= len(got_ids):
        return "real"  # one stopped where the other went on: not a single-token flip
    logits = model(mx.array([list(prefix_ids) + list(ref_ids[:i])]))[0, -1].astype(mx.float32)
    return "tie" if is_tie(logits, ref_ids[i], got_ids[i]) else "real"
