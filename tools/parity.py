"""Parity eval (docs/specs/2026-10-15-parity-eval-design.md): moeStream against lagunaLab's references.

lagunaLab's replay harness is used read-only; outputs go to results/ here only.
"""
import json
import re

from tools.serve import cap_buffer_pool

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


def classify(model, prefix_ids, ref_ids, got_ids, chunk=2048):
    """"exact", "tie" (the first differing token was a near-tie under this model's own logits) or "real".
    The prefix is prefilled in chunks through the model's cache, as the references were built."""
    import mlx.core as mx

    i = first_diff(ref_ids, got_ids)
    if i is None:
        return "exact"
    if i >= len(ref_ids) or i >= len(got_ids):
        return "real"  # one stopped where the other went on: not a single-token flip
    ids = [int(t) for t in prefix_ids] + [int(t) for t in ref_ids[:i]]  # .npz ids are numpy int32
    cache = model.make_cache()
    for k in range(0, len(ids), chunk):
        logits = model(mx.array([ids[k:k + chunk]]), cache=cache)[0, -1]
        mx.eval(logits)
    return "tie" if is_tie(logits.astype(mx.float32), int(ref_ids[i]), int(got_ids[i])) else "real"


def summarize(sessions):
    """Assistant-token-weighted KL and top-1 over the sessions that scored."""
    ok = [r for r in sessions.values() if "error" not in r]
    n = sum(r["assistant_tokens"] for r in ok)
    if not n:
        return {"sessions": 0, "assistant_tokens": 0}
    return {"sessions": len(ok), "failed": len(sessions) - len(ok), "assistant_tokens": n,
            "kl_top64_mean": sum(r["kl_top64_mean"] * r["assistant_tokens"] for r in ok) / n,
            "kl_top64_p99_max": max(r["kl_top64_p99"] for r in ok),
            "top1_agree": sum(r["top1_agree"] * r["assistant_tokens"] for r in ok) / n,
            "peak_mem_gb_max": max(r["peak_mem_gb"] for r in ok)}


def gate(summary, kl, top1):
    return summary.get("kl_top64_mean", 1e9) <= kl and summary.get("top1_agree", 0) >= top1


def score_sessions(model, refs_dir, sessions, out, chunk=2048, make_cache=None, forward=None, meta=None):
    """lagunaLab's score() per session; out (JSON) rewritten after each, so an abort keeps what ran.
    A session that fails (e.g. ids not matching the reference) is recorded and the run goes on."""
    import traceback
    from pathlib import Path

    from replay.replay import score

    res = {**(meta or {}), "sessions": {}, "summary": {}}
    for s in sessions:
        try:
            r = score(model, str(Path(refs_dir) / f"{s}.npz"), chunk=chunk, make_cache=make_cache, forward=forward)
        except Exception as e:  # recorded, not fatal
            r = {"error": repr(e), "trace": traceback.format_exc(limit=3)}
        res["sessions"][s] = r
        res["summary"] = summarize(res["sessions"])
        print(s, json.dumps(r)[:300], flush=True)
        Path(out).write_text(json.dumps(res, indent=1))
    return res


def load_engine(pack, kv_bits, memory_gb, context_tokens, calibration="budget.json", mlx_cache_gb=1.0):
    """moeStream as shipped (lookahead off, pool capped), through the config path. The calibration
    is skipped when its grid was made for another expert count (pruned-50 has 128, budget.json 256)."""
    from pathlib import Path

    import moestream.load
    from moestream.config import EngineConfig

    experts = json.loads((Path(pack) / "layout.json").read_text())["num_experts"]
    cal = str(calibration) if calibration and Path(calibration).exists() else None
    if cal and max(json.loads(Path(cal).read_text())["grid"]) != experts:
        cal = None
    cap_buffer_pool(mlx_cache_gb)
    return moestream.load.load(EngineConfig(pack_dir=str(pack), memory_gb=memory_gb, kv_bits=kv_bits,
                                            context_tokens=context_tokens, calibration=cal, lookahead=False,
                                            mlx_cache_gb=mlx_cache_gb))


def _tool(a):
    from scripts.diag_actions import tool  # lagunaLab's same-tool rule, read-only

    return tool(a)


def action_rows(model, tok, refs_dir, sessions, tools, greedy=None, classifier=None, refs=None):
    """Greedy actions at each reference start, against the reference's strings: exact, same tool,
    common prefix, validity of every generated tool call, and for mismatches "tie" or "real".
    tools: {session: the session's tool schemas}. refs: {session: (ids, starts, actions, max_new)}
    instead of the .npz files (A0's self-references)."""
    import os
    from pathlib import Path

    import numpy as np

    if greedy is None:
        from replay.replay import greedy_actions as greedy
    classifier = classifier or classify
    rows = []
    for s in sessions:
        if refs and s in refs:
            ids, starts, want, max_new = refs[s]
        else:
            r = np.load(Path(refs_dir) / f"{s}.npz")
            ids, starts, want, max_new = r["ids"], r["action_starts"], json.loads(str(r["actions"])), int(r["max_new"])
        got = greedy(model, tok, ids, starts, max_new)
        for st, w, g in zip([int(x) for x in starts], want, got):
            calls = parse_calls(g)
            checks = [valid_call(c, tools[s]) for c in calls]
            kind = "exact" if w == g else classifier(model, list(ids[:st]), tok.encode(w, add_special_tokens=False),
                                                     tok.encode(g, add_special_tokens=False))
            wc, gc = parse_calls(w), calls
            same_args = bool(wc and gc and wc[0] and gc[0] and set(wc[0][1]) == set(gc[0][1]))
            rows.append({"session": s, "start": st, "exact": w == g, "same_tool": _tool(w) == _tool(g),
                         "same_arg_keys": same_args, "common_prefix": len(os.path.commonprefix([w, g])),
                         "calls_total": len(calls), "calls_valid": sum(ok for ok, _ in checks),
                         "invalid": [why for ok, why in checks if not ok], "kind": kind, "ref": w, "got": g})
            print(s, st, rows[-1]["exact"], rows[-1]["same_tool"], kind, flush=True)
    summary = {"n": len(rows), "exact": sum(r["exact"] for r in rows), "same_tool": sum(r["same_tool"] for r in rows),
               "calls_total": sum(r["calls_total"] for r in rows), "calls_valid": sum(r["calls_valid"] for r in rows),
               "real_mismatches": sum(r["kind"] == "real" and not (r["same_tool"] and r["same_arg_keys"]) for r in rows),
               "ties": sum(r["kind"] == "tie" for r in rows)}
    return rows, summary


def action_gate(summary, n=20, exact=14, same_tool=19):
    """A1's gate, scaled to n actions: same tool >= 19/20, exact >= 14/20, no real mismatch in a tool
    name or argument keys, every generated tool call valid."""
    return (summary["same_tool"] >= same_tool * n / 20 and summary["exact"] >= exact * n / 20
            and summary["real_mismatches"] == 0 and summary["calls_valid"] == summary["calls_total"])


def build_self_refs(model, tok, refs_dir, sessions, out_dir, greedy=None):
    """A0: the engine's own greedy actions at the reference starts, one JSON per session."""
    from pathlib import Path

    import numpy as np

    if greedy is None:
        from replay.replay import greedy_actions as greedy
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for s in sessions:
        r = np.load(Path(refs_dir) / f"{s}.npz")
        acts = greedy(model, tok, r["ids"], r["action_starts"], int(r["max_new"]))
        (Path(out_dir) / f"{s}.json").write_text(json.dumps({"actions": acts}))


def load_self_refs(refs_dir, sessions, self_dir):
    """{session: (ids, starts, actions, max_new)}: the reference's prompts with the engine's own actions."""
    from pathlib import Path

    import numpy as np

    out = {}
    for s in sessions:
        r = np.load(Path(refs_dir) / f"{s}.npz")
        acts = json.loads((Path(self_dir) / f"{s}.json").read_text())["actions"]
        out[s] = (r["ids"], r["action_starts"], acts, int(r["max_new"]))
    return out


LAB = "../lagunaLab"
DJANGO = ["django__django-10914", "django__django-10924", "django__django-11001", "django__django-11019",
          "django__django-11039"]
_PRUNED, _FULL = f"{LAB}/models/Laguna-XS-2.1-pruned-50-packed", f"{LAB}/models/Laguna-XS-2.1-packed"
SETTINGS = {  # the spec's comparison table
    "E1": dict(pack=_PRUNED, kv_bits=8, refs=f"{LAB}/refs/kv", context_tokens=90112, gate=(0.02, 0.98)),
    "E2": dict(pack=_PRUNED, kv_bits=0, refs=f"{LAB}/refs/kv", context_tokens=90112, gate=(0.005, 0.995)),
    "M1": dict(pack=_FULL, kv_bits=8, refs=f"{LAB}/refs/prune", context_tokens=32768, gate=(0.02, 0.98)),
    "A0": dict(pack=_FULL, kv_bits=8, refs=f"{LAB}/refs/prune_actions", context_tokens=32768),
    "A1": dict(pack=_FULL, kv_bits=8, refs=f"{LAB}/refs/prune_actions", context_tokens=32768),
}


def parse_args(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="parity eval (docs/specs/2026-10-15-parity-eval-design.md)")
    p.add_argument("--id", required=True, choices=sorted(SETTINGS))
    p.add_argument("--sessions", nargs="+", default=DJANGO)
    p.add_argument("--out", required=True, help="a path in this repo, never lagunaLab's")
    p.add_argument("--memory-gb", type=float, default=12.0)
    return p.parse_args(argv)


def main(argv=None):
    from pathlib import Path

    from replay.session import load_session

    a = parse_args(argv)
    s = SETTINGS[a.id]
    model, tok = load_engine(s["pack"], s["kv_bits"], a.memory_gb, s["context_tokens"])
    meta = {"id": a.id, "pack": s["pack"], "kv_bits": s["kv_bits"], "memory_gb": a.memory_gb, "refs": s["refs"]}
    if "gate" in s:
        res = score_sessions(model, s["refs"], a.sessions, a.out, meta=meta)
        res["gate"] = {"kl": s["gate"][0], "top1": s["gate"][1], "pass": gate(res["summary"], *s["gate"])}
    else:
        tools = {x: load_session(f"{LAB}/traces/{x}.jsonl")[1] for x in a.sessions}
        refs = None
        if a.id == "A0":
            self_dir = Path(a.out).with_suffix("") / "self_refs"
            build_self_refs(model, tok, s["refs"], a.sessions, self_dir)
            refs = load_self_refs(s["refs"], a.sessions, self_dir)
        rows, summary = action_rows(model, tok, s["refs"], a.sessions, tools, refs=refs)
        res = {**meta, "rows": rows, "summary": summary}
        if a.id == "A1":
            res["gate"] = {"pass": action_gate(summary, n=summary["n"])}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("summary:", json.dumps(res.get("summary")), "gate:", json.dumps(res.get("gate")), flush=True)


if __name__ == "__main__":
    main()
