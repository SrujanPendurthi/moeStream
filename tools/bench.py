"""Decode speed, hit rate, prediction accuracy and time split: lagunaLab streaming vs moeStream.

PYTHONPATH=.:../lagunaLab python3 -m tools.bench [--cold] [--budget-gb X] [--slots N] [--eval-writes] [--no-base] [--no-prefetch]
Loads real weights one model at a time: message the other sessions and wait for a go-ahead first.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import mlx.core as mx

def decode_tok_s(model, prompts, max_tokens, eos, stats=None):
    """Greedy decode tok/s with each prompt's prefill excluded; numeric stats(model) deltas over decode only.
    Tokens equal generate_greedy's."""
    n, secs, outs, delta = 0, 0.0, [], {}
    for pr in prompts:
        cache = model.make_cache()
        t = int(mx.argmax(model(mx.array([pr]), cache=cache)[0, -1]))
        out = [t]
        s0 = stats(model) if stats else {}
        t0 = time.perf_counter()
        while t not in eos and len(out) < max_tokens:
            t = int(mx.argmax(model(mx.array([[t]]), cache=cache)[0, -1]))
            out.append(t)
        secs += time.perf_counter() - t0
        n += len(out) - 1
        for k, v in (stats(model) if stats else {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                delta[k] = delta.get(k, 0) + v - s0[k]
        outs.append(out)
    return n / max(secs, 1e-9), outs, delta


def long_ctx(model, ids, gen=64, chunk=2048):
    cache = model.make_cache()
    t0 = time.perf_counter()
    for a in range(0, len(ids), chunk):
        logits = model(mx.array([ids[a : a + chunk]]), cache=cache)[0, -1]
        mx.eval(logits)
    prefill = time.perf_counter() - t0
    t0 = time.perf_counter()
    t = int(mx.argmax(logits))
    for _ in range(gen):
        logits = model(mx.array([[t]]), cache=cache)[0, -1]
        t = int(mx.argmax(logits))
    return {"ctx_tokens": len(ids), "prefill_ctx_tok_s": len(ids) / prefill,
            "decode_ctx_tok_s": gen / (time.perf_counter() - t0)}


def evict_page_cache(paths):
    """Cold-cache approximation without sudo: stream unrelated big files through the page cache."""
    for f in paths:
        with open(f, "rb") as fh:
            while fh.read(64 << 20):
                pass


def main():
    from laguna_mlx.streaming import load_streaming
    from laguna_mlx.streaming import stream_stats as lab_stats
    from moestream.layer import stream_stats
    from moestream.load import load
    from scripts.check_parity import PROMPTS

    p = argparse.ArgumentParser()
    p.add_argument("--pack", default="../lagunaLab/models/Laguna-XS-2.1-packed")
    p.add_argument("--budget-gb", type=float, default=None)
    p.add_argument("--calibration", default="budget.json")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--ctx", type=int, default=16000)
    p.add_argument("--cold", action="store_true")
    p.add_argument("--slots", type=int, default=None, help="diagnostic: same slot count on every layer")
    p.add_argument("--eval-writes", action="store_true", help="diagnostic: time slot writes in write_s")
    p.add_argument("--no-base", action="store_true", help="diagnostic: skip the lagunaLab baseline run")
    p.add_argument("--no-prefetch", action="store_true", help="diagnostic: no lookahead prediction or prefetch")
    a = p.parse_args()
    import moestream.layer
    moestream.layer.EVAL_WRITES = a.eval_writes
    ctx_text = Path("../lagunaLab/tests/reference/modeling_laguna.py").read_text()
    cold_files = sorted(glob.glob("../lagunaLab/models/Laguna-XS-2.1-mlx-4bit/*.safetensors"))

    def run(label, loader, stats):
        if a.cold:
            evict_page_cache(cold_files)
        mx.reset_peak_memory()
        model, tok = loader()
        prompts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                           enable_thinking=False) for q in PROMPTS[:3]]
        tps, outs, d = decode_tok_s(model, prompts, a.max_tokens, tuple(tok.eos_token_ids), stats)
        s1 = stats(model)
        n = sum(len(o) - 1 for o in outs)
        row = {"label": label, "decode_tok_s": tps, "peak_gpu_gb": mx.get_peak_memory() / 1e9,
               "hit_rate_decode": d["hits"] / max(1, d["hits"] + d["misses"])}
        for k in ("sync_s", "read_s", "write_s"):
            if k in d:
                row[f"{k[:-2]}_ms_per_tok"] = 1000 * d[k] / max(1, n)
        if "pred_acc" in s1:
            row["pred_acc"] = s1["pred_acc"]
            row["slots_per_layer"] = [min(s1["slots"].values()), max(s1["slots"].values())]
        ids = tok.encode(ctx_text)
        ids = (ids * (a.ctx // len(ids) + 1))[: a.ctx]
        row.update(long_ctx(model, ids))
        row["peak_gpu_gb"] = mx.get_peak_memory() / 1e9
        print(row, flush=True)
        del model
        mx.clear_cache()
        return row, outs

    base = None
    if not a.no_base:
        base, base_outs = run("lagunaLab slots=32", lambda: load_streaming(a.pack, 32), lab_stats)
    def load_mine():
        model, tok = load(a.pack, slots=a.slots, budget_gb=a.budget_gb, calibration=a.calibration)
        if a.no_prefetch:
            for l in model.layers:
                if hasattr(l.mlp, "io"):
                    l.mlp.io.next = None
        return model, tok

    mine, my_outs = run("moeStream", load_mine, stream_stats)
    if base:
        mine["identical_to_lagunaLab"] = my_outs == base_outs
        mine["speedup_vs_lagunaLab"] = mine["decode_tok_s"] / base["decode_tok_s"]
    mine["ctx_decode_ratio"] = mine["decode_ctx_tok_s"] / mine["decode_tok_s"]
    tag = ("_cold" if a.cold else "") + (f"_slots{a.slots}" if a.slots else "") + ("_evalwrites" if a.eval_writes else "") + ("_nobase" if a.no_base else "") + ("_noprefetch" if a.no_prefetch else "")
    out = Path("results") / f"bench{tag}.json"
    out.write_text(json.dumps({"rows": [r for r in (base, mine) if r], "cold": a.cold}, indent=1))


if __name__ == "__main__":
    main()
