"""Decode speed, hit rate, prediction accuracy and time split: lagunaLab streaming vs moeStream.

PYTHONPATH=.:../lagunaLab python3 -m tools.bench [--cold] [--budget-gb X]
Loads real weights one model at a time: message the other sessions and wait for a go-ahead first.
"""
import argparse
import glob
import json
import time
from pathlib import Path

import mlx.core as mx

from laguna_mlx.dflash import generate_greedy


def decode_tok_s(model, prompts, max_tokens, eos):
    outs, t0 = [], time.perf_counter()
    for pr in prompts:
        outs.append(generate_greedy(model, pr, max_tokens, eos))
    return sum(map(len, outs)) / (time.perf_counter() - t0), outs


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
    a = p.parse_args()
    ctx_text = Path("../lagunaLab/tests/reference/modeling_laguna.py").read_text()
    cold_files = sorted(glob.glob("../lagunaLab/models/Laguna-XS-2.1-mlx-4bit/*.safetensors"))

    def run(label, loader, stats):
        if a.cold:
            evict_page_cache(cold_files)
        mx.reset_peak_memory()
        model, tok = loader()
        prompts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                           enable_thinking=False) for q in PROMPTS[:3]]
        s0 = stats(model)
        tps, outs = decode_tok_s(model, prompts, a.max_tokens, tuple(tok.eos_token_ids))
        s1 = stats(model)
        n = sum(map(len, outs))
        row = {"label": label, "decode_tok_s": tps, "peak_gpu_gb": mx.get_peak_memory() / 1e9,
               "hit_rate_decode": (s1["hits"] - s0["hits"]) / max(1, s1["hits"] + s1["misses"] - s0["hits"] - s0["misses"])}
        for k in ("sync_s", "read_s", "write_s"):
            if k in s1:
                row[f"{k[:-2]}_ms_per_tok"] = 1000 * (s1[k] - s0[k]) / n
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

    base, base_outs = run("lagunaLab slots=32", lambda: load_streaming(a.pack, 32), lab_stats)
    mine, my_outs = run("moeStream", lambda: load(a.pack, budget_gb=a.budget_gb, calibration=a.calibration),
                        stream_stats)
    mine["identical_to_lagunaLab"] = my_outs == base_outs
    mine["speedup_vs_lagunaLab"] = mine["decode_tok_s"] / base["decode_tok_s"]
    mine["ctx_decode_ratio"] = mine["decode_ctx_tok_s"] / mine["decode_tok_s"]
    out = Path("results") / ("bench_cold.json" if a.cold else "bench.json")
    out.write_text(json.dumps({"rows": [base, mine], "cold": a.cold}, indent=1))


if __name__ == "__main__":
    main()
