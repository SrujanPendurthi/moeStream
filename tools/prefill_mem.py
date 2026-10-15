"""Prefill transient memory against prefill step size and prompt length, on the serving load path.

usage: PYTHONPATH=.:../lagunaLab python3 -m tools.prefill_mem --config results/prefill.toml
           [--steps 2048,1024,512] [--ctx 5000,15000,30000] [--eval-groups] [--out results/prefill_mem.json]

Loads exactly as tools.serve does (EngineConfig, mlx_cache_gb pool cap, q8 KV caches), runs one
unrecorded 2k-token warm-up (it fills every layer's slots; the output records whether active then
equals slots + resident), then for each step size, largest first whatever the order given (the
smallest step reads the most expert bytes per token, so it pressures a shared machine most), and
each prompt length, shortest first, prefills a fresh cache through mlx-lm's generate_step as the
server does. Per point: MLX's peak over the active memory before it, MLX's pool after it, top's MEM
for the process, expert bytes read, pageins, and prefill tok/s. Rows are written after each point,
so a watchdog abort keeps what finished. The fit per step is peak - active = a + b * tokens, the
activation term the budget reserves as a flat 1 GB today. --eval-groups sets layer.EVAL_GROUPS,
--prefill-set layer.PREFILL_SET
(docs/prefill-memory.md).
"""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import re

import mlx.core as mx

GB = 1e9


def parse_pageins(text):
    return int(re.search(r"Pageins:\s+(\d+)\.", text).group(1))


def pageins():
    return parse_pageins(subprocess.run(["vm_stat"], capture_output=True, text=True).stdout)


def top_mb(pid):
    """top's MEM for pid in MB (0 if top fails)."""
    try:
        v = subprocess.run(["top", "-l", "1", "-pid", str(pid), "-stats", "mem"], capture_output=True,
                           text=True).stdout.split()[-1].rstrip("+-")
        return int(float(v[:-1]) * {"G": 1024, "M": 1, "K": 1 / 1024}[v[-1]])
    except Exception:
        return 0


def measure(model, ids, step, misses=None, pageins=None, stride=0):
    """One prefill of ids into a fresh cache: MLX active before, peak during, tok/s, top's max MEM;
    with misses/pageins (counter callables), expert bytes read and pageins during it."""
    from mlx_lm.generate import generate_step

    top, done = [0], threading.Event()

    def sample():
        while not done.wait(1.0):
            top[0] = max(top[0], top_mb(os.getpid()))

    cache = model.make_cache()
    mx.clear_cache()
    mx.reset_peak_memory()
    active = mx.get_active_memory()
    m0, p0 = misses() if misses else 0, pageins() if pageins else 0
    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t0 = time.perf_counter()
    next(generate_step(mx.array(ids), model, max_tokens=1, prompt_cache=cache, prefill_step_size=step))
    secs = time.perf_counter() - t0
    done.set()
    th.join()
    row = {"step": step, "tokens": len(ids), "active_gb": active / GB, "peak_gb": mx.get_peak_memory() / GB,
           "transient_gb": (mx.get_peak_memory() - active) / GB, "prefill_tok_s": len(ids) / secs,
           "top_mb": top[0], "cache_gb": mx.get_cache_memory() / GB}
    if misses:
        row["bytes_read"] = (misses() - m0) * stride
    if pageins:
        row["pageins"] = pageins() - p0
    del cache
    mx.clear_cache()
    return row


def fit(rows):
    """Per step: least-squares transient_gb = a + b * tokens, b reported per 1k tokens."""
    out = {}
    for step in sorted({r["step"] for r in rows}):
        pts = [(r["tokens"] / 1000, r["transient_gb"]) for r in rows if r["step"] == step]
        if len(pts) < 2:
            continue
        n = len(pts)
        mx_, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
        b = sum((x - mx_) * (y - my) for x, y in pts) / sum((x - mx_) ** 2 for x, _ in pts)
        out[step] = {"gb": my - b * mx_, "gb_per_1k_tokens": b}
    return out


def sweep(model, base, steps, ctxs, out, warmup=2048, meta=None, misses=None, pageins=None, stride=0):
    """One unrecorded warm-up prefill (the slots fill during the first one, so a cold baseline would
    count them as transient), then every step, largest first, x length, saved after each point.
    meta's slots_gb + resident_gb, when given, are checked against the active memory after warm-up."""
    steps = sorted(steps, reverse=True)
    meta = dict(meta or {})
    measure(model, (base * (warmup // len(base) + 1))[:warmup], steps[0])
    meta["active_after_warmup_gb"] = mx.get_active_memory() / GB
    if "slots_gb" in meta:
        meta["warmup_filled"] = abs(meta["active_after_warmup_gb"] - meta["slots_gb"] - meta["resident_gb"]) < 0.2
        print(f"after warm-up: active {meta['active_after_warmup_gb']:.2f} GB, slots + resident "
              f"{meta['slots_gb'] + meta['resident_gb']:.2f} GB", flush=True)
    rows = []
    for step in steps:
        for n in ctxs:
            ids = (base * (n // len(base) + 1))[:n]
            rows.append(measure(model, ids, step, misses, pageins, stride))
            print(rows[-1], flush=True)
            Path(out).write_text(json.dumps({**meta, "rows": rows, "fit": fit(rows)}, indent=1))
    return rows


def main():
    import moestream.layer
    from moestream.config import EngineConfig
    from moestream.layer import stream_stats
    from moestream.load import load
    from tools.serve import cap_buffer_pool

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--steps", default="2048,1024,512")
    p.add_argument("--ctx", default="5000,15000,30000")
    p.add_argument("--out", default="results/prefill_mem.json")
    p.add_argument("--eval-groups", action="store_true", help="diagnostic: layer.EVAL_GROUPS")
    p.add_argument("--prefill-set", action="store_true", help="layer.PREFILL_SET: one group per layer in prefill")
    a = p.parse_args()
    moestream.layer.EVAL_GROUPS, moestream.layer.PREFILL_SET = a.eval_groups, a.prefill_set
    cfg = EngineConfig.read(a.config)
    cap_buffer_pool(cfg.mlx_cache_gb)
    model, tok = load(cfg)
    text = Path("../lagunaLab/tests/reference/modeling_laguna.py").read_text()  # tools.bench's long prompt
    stride = json.loads((Path(cfg.pack_dir) / "layout.json").read_text())["stride"]
    meta = {"memory_gb": cfg.memory_gb, "eval_groups": a.eval_groups, "prefill_set": a.prefill_set,
            "slots_gb": sum(stream_stats(model)["slots"].values()) * stride / GB,
            "resident_gb": (Path(cfg.pack_dir) / "resident.safetensors").stat().st_size / GB}
    rows = sweep(model, tok.encode(text), [int(s) for s in a.steps.split(",")],
                 [int(n) for n in a.ctx.split(",")], a.out, meta=meta,
                 misses=lambda: stream_stats(model)["misses"], pageins=pageins, stride=stride)
    print("fit:", fit(rows), flush=True)


if __name__ == "__main__":
    main()
