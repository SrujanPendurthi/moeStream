"""Prefill transient memory against prefill step size and prompt length, on the serving load path.

usage: PYTHONPATH=.:../lagunaLab python3 -m tools.prefill_mem --config results/prefill.toml
           [--steps 512,1024,2048] [--ctx 5000,15000,30000] [--out results/prefill_mem.json]

Loads exactly as tools.serve does (EngineConfig, mlx_cache_gb pool cap, q8 KV caches), then for each
step size (smallest first) and prompt length (shortest first) prefills a fresh cache through mlx-lm's
generate_step, as the server does, and records MLX's peak over the active memory before it, top's
MEM for the process, and prefill tok/s. Rows are written after each point, so a watchdog abort keeps
what finished. The fit per step is peak - active = a + b * tokens, the activation term the budget
reserves as a flat 1 GB today.
"""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import mlx.core as mx

GB = 1e9


def top_mb(pid):
    """top's MEM for pid in MB (0 if top fails)."""
    try:
        v = subprocess.run(["top", "-l", "1", "-pid", str(pid), "-stats", "mem"], capture_output=True,
                           text=True).stdout.split()[-1].rstrip("+-")
        return int(float(v[:-1]) * {"G": 1024, "M": 1, "K": 1 / 1024}[v[-1]])
    except Exception:
        return 0


def measure(model, ids, step):
    """One prefill of ids into a fresh cache: MLX active before, peak during, tok/s, top's max MEM."""
    from mlx_lm.generate import generate_step

    top, done = [0], threading.Event()

    def sample():
        while not done.wait(1.0):
            top[0] = max(top[0], top_mb(os.getpid()))

    cache = model.make_cache()
    mx.clear_cache()
    mx.reset_peak_memory()
    active = mx.get_active_memory()
    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t0 = time.perf_counter()
    next(generate_step(mx.array(ids), model, max_tokens=1, prompt_cache=cache, prefill_step_size=step))
    secs = time.perf_counter() - t0
    done.set()
    th.join()
    row = {"step": step, "tokens": len(ids), "active_gb": active / GB, "peak_gb": mx.get_peak_memory() / GB,
           "transient_gb": (mx.get_peak_memory() - active) / GB, "prefill_tok_s": len(ids) / secs,
           "top_mb": top[0]}
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


def main():
    from moestream.config import EngineConfig
    from moestream.load import load
    from tools.serve import cap_buffer_pool

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--steps", default="512,1024,2048")
    p.add_argument("--ctx", default="5000,15000,30000")
    p.add_argument("--out", default="results/prefill_mem.json")
    a = p.parse_args()
    cfg = EngineConfig.read(a.config)
    cap_buffer_pool(cfg.mlx_cache_gb)
    model, tok = load(cfg)
    text = Path("../lagunaLab/tests/reference/modeling_laguna.py").read_text()  # tools.bench's long prompt
    base = tok.encode(text)
    rows = []
    for step in map(int, a.steps.split(",")):
        for n in map(int, a.ctx.split(",")):
            ids = (base * (n // len(base) + 1))[:n]
            rows.append(measure(model, ids, step))
            print(rows[-1], flush=True)
            Path(a.out).write_text(json.dumps({"memory_gb": cfg.memory_gb, "rows": rows, "fit": fit(rows)}, indent=1))
    print("fit:", fit(rows), flush=True)


if __name__ == "__main__":
    main()
