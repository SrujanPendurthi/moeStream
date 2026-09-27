"""Offline: replay recorded routing through SlotStore, producing per-layer miss curves and a decay half-life.

PYTHONPATH=.:../lagunaLab python3 -m tools.calibrate [--routing data/routing] [--out budget.json]
No model load. Layers run in parallel processes.
"""
import argparse
import glob
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from moestream.store import SlotStore

GRID = [16, 32, 48, 64, 96, 128, 192, 256]
HALF_LIVES = [64.0, 128.0, 256.0, 512.0, 1024.0, float("inf")]


def miss_rate(seq, slots, half_life, num_experts=256):
    s = SlotStore(num_experts, slots, half_life)
    for row in seq:
        s.tick()
        s.assign(row.tolist())
    return s.misses / max(1, s.misses + s.hits)


def _layer_rate(seqs, slots, half_life, num_experts):
    # sessions are independent agent runs: fresh cache each, rate over all tokens
    rates = [miss_rate(q, slots, half_life, num_experts) for q in seqs]
    n = [len(q) for q in seqs]
    return float(np.average(rates, weights=n))


def calibrate(seqs, grid=GRID, half_lives=HALF_LIVES, ref_slots=64, num_experts=256):
    layers = sorted(seqs)
    with ProcessPoolExecutor() as ex:
        search = {}
        for h in half_lives:
            f = partial(_layer_rate, slots=ref_slots, half_life=h, num_experts=num_experts)
            search[str(h)] = float(np.mean(list(ex.map(f, [seqs[l] for l in layers]))))
        best = min(half_lives, key=lambda h: search[str(h)])
        curves = {}
        for l in layers:
            f = partial(_layer_rate, seqs[l], half_life=best, num_experts=num_experts)
            curves[str(l)] = list(ex.map(f, grid))
    return {"half_life": best, "grid": list(grid), "curves": curves, "half_life_search": search,
            "tokens": int(sum(len(q) for q in seqs[layers[0]]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--routing", default="data/routing")
    p.add_argument("--out", default="budget.json")
    p.add_argument("--max-tokens", type=int, default=2048, help="per session; bounds runtime")
    a = p.parse_args()
    seqs = {}
    for f in sorted(glob.glob(f"{a.routing}/*.npz")):
        z = np.load(f)
        for k in z.files:
            if k.startswith("layer_"):
                seqs.setdefault(int(k[6:]), []).append(z[k][: a.max_tokens])
    out = calibrate(seqs)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("half_life", "half_life_search", "tokens")}, indent=1))


if __name__ == "__main__":
    main()
