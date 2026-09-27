"""Record per-layer expert routing over captured agent sessions, for offline calibration.

PYTHONPATH=.:../lagunaLab python3 -m tools.record [--max-tokens 4096] [--budget-gb 5]
Loads real weights (~budget + 1.5 GB): message the other sessions and wait for a go-ahead first.
Teacher-forced prefill gives the same routing as decode, token by token.
"""
import argparse
import glob
from pathlib import Path

import mlx.core as mx
import numpy as np

from moestream.layer import StreamingMoE


def record_ids(model, ids, chunk=1024):
    moe = {i: l.mlp for i, l in enumerate(model.layers) if isinstance(l.mlp, StreamingMoE)}
    for m in moe.values():
        m.io.trace = []
    try:
        cache = model.make_cache()
        ids = np.asarray(ids)
        for a in range(0, len(ids), chunk):
            mx.eval(model(mx.array(ids[None, a : a + chunk]), cache=cache))
        return {i: np.concatenate(m.io.trace) for i, m in moe.items()}
    finally:
        for m in moe.values():
            m.io.trace = None


def main():
    from transformers import AutoTokenizer

    from moestream.load import load
    from replay.session import load_session, tokenize

    p = argparse.ArgumentParser()
    p.add_argument("--pack", default="../lagunaLab/models/Laguna-XS-2.1-packed")
    p.add_argument("--traces", default="../lagunaLab/traces")
    p.add_argument("--out", default="data/routing")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--budget-gb", type=float, default=5.0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, _ = load(a.pack, budget_gb=a.budget_gb)
    tok = AutoTokenizer.from_pretrained(a.pack)
    for f in sorted(glob.glob(f"{a.traces}/*.jsonl")):
        name = Path(f).stem
        if (out / f"{name}.npz").exists():
            continue  # resumable
        msgs, tools = load_session(f)
        ids, mask, _ = tokenize(tok, msgs, tools, max_tokens=a.max_tokens)
        rec = record_ids(model, ids)
        np.savez(out / f"{name}.npz", assistant_mask=mask, **{f"layer_{l}": r for l, r in rec.items()})
        print(name, len(ids), flush=True)


if __name__ == "__main__":
    main()
