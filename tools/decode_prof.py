"""Decode profile at long context: where a decode step's time goes, per KV width, slots held fixed.

usage: PYTHONPATH=.:../lagunaLab python3 -m tools.decode_prof --config results/decode_prof.toml
           [--ctx 32000] [--tokens 64] [--slots 64] [--kv-bits 8] [--out results/decode_prof.json]

Run one kv_bits per process: a second load in the same process ran under memory pressure (the first
model was not freed; results/decode_prof.json, kv_bits 0 row). For that kv_bits: load with the same slots on every layer (so the KV width does not
change the slot count), prefill --ctx tokens (step 2048, as the server), then decode greedily twice
from the same prefix: once plain (the real tok/s) and once instrumented, with an mx.eval after every
attention and every layer, which splits a step into full-attention, sliding-attention, MoE (with its
own sync / read / write timers) and other (embedding, final norm, lm head, argmax). The
instrumented pass adds syncs, so its total is above the plain one; read the split as shares.
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

GB = 1e9


class Timed:
    """Stand-in for a decoder layer or its attention: runs it, evaluates the output, and adds the
    wall time of both to acc[key]."""

    def __init__(self, inner, acc, key):
        self.inner, self.acc, self.key = inner, acc, key

    def __call__(self, *a, **k):
        t = time.perf_counter()
        out = self.inner(*a, **k)
        mx.eval(out)
        self.acc[self.key] += time.perf_counter() - t
        return out

    def __getattr__(self, name):
        return getattr(self.inner, name)


def decode(model, cache, token, n, layer_types, instrument):
    """n greedy decode steps from token. Returns tokens, ms_per_tok, and with instrument the split (ms per token)."""
    from moestream.layer import stream_stats

    layers = model.layers
    acc = {"attn_full": 0.0, "attn_sliding": 0.0, "layer": 0.0}
    saved = list(layers)
    if instrument:
        for i, (l, lt) in enumerate(zip(saved, layer_types)):
            l.self_attn = Timed(l.self_attn, acc, "attn_full" if lt == "full_attention" else "attn_sliding")
            layers[i] = Timed(l, acc, "layer")
    s0 = stream_stats(model)
    out, t = [], token
    t0 = time.perf_counter()
    try:
        for _ in range(n):
            t = int(mx.argmax(model(mx.array([[t]]), cache=cache)[0, -1]))
            out.append(t)
    finally:
        secs = time.perf_counter() - t0
        layers[:] = saved
        for l in saved:
            if isinstance(l.self_attn, Timed):
                l.self_attn = l.self_attn.inner
    s1 = stream_stats(model)
    ms = 1000 / n
    row = {"tokens": out, "ms_per_tok": secs * ms, "tok_s": n / secs}
    d = {k: s1[k] - s0[k] for k in ("hits", "misses", "sync_s", "read_s", "write_s")}
    row["hit_rate"] = d["hits"] / max(1, d["hits"] + d["misses"])
    for k in ("sync", "read", "write"):
        row[f"{k}_ms"] = d[f"{k}_s"] * ms
    if instrument:  # a layer's wall time holds its attention's; the rest of the layer is the MLP/MoE
        row["attn_full_ms"] = acc["attn_full"] * ms
        row["attn_sliding_ms"] = acc["attn_sliding"] * ms
        row["moe_ms"] = (acc["layer"] - acc["attn_full"] - acc["attn_sliding"]) * ms
        row["other_ms"] = (secs - acc["layer"]) * ms
    return row


def prefill(model, ids, step=2048):
    cache = model.make_cache()
    t0 = time.perf_counter()
    for a in range(0, len(ids) - 1, step):
        model(mx.array([ids[a : min(a + step, len(ids) - 1)]]), cache=cache)
        mx.eval([c.state for c in cache])
    logits = model(mx.array([ids[-1:]]), cache=cache)
    return cache, int(mx.argmax(logits[0, -1])), len(ids) / (time.perf_counter() - t0)


def main():
    from moestream.config import EngineConfig
    from moestream.layer import stream_stats
    from moestream.load import load
    from tools.serve import cap_buffer_pool

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ctx", type=int, default=32000)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--slots", type=int, default=64)
    p.add_argument("--kv-bits", default="8", help="one value; see the module docstring")
    p.add_argument("--out", default="results/decode_prof.json")
    a = p.parse_args()
    cfg = EngineConfig.read(a.config)
    cap_buffer_pool(cfg.mlx_cache_gb)
    text = Path("../lagunaLab/tests/reference/modeling_laguna.py").read_text()  # tools.bench's long prompt
    rows = []
    for bits in (int(b) for b in a.kv_bits.split(",")):
        mx.reset_peak_memory()
        model, tok = load(cfg.pack_dir, slots=a.slots, calibration=cfg.calibration, kv_bits=bits,
                          eviction=cfg.eviction, target_ctx=2 * cfg.context_tokens)
        base = tok.encode(text)
        ids = (base * (a.ctx // len(base) + 1))[: a.ctx]
        lt = model.args.layer_types
        cache, t, pf = prefill(model, ids)
        plain = decode(model, cache, t, a.tokens, lt, instrument=False)
        cache, t, _ = prefill(model, ids)
        timed = decode(model, cache, t, a.tokens, lt, instrument=True)
        row = {"kv_bits": bits, "ctx": a.ctx, "slots_per_layer": a.slots,
               "slots_total": sum(stream_stats(model)["slots"].values()), "prefill_tok_s": pf,
               "plain": {k: v for k, v in plain.items() if k != "tokens"},
               "instrumented": {k: v for k, v in timed.items() if k != "tokens"},
               "same_tokens": plain["tokens"] == timed["tokens"], "peak_gb": mx.get_peak_memory() / GB}
        rows.append(row)
        print(json.dumps(row), flush=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))
        del model, cache
        mx.clear_cache()


if __name__ == "__main__":
    main()
