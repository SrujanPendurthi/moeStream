"""StreamingMoE: Laguna's MoE block with routed experts streamed from SSD into per-layer slots.

One GPU sync per layer returns this layer's routing and, during decode, a guess at the
next layer's (the next router run on this layer's input). The guess is read from disk in
the background while the GPU works; slots are only filled when a layer uses an expert.
"""
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import SwitchGLU

from laguna_mlx import laguna
from moestream.prefetch import Prefetcher
from moestream.store import SlotStore
from scripts.pack_experts import expert_from_bytes

EVAL_WRITES = False  # diagnostic: force slot writes inside write_s instead of the next layer's sync_s


def _glu(args, slots, group_size, bits, dtype):
    glu = SwitchGLU(args.hidden_size, args.moe_intermediate_size, slots)
    glu.set_dtype(dtype)  # match packed scales dtype: no cast on write
    nn.quantize(glu, group_size=group_size, bits=bits)
    return glu


class StreamingMoE(nn.Module):
    route = laguna.SparseMoeBlock.route

    def __init__(self, args, layer, pack_dir, layout, slots, half_life, group_size, bits, dtype, policy="lfu"):
        super().__init__()
        h = args.hidden_size
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.moe_routed_scaling_factor
        self.gate = nn.Linear(h, args.num_experts, bias=False)
        self.e_score_correction_bias = mx.zeros((args.num_experts,))
        self.shared_expert = laguna.MLP(h, args.shared_expert_intermediate_size)
        # SimpleNamespace keeps slots and links out of the parameter tree (load_weights strict=True)
        self.io = SimpleNamespace(
            glu=_glu(args, slots, group_size, bits, dtype),
            make_glu=lambda n: _glu(args, n, group_size, bits, dtype),
            layout=layout, next=None, predictor=None, trace=None, guess=None, lookahead=True, hook=None,
            store=SlotStore(args.num_experts, slots, half_life, policy),
            prefetcher=Prefetcher(Path(pack_dir) / "experts" / f"layer_{layer:02d}.bin", layout["stride"],
                                  keep=2 * self.top_k),
            stats=dict(pred_hit=0, pred_total=0, sync_s=0.0, read_s=0.0, write_s=0.0),
        )

    def __call__(self, x):
        io = self.io
        inds, w = self.route(x)
        decode = x.shape[0] * x.shape[1] == 1
        lookahead = decode and io.next is not None and io.predictor is None and io.lookahead
        t0 = time.perf_counter()
        if lookahead:
            pinds = io.next.route(x)[0]
            mx.eval(inds, pinds)
        else:
            mx.eval(inds)
        io.stats["sync_s"] += time.perf_counter() - t0
        idx = np.array(inds)
        if io.trace is not None:
            io.trace.append(idx.reshape(-1, idx.shape[-1]).astype(np.uint8))
        experts, counts = (a.tolist() for a in np.unique(idx, return_counts=True))
        if io.guess is not None:
            io.stats["pred_hit"] += len(io.guess & set(experts))
            io.stats["pred_total"] += len(experts)
            io.guess = None
        io.store.tick(idx.size // idx.shape[-1])
        y = None
        for g in range(0, len(experts), io.store.slots):
            yg = self._run(x, idx, experts[g : g + io.store.slots], counts[g : g + io.store.slots],
                           masked=len(experts) > io.store.slots)
            y = yg if y is None else y + yg
        io.prefetcher.drop_pending()
        if decode and io.next is not None and (lookahead or io.predictor is not None):
            guess = np.unique(np.array(pinds)).tolist() if lookahead else list(io.predictor(x))
            io.next.io.guess = set(guess)
            io.next.io.prefetcher.submit([e for e in guess if e not in io.next.io.store.expert_slot])
        y = (y * w[..., None]).sum(axis=-2)
        if io.hook is not None:
            io.hook(decode)
        return y * self.routed_scaling_factor + self.shared_expert(x)

    def resize(self, n):
        """Change this layer's slot count between tokens, keeping the n experts the eviction key ranks highest."""
        io, old = self.io, self.io.store
        keep = sorted(old.expert_slot, key=old.key, reverse=True)[:n]
        store = SlotStore(len(old.score), n, old.half_life, old.policy)
        store.score, store.last, store.loaded = old.score, old.last, old.loaded
        store.clock, store.hits, store.misses = old.clock, old.hits, old.misses
        glu = io.make_glu(n)
        if keep:
            src = mx.array([old.expert_slot[e] for e in keep])
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for part in ("weight", "scales", "biases"):
                    getattr(getattr(glu, proj), part)[: len(keep)] = getattr(getattr(io.glu, proj), part)[src]
        for s, e in enumerate(keep):
            store.slot_expert[s], store.expert_slot[e] = e, s
        mx.eval(glu.parameters())
        io.glu, io.store = glu, store

    def _run(self, x, idx, group, counts, masked):
        io = self.io
        slot_of, fills = io.store.assign(group, counts)
        t0 = time.perf_counter()
        bufs = io.prefetcher.take([e for e, _ in fills])
        t1 = time.perf_counter()
        for e, s in fills:
            for name, arr in expert_from_bytes(bufs[e], io.layout).items():  # mx.array copies here
                proj, part = name.split(".")
                getattr(getattr(io.glu, proj), part)[s] = arr
            io.prefetcher.release(bufs[e])
        if EVAL_WRITES and fills:
            mx.eval(io.glu.parameters())
        io.stats["read_s"] += t1 - t0
        io.stats["write_s"] += time.perf_counter() - t1
        E = io.layout["num_experts"]
        table, member = np.zeros(E, np.int32), np.zeros(E, bool)
        for e, s in slot_of.items():
            table[e], member[e] = s, True
        y = io.glu(x, mx.array(table[idx]))
        if masked:
            y = mx.where(mx.array(member[idx])[..., None], y, 0)
        return y


def stream_stats(model):
    moe = [l.mlp for l in model.layers if isinstance(l.mlp, StreamingMoE)]
    hits = sum(m.io.store.hits for m in moe)
    misses = sum(m.io.store.misses for m in moe)
    tot = {k: sum(m.io.stats[k] for m in moe) for k in moe[0].io.stats}
    return {
        "hits": hits, "misses": misses, "hit_rate": hits / max(1, hits + misses),
        "pred_acc": tot["pred_hit"] / max(1, tot["pred_total"]),
        "sync_s": tot["sync_s"], "read_s": tot["read_s"], "write_s": tot["write_s"],
        "bytes_read": misses * moe[0].io.layout["stride"],
        "slots": {i: l.mlp.io.store.slots for i, l in enumerate(model.layers) if isinstance(l.mlp, StreamingMoE)},
    }
