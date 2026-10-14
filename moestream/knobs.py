"""The system-level knobs the tuner turns: eviction policy, half-life, slot split, lookahead prefetch, read threads."""
import mlx.core as mx
import numpy as np

from moestream import prefetch
from moestream.layer import StreamingMoE
from moestream.tune import Knob


def system_knobs(model, max_slots, ready=lambda: True):
    by = {i: l.mlp for i, l in enumerate(model.layers) if isinstance(l.mlp, StreamingMoE)}
    layers, moe = list(by), list(by.values())
    E = moe[0].io.layout["num_experts"]
    pre = [l for l in layers if by[l].io.next is not None]
    quarters = [[int(l) for l in q] for q in np.array_split(pre, 4) if len(q)]
    turn = {"q": 0, "t": 0}

    def set_policy(p):
        for m in moe:
            m.io.store.set_policy(p)

    def set_half_life(h):
        for m in moe:
            m.io.store.set_half_life(float(h))

    def get_split():
        return {l: m.io.store.slots for l, m in by.items()}

    def set_split(split):
        split = {int(l): int(n) for l, n in split.items()}
        if sorted(split) != layers or not all(1 <= n <= E for n in split.values()):
            raise ValueError(f"slot split {split} does not fit layers {layers} with {E} experts")
        if sum(split.values()) > max_slots:
            raise ValueError(f"slot split needs {sum(split.values())} slots; the cap allows {max_slots}")
        for l in sorted(split, key=lambda l: split[l] - by[l].io.store.slots):  # shrink before growing
            if split[l] != by[l].io.store.slots:
                by[l].resize(split[l])
        mx.clear_cache()  # freed slot buffers go back to the OS; MLX's buffer cache would keep them

    def get_prefetch():
        return [l for l in pre if by[l].io.lookahead]

    def set_prefetch(on):
        on = {int(l) for l in on}
        for l in pre:
            by[l].io.lookahead = l in on

    def propose_prefetch():
        q = quarters[turn["q"] % len(quarters)]
        turn["q"] += 1
        return sorted(set(get_prefetch()) ^ set(q))

    def propose_threads():
        turn["t"] ^= 1
        n = prefetch.threads() * 2 if turn["t"] else prefetch.threads() // 2
        return n if 2 <= n <= 16 else None

    store = lambda: moe[0].io.store  # noqa: E731
    return [
        Knob("policy", "replay", lambda: store().policy, set_policy),
        Knob("half_life", "replay", lambda: store().half_life, set_half_life),
        Knob("slots", "replay", get_split, set_split, changes_math=True, ready=ready),
        Knob("prefetch", "live", get_prefetch, set_prefetch, propose_prefetch),
        Knob("threads", "live", prefetch.threads, prefetch.set_threads, propose_threads),
    ]
