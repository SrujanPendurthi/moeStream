"""Self-optimizing runtime: measure, try one change, keep it only if it measured better.

Runs on mlx-lm's generation thread: on_token() after each decode step, idle() when no request
is waiting. Replay trials run in up to 2 low-priority worker processes, so this module stays
free of MLX imports.
"""
import numpy as np

from moestream.budget import allocate
from moestream.store import POLICIES
from tools.calibrate import HALF_LIVES, miss_rate


class Ring:
    """The last `cap` routing rows of one layer; StreamingMoE appends to it through io.trace."""

    def __init__(self, cap):
        self.cap, self.parts, self.n = cap, [], 0

    def append(self, rows):
        self.parts.append(rows)
        self.n += len(rows)
        if self.n > 2 * self.cap:  # ponytail: holds up to 2x cap rows between compactions
            self.parts = [self.rows()]
            self.n = len(self.parts[0])

    def rows(self):
        return np.concatenate(self.parts)[-self.cap :] if self.parts else None


def replay_rate(rows, slots, policy, half_life, num_experts):
    """Token-weighted simulated miss rate over layers; rows {layer: (T, k) uint8}, slots {layer: n}."""
    n = sum(len(r) for r in rows.values())
    return sum(miss_rate(r, slots[l], half_life, num_experts, policy) * len(r) for l, r in rows.items()) / max(1, n)


def replay_trial(knob, snap):
    """Worker: best candidate for one replay knob and both simulated miss rates; candidate None if none is lower."""
    rows, slots, policy, hl, E = (snap[k] for k in ("rows", "slots", "policy", "half_life", "num_experts"))
    if knob == "policy":
        cands = [dict(policy=p) for p in POLICIES if p != policy]
    elif knob == "half_life":
        cands = [dict(half_life=h) for h in HALF_LIVES if h != hl]
    else:
        curves = {l: [miss_rate(r, g, hl, E, policy) for g in snap["grid"]] for l, r in rows.items()}
        cands = [dict(slots=allocate(curves, snap["grid"], sum(slots.values())))]
    cur = replay_rate(rows, slots, policy, hl, E)
    best, best_rate = None, cur
    for c in cands:
        r = replay_rate(rows, c.get("slots", slots), c.get("policy", policy), c.get("half_life", hl), E)
        if r < best_rate:
            best, best_rate = next(iter(c.values())), r
    return best, cur, best_rate


def ab_keep(pairs, n=8, need=7):
    """pairs: (current seconds, candidate seconds) per matched window pair. None until n pairs."""
    if len(pairs) < n:
        return None
    return sum(c < a for a, c in pairs[:n]) >= need
