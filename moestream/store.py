"""Per-layer expert slot bookkeeping: which expert sits in which slot, and who gets evicted.

Pure Python/numpy: no MLX, no threads. Eviction is decayed LFU with a recency tie-break
(default; experts that were hot long ago stop pinning slots in long sessions), LRU or FIFO.
"""
import numpy as np


POLICIES = ("lfu", "lru", "fifo")


class SlotStore:
    def __init__(self, num_experts, slots, half_life=256.0, policy="lfu"):
        self.slots = slots
        self.set_policy(policy)
        self.set_half_life(half_life)
        self.slot_expert = [-1] * slots
        self.expert_slot = {}
        self.score = np.zeros(num_experts)
        self.last = np.zeros(num_experts, np.int64)
        self.loaded = np.zeros(num_experts, np.int64)
        self.clock = 0
        self.hits = self.misses = 0

    def set_policy(self, policy):
        if policy not in POLICIES:
            raise ValueError(f"eviction policy {policy!r} not in {POLICIES}")
        self.policy = policy

    def set_half_life(self, half_life):
        self.half_life = half_life
        self.decay = 0.5 ** (1.0 / half_life)

    def key(self, e):
        """Eviction rank of a resident expert: the lowest key is evicted first."""
        if self.policy == "lfu":
            return (self.score[e], self.last[e])
        return (self.last[e] if self.policy == "lru" else self.loaded[e],)

    def tick(self, n_tokens=1):
        """Age every score by n_tokens tokens. Call once per layer call, before assign()."""
        self.clock += 1
        self.score *= self.decay**n_tokens

    def assign(self, experts, counts=1):
        """Make `experts` (distinct) resident; `counts` = tokens using each. Returns ({expert: slot}, [(expert, slot) to fill])."""
        if len(experts) > self.slots:
            raise ValueError(f"{len(experts)} experts > {self.slots} slots")
        keep = set(experts)
        missing = [e for e in experts if e not in self.expert_slot]
        self.hits += len(experts) - len(missing)
        self.misses += len(missing)
        self.score[experts] += counts
        self.last[experts] = self.clock
        fills = list(zip(missing, self._victims(len(missing), keep)))
        for e, s in fills:
            old = self.slot_expert[s]
            if old != -1:
                del self.expert_slot[old]
            self.slot_expert[s], self.expert_slot[e] = e, s
            self.loaded[e] = self.clock
        return {e: self.expert_slot[e] for e in experts}, fills

    def _victims(self, n, keep):
        free = [s for s, e in enumerate(self.slot_expert) if e == -1]
        used = [s for s, e in enumerate(self.slot_expert) if e != -1 and e not in keep]
        used.sort(key=lambda s: self.key(self.slot_expert[s]))
        return (free + used)[:n]
