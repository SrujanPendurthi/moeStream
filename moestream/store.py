"""Per-layer expert slot bookkeeping: which expert sits in which slot, and who gets evicted.

Pure Python/numpy: no MLX, no threads. Eviction is decayed LFU with a recency tie-break,
so experts that were hot long ago stop pinning slots in long sessions.
"""
import numpy as np


class SlotStore:
    def __init__(self, num_experts, slots, half_life=256.0):
        self.slots = slots
        self.decay = 0.5 ** (1.0 / half_life)
        self.slot_expert = [-1] * slots
        self.expert_slot = {}
        self.score = np.zeros(num_experts)
        self.last = np.zeros(num_experts, np.int64)
        self.clock = 0
        self.hits = self.misses = 0

    def tick(self, n_tokens=1):
        """Age every score by n_tokens tokens. Call once per layer call, before assign()."""
        self.clock += 1
        self.score *= self.decay**n_tokens

    def assign(self, experts):
        """Make `experts` (distinct) resident. Returns ({expert: slot}, [(expert, slot) to fill])."""
        if len(experts) > self.slots:
            raise ValueError(f"{len(experts)} experts > {self.slots} slots")
        keep = set(experts)
        missing = [e for e in experts if e not in self.expert_slot]
        self.hits += len(experts) - len(missing)
        self.misses += len(missing)
        self.score[experts] += 1
        self.last[experts] = self.clock
        fills = list(zip(missing, self._victims(len(missing), keep)))
        for e, s in fills:
            old = self.slot_expert[s]
            if old != -1:
                del self.expert_slot[old]
            self.slot_expert[s], self.expert_slot[e] = e, s
        return {e: self.expert_slot[e] for e in experts}, fills

    def _victims(self, n, keep):
        free = [s for s, e in enumerate(self.slot_expert) if e == -1]
        used = [s for s, e in enumerate(self.slot_expert) if e != -1 and e not in keep]
        used.sort(key=lambda s: (self.score[self.slot_expert[s]], self.last[self.slot_expert[s]]))
        return (free + used)[:n]
