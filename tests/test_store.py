import pytest

from moestream.store import SlotStore


def test_empty_slots_fill_before_eviction():
    s = SlotStore(8, 3)
    s.tick(); _, f = s.assign([0, 1])
    assert f == [(0, 0), (1, 1)]
    s.tick(); _, f = s.assign([2])
    assert f == [(2, 2)]
    assert (s.misses, s.hits) == (3, 0)


def test_never_evicts_current_set():
    s = SlotStore(8, 2)
    s.tick(); s.assign([0, 1])
    s.tick(); slot_of, f = s.assign([1, 2])
    assert f == [(2, 0)]
    assert slot_of == {1: 1, 2: 0} and s.expert_slot == {1: 1, 2: 0}
    assert (s.misses, s.hits) == (3, 1)


def test_lowest_score_evicted_recency_breaks_ties():
    s = SlotStore(8, 3, half_life=float("inf"))
    for _ in range(3):
        s.tick(); s.assign([0])
    s.tick(); s.assign([1])
    s.tick(); s.assign([2])  # scores 0:3, 1:1, 2:1; expert 1 is older
    s.tick(); _, f = s.assign([3])
    assert f == [(3, 1)]


def test_old_heavy_use_decays_away():
    def run(half_life):
        s = SlotStore(8, 2, half_life=half_life)
        for _ in range(60):
            s.tick(); s.assign([0])   # hot long ago
        for _ in range(20):
            s.tick(); s.assign([1])   # warm recently
        s.tick(); _, f = s.assign([2])
        return f
    assert run(float("inf")) == [(2, 1)]  # plain LFU keeps stale expert 0
    assert run(4.0) == [(2, 0)]           # decayed LFU evicts it


def test_too_many_experts_raises():
    with pytest.raises(ValueError):
        SlotStore(8, 2).assign([0, 1, 2])


def test_assign_counts_weight_the_score():
    s = SlotStore(8, 2, half_life=float("inf"))
    s.tick(); s.assign([0, 1], counts=[5, 1])  # a prefill chunk: 0 used by 5 tokens, 1 by one
    s.tick(); _, f = s.assign([2])
    assert f == [(2, 1)]
