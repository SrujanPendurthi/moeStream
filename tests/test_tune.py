import numpy as np

from moestream.tune import Ring, ab_keep, replay_rate, replay_trial


def _phase_shift():
    # {0,1} hammered, then {2,3},{4,5} alternating: needs 4 slots; plain LFU keeps 0,1 pinned
    return np.array([[0, 1]] * 300 + [[2, 3], [4, 5]] * 150, np.uint8)


def _snap(rows, policy="lfu", half_life=float("inf"), slots=4, grid=(2, 4, 8)):
    return dict(rows={1: rows, 2: rows}, slots={1: slots, 2: slots}, policy=policy, half_life=half_life,
                num_experts=8, grid=list(grid))


def test_ring_keeps_the_last_cap_rows():
    r = Ring(5)
    assert r.rows() is None
    for i in range(12):
        r.append(np.full((1, 2), i, np.uint8))
    assert r.rows()[:, 0].tolist() == [7, 8, 9, 10, 11]
    r.append(np.zeros((3, 2), np.uint8))
    assert len(r.rows()) == 5


def test_replay_rate_matches_per_layer_miss_rate():
    rows = _phase_shift()
    lfu = replay_rate({1: rows}, {1: 4}, "lfu", float("inf"), 8)
    lru = replay_rate({1: rows}, {1: 4}, "lru", float("inf"), 8)
    assert lru < lfu


def test_policy_trial_finds_a_better_policy():
    value, cur, cand = replay_trial("policy", _snap(_phase_shift()))
    assert value in ("lru", "fifo") and cand < cur * 0.99


def test_half_life_trial_leaves_infinite_half_life():
    value, cur, cand = replay_trial("half_life", _snap(_phase_shift()))
    assert value != float("inf") and cand < cur


def test_trial_without_a_better_value_returns_none():
    rows = np.array([[0, 1]] * 100, np.uint8)  # fits: only cold misses, every policy ties
    value, cur, cand = replay_trial("policy", _snap(rows))
    assert value is None and cand == cur


def test_slot_trial_moves_slots_to_the_layer_that_misses():
    busy, quiet = _phase_shift(), np.array([[0, 1]] * 600, np.uint8)
    snap = _snap(busy, slots=3, half_life=4.0)  # infinite half-life pins {0,1}: 4 slots would not help
    snap["rows"] = {1: busy, 2: quiet}
    value, cur, cand = replay_trial("slots", snap)
    assert value == {1: 4, 2: 2} and cand < cur


def test_ab_keep_needs_seven_of_eight():
    faster = [(1.0, 0.9)] * 7 + [(1.0, 1.1)]
    assert ab_keep(faster[:7]) is None
    assert ab_keep(faster) is True
    assert ab_keep([(1.0, 0.9)] * 6 + [(1.0, 1.1)] * 2) is False
