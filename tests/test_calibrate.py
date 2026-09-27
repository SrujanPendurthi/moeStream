import numpy as np

from tools.calibrate import calibrate, miss_rate


def test_miss_rate_falls_with_more_slots_and_hits_zero_when_all_fit():
    rng = np.random.default_rng(0)
    seq = rng.integers(0, 16, size=(400, 2)).astype(np.uint8)
    seq[:, 1] = (seq[:, 0] + 1) % 16  # distinct per row
    r = [miss_rate(seq, s, 256.0, 16) for s in (2, 4, 8, 16)]
    assert r == sorted(r, reverse=True)
    assert r[-1] == 16 / 800  # only cold misses


def test_calibrate_picks_short_half_life_for_phase_shift():
    # phase 1 hammers {0,1}; phase 2 alternates {2,3},{4,5} and needs all 4 slots.
    # Plain LFU keeps 0,1 pinned (miss every step); decay frees them within a few steps.
    phase2 = [[2, 3], [4, 5]] * 150
    seq = np.array([[0, 1]] * 300 + phase2, np.uint8)
    seqs = {1: [seq], 2: [seq]}
    out = calibrate(seqs, grid=[2, 4, 8], half_lives=[4.0, float("inf")], ref_slots=4, num_experts=8)
    assert set(out) >= {"half_life", "grid", "curves", "half_life_search", "tokens"}
    assert out["grid"] == [2, 4, 8] and sorted(out["curves"]) == ["1", "2"]
    assert all(len(c) == 3 for c in out["curves"].values())
    assert out["half_life_search"]["4.0"] < out["half_life_search"]["inf"]
    assert out["half_life"] == 4.0 and out["tokens"] == 600
