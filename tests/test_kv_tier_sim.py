from tools.kv_tier_sim import interleave, simulate

# two sessions of three turns: (prompt tokens, held after the turn, previous held is a prefix)
S = [[(100, 150, False), (200, 250, True), (300, 350, True)],
     [(100, 150, False), (200, 250, True), (300, 350, True)]]


def test_one_session_reuses_its_previous_turn_from_ram():
    t = simulate(S[:1], [(0, 0), (0, 1), (0, 2)], ram_tokens=1000)
    assert t["from_ram"] == 150 + 250 and t["recomputed"] == 600 - 400


def test_interleaved_sessions_over_the_ram_cap_recompute_and_the_ssd_tier_restores():
    order = interleave(S, 2)
    assert order == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    ram = simulate(S, order, ram_tokens=200)          # one 150-token entry fits, not two
    ssd = simulate(S, order, ram_tokens=200, ssd_tokens=10_000)
    assert ram["from_ram"] == 0 and ram["recomputed"] == ram["prompt"]
    assert ssd["from_ssd"] == 150 + 150 + 250 + 250 and ssd["recomputed"] == ram["recomputed"] - 800


def test_restart_clears_ram_but_not_the_ssd_tier():
    order = [(0, 0), (0, 1), (0, 2)]
    t = simulate(S[:1], order, ram_tokens=1000, ssd_tokens=1000, restart_at=2)
    assert t["from_ram"] == 150 and t["from_ssd"] == 250


def test_a_broken_prefix_chain_reuses_nothing():
    broken = [[(100, 150, False), (200, 250, False)]]
    t = simulate(broken, [(0, 0), (0, 1)], ram_tokens=1000, ssd_tokens=1000)
    assert t["recomputed"] == t["prompt"]
