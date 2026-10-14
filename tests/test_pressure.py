from moestream.budget import GB
from moestream.pressure import Pressure, shrink, swap_used


def test_shrink_takes_ten_percent_from_the_largest_layers():
    s = shrink({1: 50, 2: 30, 3: 20}, floor=16)
    assert sum(s.values()) == 90 and s[1] < 50 and s[3] == 20


def test_shrink_never_goes_under_the_floor():
    assert shrink({1: 16, 2: 16}, floor=16) == {1: 16, 2: 16}


def test_swap_used_reads_sysctl():
    assert swap_used() >= 0


class Split:
    def __init__(self):
        self.v = {1: 100, 2: 100}

    def get(self):
        return dict(self.v)

    def apply(self, v):
        self.v = dict(v)


def test_low_memory_shrinks_and_calm_restores():
    k, mem, now, logs = Split(), {"free": 0.5 * GB, "swap": 0.0}, [0.0], []
    p = Pressure(k, 16, every=1, free=lambda: mem["free"], swap=lambda: mem["swap"], clock=lambda: now[0],
                 log=logs.append)
    p.on_token()
    assert p.shrunk and sum(k.v.values()) == 180
    mem["free"] = 4 * GB
    p.on_token(); now[0] += 299; p.on_token()
    assert p.shrunk
    now[0] += 2; p.on_token()
    assert not p.shrunk and k.v == {1: 100, 2: 100} and len(logs) == 2


def test_growing_swap_shrinks_even_with_free_memory():
    k, mem = Split(), {"swap": 0.0}
    p = Pressure(k, 16, every=1, free=lambda: 8 * GB, swap=lambda: mem["swap"], clock=lambda: 0.0, log=lambda r: None)
    p.on_token()
    assert not p.shrunk
    mem["swap"] = 1e6
    p.on_token()
    assert p.shrunk and sum(k.v.values()) < 200


def _knob():
    from moestream.tune import Knob

    s = Split()
    return s, Knob("slots", "replay", s.get, s.apply, changes_math=True)


def test_guarded_knob_writes_go_to_the_restore_target_while_shrunk():
    s, k = _knob()
    p = Pressure(k, 16, every=1, free=lambda: 0.5 * GB, swap=lambda: 0.0, clock=lambda: 0.0, log=lambda r: None)
    g = p.guarded()
    p.on_token()
    shrunk = dict(s.v)
    assert g.get() == {1: 100, 2: 100} and not g.ready()      # saved state never records the shrunk split
    g.apply({"1": 120, "2": 80})
    assert s.v == shrunk and p.target == {1: 120, 2: 80}      # no resize while memory is short


def test_shrink_cancels_a_slot_trial_and_restores_its_old_split():
    s, k = _knob()
    s.v = {1: 130, 2: 70}                                     # candidate under its KL guard
    p = Pressure(k, 16, every=1, free=lambda: 0.5 * GB, swap=lambda: 0.0, clock=lambda: 0.0, log=lambda r: None,
                 cancel=lambda: {1: 100, 2: 100})
    p.on_token()
    assert p.target == {1: 100, 2: 100} and sum(s.v.values()) == 180
