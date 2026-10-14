import json
from concurrent.futures import Future

import numpy as np
import pytest

from moestream.tune import Knob, Tuner


class Now:
    """Runs replay work inline: futures come back done."""

    def submit(self, fn, *a):
        f = Future()
        try:
            f.set_result(fn(*a))
        except Exception as e:
            f.set_exception(e)
        return f


def _live(cur, propose="new", **kw):
    return Knob("x", "live", lambda: cur["v"], lambda v: cur.update(v=v), lambda: propose, **kw)


def _ab(tmp_path, slow_pairs):
    cur, t = {"v": "old"}, [0.0]
    tuner = Tuner([_live(cur)], lambda: None, tmp_path, "fp", clock=lambda: t[0])
    tuner.idle()
    tuner.on_token()
    while tuner.trial is not None:
        p = len(tuner.trial["pairs"])
        t[0] += 1.0 if cur["v"] == "old" else (1.1 if p < slow_pairs else 0.9)
        tuner.on_token()
    return cur["v"], tuner


def test_live_candidate_kept_at_seven_of_eight(tmp_path):
    assert _ab(tmp_path, 1)[0] == "new"


def test_live_candidate_reverted_at_six_of_eight(tmp_path):
    v, tuner = _ab(tmp_path, 2)
    assert v == "old"
    rec = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    trials = [r for r in rec if "kept" in r]  # stats windows share the log
    assert trials[-1]["knob"] == "x" and trials[-1]["kept"] is False


def test_interrupt_drops_half_pair_and_reverts_arm(tmp_path):
    cur, t = {"v": "old"}, [0.0]
    tuner = Tuner([_live(cur)], lambda: None, tmp_path, "fp", clock=lambda: t[0])
    tuner.idle()
    for _ in range(Tuner.WARM + Tuner.WINDOW + 1):
        t[0] += 1; tuner.on_token()
    assert cur["v"] == "new"              # candidate arm running
    tuner.interrupt()
    assert cur["v"] == "old" and tuner.trial["pending"] is None and tuner.trial["pairs"] == []


def test_rejected_knob_waits_before_retry(tmp_path):
    v, tuner = _ab(tmp_path, 8)
    tuner.retry_tokens = 10
    tuner.idle()
    assert tuner.trial is None
    for _ in range(11):
        tuner.on_token()
    tuner.idle()
    assert tuner.trial is not None


def test_knob_not_ready_is_skipped(tmp_path):
    cur = {"v": "old"}
    tuner = Tuner([_live(cur, ready=lambda: False)], lambda: None, tmp_path, "fp")
    tuner.idle()
    assert tuner.trial is None


def test_failing_apply_disables_knob_and_reverts(tmp_path):
    cur = {"v": "old"}

    def apply(v):
        if v == "new":
            raise RuntimeError("no")
        cur.update(v=v)

    t = [0.0]
    tuner = Tuner([Knob("x", "live", lambda: cur["v"], apply, lambda: "new")], lambda: None, tmp_path, "fp",
                  clock=lambda: t[0])
    tuner.idle()
    for _ in range(Tuner.WARM + Tuner.WINDOW + 1):
        t[0] += 1; tuner.on_token()
    assert cur["v"] == "old" and "x" in tuner.disabled and tuner.trial is None


def _replay_snap():
    rows = np.array([[0, 1]] * 300 + [[2, 3], [4, 5]] * 150, np.uint8)
    return dict(rows={1: rows}, slots={1: 4}, policy="lfu", half_life=float("inf"), num_experts=8, grid=[2, 4, 8])


def test_replay_candidate_applied_when_simulation_is_lower(tmp_path):
    cur = {"v": "lfu"}
    knob = Knob("policy", "replay", lambda: cur["v"], lambda v: cur.update(v=v))
    tuner = Tuner([knob], _replay_snap, tmp_path, "fp", pool=Now())
    tuner.idle(); tuner.idle()
    assert cur["v"] in ("lru", "fifo") and tuner.trial is None


def test_replay_worker_error_ends_trial_without_raising(tmp_path):
    cur = {"v": "lfu"}
    knob = Knob("policy", "replay", lambda: cur["v"], lambda v: cur.update(v=v))
    tuner = Tuner([knob], lambda: {"rows": None}, tmp_path, "fp", pool=Now())
    tuner.idle(); tuner.idle()
    assert cur["v"] == "lfu" and tuner.trial is None


class FakeGuard:
    def __init__(self, ok):
        self.ready = self.math_ok = True
        self.ok, self.started = ok, 0

    def start(self):
        self.started += 1

    def step(self):
        return {"ok": self.ok, "kl_mean": 0.0}


@pytest.mark.parametrize("ok", [True, False])
def test_math_changing_candidate_is_kept_only_if_the_guard_passes(tmp_path, ok):
    cur = {"v": "lfu"}
    knob = Knob("policy", "replay", lambda: cur["v"], lambda v: cur.update(v=v), changes_math=True)
    g = FakeGuard(ok)
    tuner = Tuner([knob], _replay_snap, tmp_path, "fp", pool=Now(), guard=g)
    tuner.idle(); tuner.idle()
    assert tuner.trial["phase"] == "guard" and cur["v"] != "lfu" and g.started == 1
    tuner.idle()
    assert (cur["v"] != "lfu") is ok and tuner.trial is None


def test_math_changing_knob_off_when_guard_is_not_quiet(tmp_path):
    knob = Knob("policy", "replay", lambda: "lfu", lambda v: None, changes_math=True)
    g = FakeGuard(True)
    g.math_ok = False
    tuner = Tuner([knob], _replay_snap, tmp_path, "fp", pool=Now(), guard=g)
    tuner.idle()
    assert tuner.trial is None


def test_state_round_trip(tmp_path):
    cur = {"v": "a"}
    Tuner([_live(cur)], lambda: None, tmp_path, "fp").save()
    cur["v"] = "b"
    Tuner([_live(cur)], lambda: None, tmp_path, "fp").load_state()
    assert cur["v"] == "a"


def test_state_fingerprint_mismatch_is_ignored(tmp_path):
    cur = {"v": "a"}
    Tuner([_live(cur)], lambda: None, tmp_path, "fp").save()
    cur["v"] = "b"
    with pytest.warns(UserWarning, match="different pack or config"):
        Tuner([_live(cur)], lambda: None, tmp_path, "other").load_state()
    assert cur["v"] == "b"


def test_saved_value_that_no_longer_applies_is_dropped(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"fingerprint": "fp", "values": {"x": "huge"}}))

    def apply(v):
        raise ValueError("over the cap")

    with pytest.warns(UserWarning, match="saved x dropped"):
        Tuner([Knob("x", "live", lambda: "a", apply, lambda: None)], lambda: None, tmp_path, "fp").load_state()


def test_stats_windows_are_logged(tmp_path):
    t, hm = [0.0], [0, 0]
    tuner = Tuner([], lambda: None, tmp_path, "fp", clock=lambda: t[0], counts=lambda: tuple(hm))
    tuner.on_token()
    for _ in range(Tuner.STATS):
        t[0] += 0.1; hm[0] += 3; hm[1] += 1
        tuner.on_token()
    rec = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[-1])
    assert rec["tok_s"] == pytest.approx(10.0) and rec["hit_rate"] == 0.75


def test_start_wires_hooks_traces_and_resumes_state(tmp_path):
    import mlx.core as mx

    from moestream.config import EngineConfig
    from moestream.layer import StreamingMoE
    from moestream.load import load
    from moestream.tune import Ring, start
    from scripts.pack_experts import pack
    from tests.tiny import quantized_model_dir

    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    m, _ = load(tmp_path / "packed", slots=4)
    cfg = EngineConfig(pack_dir=str(tmp_path / "packed"), tuning_dir=str(tmp_path / "t"), tune=False, kv_bits=0)
    tuner = start(m, None, cfg)
    moe = [l.mlp for l in m.layers if isinstance(l.mlp, StreamingMoE)]
    assert moe[-1].io.hook is not None and all(x.io.trace is None for x in moe)  # tune=false: no traces
    tuner.knobs["policy"].apply("fifo")
    tuner.save()
    m2, _ = load(tmp_path / "packed", slots=4)
    start(m2, None, cfg)
    assert {l.mlp.io.store.policy for l in m2.layers if isinstance(l.mlp, StreamingMoE)} == {"fifo"}
    cfg.tune = True
    m3, _ = load(tmp_path / "packed", slots=4)
    t3 = start(m3, None, cfg, prompts=[[1, 2, 3, 4]], probe_tokens=8)
    assert isinstance(m3.layers[1].mlp.io.trace, Ring)
    m3(mx.array([[1, 2, 3]]))
    assert t3.tokens == 0  # prefill is not a decode token
    t3.pool.shutdown()


def test_ab_runs_abba_after_a_warm_up():
    import tempfile

    cur, t, switches = {"v": "old"}, [0.0], []

    def apply(v):
        switches.append((tuner.tokens, v)); cur.update(v=v)

    tuner = Tuner([Knob("x", "live", lambda: cur["v"], apply, lambda: "new")], lambda: None, tempfile.mkdtemp(), "fp",
                  clock=lambda: t[0])
    tuner.idle()
    for _ in range(Tuner.WINDOW + 1):
        t[0] += 1; tuner.on_token()
    assert switches == []                      # the first tokens after a break only warm the cache
    for _ in range(Tuner.WARM + 3 * Tuner.WINDOW):
        t[0] += 1; tuner.on_token()
    assert [v for _, v in switches] == ["new", "old"]
    assert switches[1][0] - switches[0][0] == 2 * Tuner.WINDOW  # pair 2 starts with the candidate: A B B A


def test_live_trial_without_pairs_ends_as_no_decision(tmp_path):
    cur, t = {"v": "old"}, [0.0]
    tuner = Tuner([_live(cur)], lambda: None, tmp_path, "fp", clock=lambda: t[0])
    tuner.idle()
    assert tuner.trial is not None
    t[0] += Tuner.AB_TIMEOUT + 1
    tuner.idle()
    assert tuner.trial is None and cur["v"] == "old" and "x" not in tuner.rejected


def test_dead_worker_pool_disables_replay_knobs_without_raising(tmp_path):
    class Dead:
        def submit(self, *a):
            raise RuntimeError("BrokenProcessPool")

    knob = Knob("policy", "replay", lambda: "lfu", lambda v: None)
    tuner = Tuner([knob], _replay_snap, tmp_path, "fp", pool=Dead())
    tuner.idle()
    assert tuner.trial is None and "policy" in tuner.disabled


def test_cancel_ends_a_guard_phase_trial_and_returns_the_old_value(tmp_path):
    cur = {"v": "lfu"}
    knob = Knob("policy", "replay", lambda: cur["v"], lambda v: cur.update(v=v), changes_math=True)
    tuner = Tuner([knob], _replay_snap, tmp_path, "fp", pool=Now(), guard=FakeGuard(True))
    tuner.idle(); tuner.idle()
    assert tuner.trial["phase"] == "guard"
    assert tuner.cancel("policy") == "lfu" and tuner.trial is None and "policy" not in tuner.rejected
    tuner.idle()
    assert tuner.trial is None or tuner.trial["phase"] == "replay"


def test_hook_error_detaches_tuning_and_decode_goes_on(tmp_path):
    import mlx.core as mx

    from moestream.config import EngineConfig
    from moestream.load import load
    from moestream.tune import start
    from scripts.pack_experts import pack
    from tests.tiny import quantized_model_dir

    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    m, _ = load(tmp_path / "packed", slots=4)
    tuner = start(m, None, EngineConfig(pack_dir=str(tmp_path / "packed"), tuning_dir=str(tmp_path / "t"), tune=False,
                                        kv_bits=0))

    def boom():
        raise OSError("disk full")

    tuner.interrupt = boom
    with pytest.warns(UserWarning, match="tuning off"):
        m(mx.array([[1, 2, 3]]))
    assert m.layers[-1].mlp.io.hook is None


def test_replay_knob_goes_before_a_live_knob_next_in_turn(tmp_path):
    cur = {"v": "old"}
    replay = Knob("policy", "replay", lambda: "lfu", lambda v: None)
    tuner = Tuner([replay, _live(cur)], _replay_snap, tmp_path, "fp", pool=Now())
    tuner.next_i = 1  # the live knob's turn, as after a replay trial
    tuner.idle()
    assert tuner.trial["knob"] == "policy"


def test_live_trial_that_keeps_gaining_pairs_still_ends_at_the_cap(tmp_path):
    cur, t = {"v": "old"}, [0.0]
    tuner = Tuner([_live(cur)], lambda: None, tmp_path, "fp", clock=lambda: t[0])
    tuner.idle()
    tuner.trial["pairs"] = [(1.0, 0.9)] * 3
    t[0] += Tuner.AB_MAX + 1
    tuner.trial["last_pair"] = t[0]  # a pair just finished: the no-pair timeout alone would not end it
    tuner.idle()
    assert tuner.trial is None and cur["v"] == "old" and "x" not in tuner.rejected


def test_ab_pair_fits_in_an_agent_turn():
    assert Tuner.WARM + 4 * Tuner.WINDOW <= 160  # agent replies average about 100-200 tokens
