import numpy as np
import pytest

from moestream.guard import THRESHOLDS, Guard, kl_topk, quiet, reference, verdict
from moestream.load import load
from scripts.pack_experts import pack
from tests.tiny import TINY, quantized_model_dir


def test_identical_logits_pass_and_hot_logits_fail():
    lg = np.random.default_rng(0).normal(size=(50, 300)).astype(np.float32)
    ref = reference(lg, 64)
    v = verdict(*kl_topk(ref, lg))
    assert v["ok"] and v["kl_mean"] < 1e-9 and v["top1"] == 1.0 and quiet(v)
    assert not verdict(*kl_topk(ref, 3 * lg))["ok"]


def test_disagreement_past_half_the_threshold_is_not_quiet():
    lg = np.random.default_rng(1).normal(size=(400, 300)).astype(np.float32)
    v = verdict(*kl_topk(reference(lg, 64), lg + np.random.default_rng(2).normal(0, 0.06, lg.shape)))
    assert v["kl_mean"] > THRESHOLDS["mean"] / 2 and not quiet(v)


@pytest.fixture
def model(tmp_path):
    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    return load(tmp_path / "packed", slots=4)[0]


def _run(g):
    while True:
        v = g.step()
        if v is not None or (g.ready and g._run is None):
            return v


def test_guard_builds_reference_in_slices_then_passes_itself(model, tmp_path):
    prompts = [list(np.random.default_rng(s).integers(0, TINY["vocab_size"], 5)) for s in range(3)]
    g = Guard(model, prompts, tmp_path / "ref.npz", "k1", total=48, chunk=8)
    steps = 0
    while not g.ready:
        g.step(); steps += 1
    assert steps > 6 and g.math_ok and g.seqs.shape == (3, 16)
    g.start()
    assert _run(g)["ok"]
    again = Guard(model, prompts, tmp_path / "ref.npz", "k1", total=48, chunk=8)
    assert again.ready and again.math_ok                      # loaded, not rebuilt
    assert not Guard(model, prompts, tmp_path / "ref.npz", "k2", total=48, chunk=8).ready


def test_guard_fails_a_changed_model_and_survives_errors(model, tmp_path):
    prompts = [[1, 2, 3, 4]]
    g = Guard(model, prompts, tmp_path / "ref.npz", "k", total=16, chunk=8)
    while not g.ready:
        g.step()

    class Hot:
        make_cache = model.make_cache

        def __call__(self, x, cache=None):
            return model(x, cache=cache) * 3.0

    g.model = Hot()
    g.start()
    assert _run(g)["ok"] is False

    class Broken:
        make_cache = model.make_cache

        def __call__(self, x, cache=None):
            raise MemoryError("oom")

    g.model = Broken()
    g.start()
    assert g.step() == {"ok": False, "error": "MemoryError('oom')"}


def test_corrupt_reference_file_is_rebuilt_not_fatal(model, tmp_path):
    f = tmp_path / "ref.npz"
    f.write_bytes(b"half a file")
    g = Guard(model, [[1, 2, 3, 4]], f, "k", total=16, chunk=8)
    assert not g.ready
    while not g.ready:
        g.step()
    assert g.math_ok and not list(tmp_path.glob("*.tmp*"))


def test_noise_numbers_are_kept_with_the_reference(model, tmp_path):
    prompts = [list(np.random.default_rng(s).integers(0, TINY["vocab_size"], 5)) for s in range(3)]
    g = Guard(model, prompts, tmp_path / "ref.npz", "k1", total=48, chunk=8)
    while not g.ready:
        g.step()
    assert set(g.noise) == {"kl_mean", "kl_p99", "top1", "ok"}
    again = Guard(model, prompts, tmp_path / "ref.npz", "k1", total=48, chunk=8)
    assert again.noise == g.noise  # loaded with the reference: a failed check says by how much
