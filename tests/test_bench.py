import time

import mlx.core as mx
import numpy as np

from moestream.load import load
from scripts.pack_experts import pack
from tests.tiny import TINY, quantized_model_dir
from tools.bench import decode_tok_s, equal_memory, long_ctx


def test_bench_helpers_run_on_tiny(tmp_path):
    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    m, _ = load(tmp_path / "packed", slots=4)
    rng = np.random.default_rng(0)
    prompts = [rng.integers(0, TINY["vocab_size"], size=6).tolist() for _ in range(2)]
    tps, outs, _ = decode_tok_s(m, prompts, max_tokens=5, eos=())
    assert tps > 0 and [len(o) for o in outs] == [5, 5]
    from laguna_mlx.dflash import generate_greedy
    assert outs == [generate_greedy(m, pr, 5, ()) for pr in prompts]
    r = long_ctx(m, rng.integers(0, TINY["vocab_size"], size=40).tolist(), gen=4, chunk=16)
    assert r["ctx_tokens"] == 40 and r["decode_ctx_tok_s"] > 0


class _SlowPrefill:
    """Fake model: a multi-token call sleeps, a one-token call is instant."""
    decode_calls = 0

    def make_cache(self):
        return []

    def __call__(self, ids, cache=None):
        if ids.shape[1] > 1:
            time.sleep(0.3)
        else:
            self.decode_calls += 1
        return mx.zeros((1, ids.shape[1], 4))


def test_decode_speed_and_stats_exclude_prefill():
    m = _SlowPrefill()
    tps, outs, delta = decode_tok_s(m, [[1, 2, 3], [4, 5]], max_tokens=5, eos=(),
                                    stats=lambda mm: {"hits": mm.decode_calls, "label": "x"})
    assert [len(o) for o in outs] == [5, 5]
    assert tps > 100                # 0.6 s of prefill would cap it near 13 tok/s
    assert delta == {"hits": 8}     # 4 decode steps per prompt; non-numeric keys skipped


def test_equal_memory_gives_both_loaders_the_same_slot_bytes():
    from moestream.budget import GB, KV_BYTES_PER_TOKEN
    budget, lab = equal_memory(20, 1 * GB, 39, 1769472, 256)
    assert budget == 20 * GB - 1 * GB - KV_BYTES_PER_TOKEN * 32768 - 1 * GB
    assert lab == budget // (39 * 1769472) and lab * 39 * 1769472 <= budget
    assert equal_memory(200, 1 * GB, 39, 1769472, 256)[1] == 256  # clamps to num_experts
