import numpy as np

from moestream.load import load
from scripts.pack_experts import pack
from tests.tiny import TINY, quantized_model_dir
from tools.bench import decode_tok_s, long_ctx


def test_bench_helpers_run_on_tiny(tmp_path):
    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    m, _ = load(tmp_path / "packed", slots=4)
    rng = np.random.default_rng(0)
    prompts = [rng.integers(0, TINY["vocab_size"], size=6).tolist() for _ in range(2)]
    tps, outs = decode_tok_s(m, prompts, max_tokens=5, eos=())
    assert tps > 0 and [len(o) for o in outs] == [5, 5]
    r = long_ctx(m, rng.integers(0, TINY["vocab_size"], size=40).tolist(), gen=4, chunk=16)
    assert r["ctx_tokens"] == 40 and r["decode_ctx_tok_s"] > 0
