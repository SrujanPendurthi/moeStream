import mlx.core as mx
import numpy as np

from moestream.load import load
from scripts.pack_experts import pack
from tests.tiny import TINY, quantized_model_dir


def _model(tmp_path):
    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    return load(tmp_path / "packed", slots=3)[0]


def _prefill(model, n=24):
    cache = model.make_cache()
    ids = np.random.default_rng(1).integers(0, TINY["vocab_size"], size=(1, n))
    logits = model(mx.array(ids), cache=cache)
    return cache, int(mx.argmax(logits[0, -1]))


def test_instrumented_decode_gives_the_same_tokens_and_a_split_that_adds_up(tmp_path):
    from tools.decode_prof import decode

    model = _model(tmp_path)
    cache, t = _prefill(model)
    plain = decode(model, cache, t, 6, model.args.layer_types, instrument=False)
    cache, t = _prefill(model)
    timed = decode(model, cache, t, 6, model.args.layer_types, instrument=True)
    assert timed["tokens"] == plain["tokens"] and len(plain["tokens"]) == 6
    parts = ("attn_full_ms", "attn_sliding_ms", "moe_ms", "other_ms")
    assert all(timed[k] >= 0 for k in parts)
    assert abs(sum(timed[k] for k in parts) - timed["ms_per_tok"]) < 1e-6
    assert {"sync_ms", "read_ms", "write_ms", "hit_rate"} <= set(timed)
    assert model.layers[1].__class__.__name__ != "Timed"          # wrappers removed afterwards
