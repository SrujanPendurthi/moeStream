import numpy as np

from moestream.load import load
from scripts.pack_experts import pack
from tests.tiny import TINY, quantized_model_dir
from tools.record import record_ids


def test_record_ids_gives_per_token_routing_per_layer(tmp_path):
    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    m, _ = load(tmp_path / "packed", slots=4)
    ids = np.random.default_rng(0).integers(0, TINY["vocab_size"], size=25)
    rec = record_ids(m, ids, chunk=10)
    assert sorted(rec) == [1, 2, 3]
    for r in rec.values():
        assert r.shape == (25, TINY["num_experts_per_tok"]) and r.dtype == np.uint8
        assert (r < TINY["num_experts"]).all()
    again = record_ids(m, ids, chunk=10)
    assert all((again[l] == rec[l]).all() for l in rec)  # deterministic, trace hooks cleared
