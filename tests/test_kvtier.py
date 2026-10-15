import json

import pytest

from moestream import kvtier
from moestream.kvtier import Store, fingerprint

FP = "v1 test kv8"


def _pack(tmp_path, layout="{}"):
    p = tmp_path / "pack"
    p.mkdir(exist_ok=True)
    (p / "layout.json").write_text(layout)
    (p / "config.json").write_text("{}")
    return p


def _chain(root, name, files, manifest=None):
    """A chain directory written by hand: files {name: bytes}, manifest naming them all."""
    d = root / name
    d.mkdir(parents=True)
    for f, b in files.items():
        (d / f).write_bytes(b)
    m = manifest or {"turns": 1,
                     "segments": [{"a": 0, "b": 4, "file": "full_000.safetensors", "bytes": 3}],
                     "snapshots": [{"n": 4, "sha": "x", "turn": 1, "file": "sliding_4.safetensors", "bytes": 3}]}
    (d / "manifest.json").write_text(m if isinstance(m, str) else json.dumps(m))
    return d


GOOD = {"full_000.safetensors": b"abc", "sliding_4.safetensors": b"def"}


def test_fingerprint_changes_with_kv_bits_and_layout_and_format_version(tmp_path, monkeypatch):
    p = _pack(tmp_path)
    a = fingerprint(p, 8)
    assert fingerprint(p, 8) == a and fingerprint(p, 0) != a
    monkeypatch.setattr(kvtier, "FORMAT_VERSION", kvtier.FORMAT_VERSION + 1)
    assert fingerprint(p, 8) != a
    monkeypatch.undo()
    _pack(tmp_path, layout='{"x": 1}')
    assert fingerprint(p, 8) != a


def test_a_good_chain_is_indexed_and_leftover_tmp_and_unnamed_files_are_deleted(tmp_path):
    Store(tmp_path / "kv", FP)
    d = _chain(tmp_path / "kv", "c1", {**GOOD, "full_001.safetensors.tmp": b"torn", "sliding_2.safetensors": b"old"})
    s = Store(tmp_path / "kv", FP)
    assert list(s.chains) == ["c1"]
    assert sorted(f.name for f in d.iterdir()) == ["full_000.safetensors", "manifest.json", "sliding_4.safetensors"]


@pytest.mark.parametrize("files", [{"full_000.safetensors": b"abc"},                          # missing
                                   {"full_000.safetensors": b"abc", "sliding_4.safetensors": b"de"}])  # short
def test_a_manifest_naming_a_missing_or_short_file_drops_the_chain(tmp_path, files):
    Store(tmp_path / "kv", FP)
    d = _chain(tmp_path / "kv", "c1", files)
    assert Store(tmp_path / "kv", FP).chains == {} and not d.exists()


def test_a_corrupt_manifest_drops_the_chain(tmp_path):
    Store(tmp_path / "kv", FP)
    d = _chain(tmp_path / "kv", "c1", GOOD, manifest='{"segments": [')
    assert Store(tmp_path / "kv", FP).chains == {} and not d.exists()


def test_a_different_fingerprint_wipes_the_directory(tmp_path):
    Store(tmp_path / "kv", FP)
    _chain(tmp_path / "kv", "c1", GOOD)
    s = Store(tmp_path / "kv", "v1 test kv0")
    assert s.chains == {} and sorted(f.name for f in (tmp_path / "kv").iterdir()) == ["FORMAT"]


def test_a_non_empty_directory_that_is_not_a_kv_tier_is_left_alone(tmp_path):
    (tmp_path / "kv").mkdir()
    (tmp_path / "kv" / "notes.txt").write_text("mine")
    with pytest.raises(ValueError, match="FORMAT"):
        Store(tmp_path / "kv", FP)
    assert (tmp_path / "kv" / "notes.txt").read_text() == "mine"


import threading
import time

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache, RotatingKVCache

from laguna_mlx.cache import DequantKVCache


def _make(kv_bits):
    """Laguna's cache shape in miniature: one full-attention layer, two sliding (window 8)."""
    return [DequantKVCache(group_size=32, bits=kv_bits) if kv_bits else KVCache(),
            RotatingKVCache(max_size=8), RotatingKVCache(max_size=8)]


def _grow(cache, n, seed=0):
    r = np.random.default_rng(seed)
    for c in cache:
        k, v = (mx.array(r.standard_normal((1, 2, n, 32)).astype(np.float32)).astype(mx.bfloat16) for _ in "kv")
        c.update_and_fetch(k, v)
    mx.eval([c.state for c in cache])


def _parts(c):
    return [*c.q.keys, *c.q.values] if hasattr(c, "q") else [c.keys, c.values]


def _turns(store, kv_bits, lengths, base=0):
    cache, tokens = _make(kv_bits), []
    for i, n in enumerate(lengths):
        _grow(cache, n - len(tokens), seed=i)
        tokens = list(range(base, base + n))
        store.save(tokens, cache)
        store.flush()
    return cache, tokens


@pytest.mark.parametrize("kv_bits", [0, 8])
def test_three_turns_append_three_segments_equal_to_the_full_attention_kv(tmp_path, kv_bits):
    s = Store(tmp_path / "kv", FP)
    cache, _ = _turns(s, kv_bits, [10, 25, 40])
    (cid, m), = s.chains.items()
    assert [(g["a"], g["b"]) for g in m["segments"]] == [(0, 10), (10, 25), (25, 40)]
    segs = [mx.load(str(tmp_path / "kv" / cid / g["file"])) for g in m["segments"]]
    for j, want in enumerate(_parts(cache[0])):
        got = mx.concatenate([seg[f"0.{j}"] for seg in segs], axis=2)
        assert got.dtype == want.dtype and mx.array_equal(got, want[:, :, :40]).item()


def test_a_snapshot_that_does_not_extend_a_chain_starts_a_new_chain(tmp_path):
    s = Store(tmp_path / "kv", FP)
    _turns(s, 0, [10, 20])
    _turns(s, 0, [12], base=1)
    assert sorted(m["turns"] for m in s.chains.values()) == [1, 2]


def test_a_slow_writer_does_not_delay_save(tmp_path, monkeypatch):
    write = kvtier._write_tensors
    monkeypatch.setattr(kvtier, "_write_tensors", lambda *a: time.sleep(0.5) or write(*a))
    s = Store(tmp_path / "kv", FP)
    cache = _make(0)
    _grow(cache, 10)
    t = time.perf_counter()
    s.save(list(range(10)), cache)
    assert time.perf_counter() - t < 0.2
    s.flush()
    assert len(s.chains) == 1


def test_a_newer_snapshot_of_the_same_chain_replaces_a_queued_one(tmp_path, monkeypatch):
    go, write = threading.Event(), kvtier._write_tensors
    monkeypatch.setattr(kvtier, "_write_tensors", lambda *a: go.wait() and write(*a))
    s = Store(tmp_path / "kv", FP)
    caches = {n: _make(0) for n in (10, 20, 30)}
    for n, c in caches.items():
        _grow(c, n)
    s.save(list(range(10)), caches[10])       # taken by the writer, blocked
    time.sleep(0.1)
    s.save(list(range(20)), caches[20])       # queued
    s.save(list(range(30)), caches[30])       # supersedes the queued one
    s.save([7] * 30, caches[30])              # another conversation: kept
    assert sorted(j["n"] for j in s.pending.values()) == [30, 30]
    go.set()
    s.flush()
    assert sorted(m["snapshots"][-1]["n"] for m in s.chains.values()) == [30, 30]


def test_three_failed_writes_in_a_row_turn_writes_off(tmp_path, monkeypatch):
    def fail(*a):
        raise OSError("disk full")
    monkeypatch.setattr(kvtier, "_write_tensors", fail)
    s = Store(tmp_path / "kv", FP)
    for i in range(3):
        _turns(s, 0, [10], base=i)
        assert s.writing is (i < 2)
    assert s.chains == {} and sorted(f.name for f in (tmp_path / "kv").iterdir()) == ["FORMAT"]
    s.save(list(range(10)), _make(0))
    assert not s.pending
