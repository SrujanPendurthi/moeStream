import json
import os

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


def _store_with(tmp_path, monkeypatch, lengths, base=0, s=None):
    monkeypatch.setattr(kvtier, "MIN_RESTORE_TOKENS", 4)
    s = s or Store(tmp_path / "kv", FP)
    _turns(s, 0, lengths, base=base)
    return s


def test_lookup_finds_the_longest_snapshot_that_prefixes_the_prompt(tmp_path, monkeypatch):
    s = _store_with(tmp_path, monkeypatch, [10, 20, 30])
    s = _store_with(tmp_path, monkeypatch, [40], base=1, s=s)          # another conversation
    (cid, m), = [(c, m) for c, m in s.chains.items() if m["turns"] == 3]
    m["snapshots"].insert(0, {**m["snapshots"][0], "n": 20, "sha": kvtier._digests(range(20), [20])[20]})
    assert s.lookup(list(range(35)), ram_prefix=0, context_tokens=1000) == (30, cid)
    assert s.lookup(list(range(25)), ram_prefix=0, context_tokens=1000) == (20, cid)
    assert s.lookup(list(range(30)), ram_prefix=0, context_tokens=1000) == (20, cid)  # one token must be left to process
    assert s.lookup([9] * 35, ram_prefix=0, context_tokens=1000) is None


def test_lookup_skips_restores_too_small_to_pay_and_entries_over_the_context(tmp_path, monkeypatch):
    s = _store_with(tmp_path, monkeypatch, [10, 30])
    tokens = list(range(40))
    assert s.lookup(tokens, ram_prefix=27, context_tokens=1000) is None   # 30 - 27 < 4
    assert s.lookup(tokens, ram_prefix=26, context_tokens=1000)[0] == 30
    assert s.lookup(tokens, ram_prefix=0, context_tokens=29)[0] == 10     # 30 is over the context


def test_a_branched_history_matches_only_its_common_snapshot(tmp_path, monkeypatch):
    s = _store_with(tmp_path, monkeypatch, [10, 16, 30])                 # 10 is no longer kept
    (cid,) = s.chains
    assert s.lookup(list(range(20)) + [99] * 20, ram_prefix=0, context_tokens=1000) == (16, cid)
    assert s.lookup(list(range(13)) + [99] * 20, ram_prefix=0, context_tokens=1000) is None


from moestream.load import load
from scripts.pack_experts import pack
from tests.tiny import TINY, quantized_model_dir


@pytest.fixture(params=[0, 8], ids=["kv0", "kv8"])
def tiny(request, tmp_path, monkeypatch):
    if request.param:
        monkeypatch.setitem(TINY, "head_dim", 32)  # 16 is below the smallest quantization group
    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    m, _ = load(tmp_path / "packed", slots=4, kv_bits=request.param)
    return m


IDS = mx.array(np.random.default_rng(5).integers(0, TINY["vocab_size"], size=(1, 48)))
TOK = IDS[0].tolist()


def _prefill(model, cuts):
    """The cache after feeding IDS in chunks ending at each cut (turn ends)."""
    cache, at = model.make_cache(), 0
    for c in cuts:
        model(IDS[:, at:c], cache=cache)
        at = c
    mx.eval([c.state for c in cache])
    return cache


def _next(model, cache, at):
    """Logits for the next three tokens, decoded one at a time."""
    return mx.concatenate([model(IDS[:, i : i + 1], cache=cache) for i in range(at, at + 3)], axis=1)


def test_save_then_restore_gives_bit_identical_logits(tiny, tmp_path):
    s = Store(tmp_path / "kv", FP)
    cache = _prefill(tiny, [30])            # window 8: the rotating buffers have wrapped
    s.save(TOK[:30], cache)
    s.flush()
    (cid,) = s.chains
    fresh = tiny.make_cache()
    assert s.restore(cid, 30, fresh)
    assert mx.array_equal(_next(tiny, fresh, 30), _next(tiny, cache, 30)).item()


def test_restoring_turn_2_of_a_3_turn_chain_equals_the_cache_at_turn_2(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(kvtier, "SLIDING_EVERY", 2)
    s = Store(tmp_path / "kv", FP)
    for cuts in ([10], [10, 20], [10, 20, 30]):
        s.save(TOK[: cuts[-1]], _prefill(tiny, cuts))
        s.flush()
    (cid, m), = s.chains.items()
    assert [x["n"] for x in m["snapshots"]] == [20, 30]
    fresh = tiny.make_cache()
    assert s.restore(cid, 20, fresh)
    assert mx.array_equal(_next(tiny, fresh, 20), _next(tiny, _prefill(tiny, [10, 20]), 20)).item()


def test_restore_uses_one_host_buffer_the_size_of_the_largest_array(tiny, tmp_path):
    sizes = []
    s = Store(tmp_path / "kv", FP, alloc=lambda n: sizes.append(n) or bytearray(n))
    cache = _prefill(tiny, [12, 30])
    s.save(TOK[:30], cache)
    s.flush()
    largest = max([a[:, :, :30].nbytes for a in _parts(cache[0])]            # one segment, [0, 30)
                  + [c.keys.nbytes for c in cache[1:]])                         # sliding, whole
    assert s.restore(next(iter(s.chains)), 30, tiny.make_cache())
    assert sizes == [largest]


def test_a_failed_read_drops_the_chain_and_returns_false(tiny, tmp_path):
    s = Store(tmp_path / "kv", FP)
    s.save(TOK[:30], _prefill(tiny, [30]))
    s.flush()
    (cid, m), = s.chains.items()
    (tmp_path / "kv" / cid / m["segments"][0]["file"]).write_bytes(b"torn")
    assert s.restore(cid, 30, tiny.make_cache()) is False
    assert s.chains == {} and not (tmp_path / "kv" / cid).exists()


def test_only_the_latest_two_and_every_8th_sliding_snapshot_survive(tmp_path):
    s = Store(tmp_path / "kv", FP)
    _turns(s, 0, [5 * (i + 1) for i in range(19)])
    (cid, m), = s.chains.items()
    assert [x["turn"] for x in m["snapshots"]] == [8, 16, 18, 19]
    assert sorted(f.name for f in (tmp_path / "kv" / cid).glob("sliding_*")) == \
        ["sliding_40.safetensors", "sliding_80.safetensors", "sliding_90.safetensors", "sliding_95.safetensors"]


def test_a_dead_end_reply_does_not_cost_the_turn_it_branched_from(tmp_path, monkeypatch):
    """Regenerate or an aborted request: a reply nobody continues extends the chain; the next prompt
    extends the turn before it, whose sliding state must still be on disk, even mid-restore."""
    s = _store_with(tmp_path, monkeypatch, [10, 20])
    nxt = list(range(20)) + [77] * 10                       # the real next prompt, after turn 20
    hit = s.lookup(nxt, ram_prefix=0, context_tokens=1000)
    cache = _make(0)
    _grow(cache, 25, seed=9)
    s.save(list(range(20)) + [55] * 5, cache)               # the dead end lands between lookup and restore
    s.flush()
    assert hit[0] == 20 and s.restore(hit[1], 20, _make(0))
    assert s.lookup(nxt, ram_prefix=0, context_tokens=1000) == hit


def test_writing_past_the_cap_deletes_whole_chains_least_recently_used_first(tmp_path):
    s = Store(tmp_path / "kv", FP)
    chain = {}
    for base, used in ((0, 300), (1, 100), (2, 200)):         # chain 1 is the least recently used
        _turns(s, 0, [10, 20], base=base)
        sha = kvtier._digests(range(base, base + 20), [20])[20]
        (chain[base],) = [c for c, m in s.chains.items() if m["snapshots"][-1]["sha"] == sha]
        os.utime(tmp_path / "kv" / chain[base] / "manifest.json", (used, used))
    s.cap = 3 * sum(kvtier._files(s.chains[chain[0]]).values())
    _turns(s, 0, [10, 20], base=3)
    assert sorted(s.chains) == sorted([chain[0], chain[2], *set(s.chains) - set(chain.values())])
    assert len(s.chains) == 3 and not (tmp_path / "kv" / chain[1]).exists()


def test_a_snapshot_superseded_between_lookup_and_restore_keeps_the_chain(tmp_path, monkeypatch):
    s = _store_with(tmp_path, monkeypatch, [10])
    n, cid = s.lookup(list(range(30)), ram_prefix=0, context_tokens=1000)
    for k in (20, 30):                                 # the writer moves the chain on twice: sliding_10 goes
        cache = _make(0)
        _grow(cache, k, seed=1)
        s.save(list(range(k)), cache)
        s.flush()
    assert s.restore(cid, n, _make(0)) is False
    assert cid in s.chains and s.chains[cid]["snapshots"][-1]["n"] == 30


def test_a_crash_while_creating_format_leaves_a_directory_the_tier_still_owns(tmp_path):
    (tmp_path / "kv").mkdir()
    (tmp_path / "kv" / "FORMAT.tmp").write_text("v1 half")  # kill -9 before the rename
    assert Store(tmp_path / "kv", FP).chains == {}
    assert (tmp_path / "kv" / "FORMAT").read_text() == FP


def test_a_crash_mid_wipe_keeps_the_old_format_so_the_next_start_wipes_again(tmp_path, monkeypatch):
    Store(tmp_path / "kv", FP)
    _chain(tmp_path / "kv", "c1", GOOD)
    real = kvtier.write_atomic
    monkeypatch.setattr(kvtier, "write_atomic", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        Store(tmp_path / "kv", "v1 test kv0")
    assert (tmp_path / "kv" / "FORMAT").read_text() == FP
    monkeypatch.setattr(kvtier, "write_atomic", real)
    s = Store(tmp_path / "kv", "v1 test kv0")
    assert s.chains == {} and sorted(f.name for f in (tmp_path / "kv").iterdir()) == ["FORMAT"]


def test_the_writer_lets_go_of_a_written_snapshot(tmp_path):
    """The snapshot shares the live cache's buffers; once written, nothing in the tier may keep them
    (it kept the last turn's whole KV, ~3.4 GB at bf16 81k, until the next save)."""
    import gc

    s = Store(tmp_path / "kv", FP)
    cache = _make(0)
    _grow(cache, 4096)                                  # 2 x 1 x 2 x 4096 x 32 bf16: 1 MB per layer
    mx.eval([c.state for c in cache])
    s.save(list(range(4096)), cache)
    s.flush()
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    del cache
    gc.collect()
    mx.clear_cache()
    assert before - mx.get_active_memory() > 500_000    # the cache's buffers were freed
