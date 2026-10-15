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
