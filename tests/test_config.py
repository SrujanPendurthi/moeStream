from pathlib import Path

import pytest

from moestream.config import EngineConfig

REPO = Path(__file__).resolve().parents[1]


def _write(tmp_path, text, name="engine.toml"):
    f = tmp_path / name
    f.write_text(text)
    return f


def test_shipped_file_reads_with_spec_defaults():
    c = EngineConfig.read(REPO / "moestream.toml")
    assert c.memory_gb == 14.0 and c.context_tokens == 131072 and c.kv_bits == 8
    assert c.eviction == "lfu" and c.thinking is True and c.port == 8081
    assert Path(c.pack_dir) == (REPO / "../lagunaLab/models/Laguna-XS-2.1-packed")
    assert Path(c.calibration) == REPO / "budget.json"


def test_omitted_keys_take_defaults(tmp_path):
    c = EngineConfig.read(_write(tmp_path, 'pack_dir = "p"\n'))
    assert c.memory_gb is None and c.half_life is None and c.calibration is None
    assert c.thinking is True and c.eviction == "lfu" and c.kv_bits == 8 and c.context_tokens == 131072


def test_relative_paths_resolve_against_the_file_not_the_cwd(tmp_path, monkeypatch):
    sub = tmp_path / "sub"
    sub.mkdir()
    f = _write(sub, 'pack_dir = "packed"\ncalibration = "cal.json"\n')
    monkeypatch.chdir(tmp_path)
    c = EngineConfig.read(f)
    assert c.pack_dir == str(sub / "packed") and c.calibration == str(sub / "cal.json")


def test_absolute_paths_are_kept(tmp_path):
    c = EngineConfig.read(_write(tmp_path, f'pack_dir = "{tmp_path}/x"\n'))
    assert c.pack_dir == f"{tmp_path}/x"


@pytest.mark.parametrize("line, key", [
    ('colour = "red"', "colour"),
    ("kv_bits = 6", "kv_bits"),
    ("kv_bits = 8.0", "kv_bits"),
    ('eviction = "lifo"', "eviction"),
    ('memory_gb = "20"', "memory_gb"),
    ("memory_gb = 0", "memory_gb"),
    ("context_tokens = 0", "context_tokens"),
    ("context_tokens = 1.5", "context_tokens"),
    ("half_life = -1", "half_life"),
    ('thinking = "yes"', "thinking"),
    ("port = 70000", "port"),
    ("kv_disk_gb = 0", "kv_disk_gb"),
    ('kv_tier = "on"', "kv_tier"),
])
def test_bad_values_raise_naming_the_key(tmp_path, line, key):
    with pytest.raises(ValueError, match=key):
        EngineConfig.read(_write(tmp_path, f'pack_dir = "p"\n{line}\n'))


def test_tuning_keys_default_and_resolve(tmp_path):
    c = EngineConfig.read(_write(tmp_path, 'pack_dir = "p"\n'))
    assert c.tune is True and c.trace_tokens == 65536
    assert Path(c.tuning_dir) == tmp_path / "tuning"
    shipped = EngineConfig.read(REPO / "moestream.toml")
    assert shipped.tune is True and Path(shipped.tuning_dir) == REPO / "tuning"


@pytest.mark.parametrize("line", ["tune = 1", "trace_tokens = 0", "tuning_dir = 3"])
def test_bad_tuning_values_raise(tmp_path, line):
    with pytest.raises(ValueError, match=line.split(" ")[0]):
        EngineConfig.read(_write(tmp_path, f'pack_dir = "p"\n{line}\n'))


def test_ssd_tier_defaults_and_kv_dir_resolves_against_the_file(tmp_path):
    c = EngineConfig.read(_write(tmp_path, 'pack_dir = "p"\n'))
    assert c.kv_tier is True and c.kv_disk_gb == 10.0 and c.kv_dir == str(tmp_path / "kvcache")
