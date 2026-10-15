"""moestream.toml -> EngineConfig: every engine setting in one flat file.

Relative paths resolve against the file's directory, so the server can start from any cwd.
Omitting memory_gb means "probe free memory"; omitting calibration means a uniform slot split.
"""
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

EVICTION = ("lfu", "lru", "fifo")
KV_BITS = (0, 4, 8)


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


@dataclass
class EngineConfig:
    pack_dir: str
    memory_gb: float | None = None
    context_tokens: int = 131072
    kv_bits: int = 8
    eviction: str = "lfu"
    half_life: float | None = None
    calibration: str | None = None
    thinking: bool = True
    port: int = 8081
    tune: bool = False
    tuning_dir: str = "tuning"
    trace_tokens: int = 65536
    kv_tier: bool = True
    kv_dir: str = "kvcache"
    kv_disk_gb: float = 10.0
    mlx_cache_gb: float = 1.0
    lookahead: bool = False
    kv_sessions: int = 1
    prompt_cache_entries: int = 6
    release_evicted_caches: bool = False

    @classmethod
    def read(cls, path):
        path = Path(path)
        raw = tomllib.loads(path.read_text())
        unknown = sorted(set(raw) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"{path}: unknown keys {unknown}")
        if "pack_dir" not in raw:
            raise ValueError(f"{path}: pack_dir is required")
        c = cls(**raw)
        c._check()
        for key in ("pack_dir", "calibration", "tuning_dir", "kv_dir"):
            v = getattr(c, key)
            if v is not None and not Path(v).is_absolute():
                setattr(c, key, str(path.parent / v))
        return c

    def _check(self):
        rules = {
            "pack_dir": isinstance(self.pack_dir, str),
            "memory_gb": self.memory_gb is None or (_num(self.memory_gb) and self.memory_gb > 0),
            "context_tokens": _int(self.context_tokens) and self.context_tokens > 0,
            "kv_bits": _int(self.kv_bits) and self.kv_bits in KV_BITS,
            "eviction": self.eviction in EVICTION,
            "half_life": self.half_life is None or (_num(self.half_life) and self.half_life > 0),
            "calibration": self.calibration is None or isinstance(self.calibration, str),
            "thinking": isinstance(self.thinking, bool),
            "port": _int(self.port) and 0 < self.port < 65536,
            "tune": isinstance(self.tune, bool),
            "tuning_dir": isinstance(self.tuning_dir, str),
            "trace_tokens": _int(self.trace_tokens) and self.trace_tokens > 0,
            "kv_tier": isinstance(self.kv_tier, bool),
            "kv_dir": isinstance(self.kv_dir, str),
            "kv_disk_gb": _num(self.kv_disk_gb) and self.kv_disk_gb > 0,
            "mlx_cache_gb": _num(self.mlx_cache_gb) and self.mlx_cache_gb >= 0,
            "lookahead": isinstance(self.lookahead, bool),
            "kv_sessions": _int(self.kv_sessions) and self.kv_sessions >= 1,
            "prompt_cache_entries": _int(self.prompt_cache_entries) and self.prompt_cache_entries >= 1,
            "release_evicted_caches": isinstance(self.release_evicted_caches, bool),
        }
        for key, ok in rules.items():
            if not ok:
                raise ValueError(f"{key} = {getattr(self, key)!r} is not allowed "
                                 f"(kv_bits: {KV_BITS}, eviction: {EVICTION})")
