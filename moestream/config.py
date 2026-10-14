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
    context_tokens: int = 32768
    kv_bits: int = 0
    eviction: str = "lfu"
    half_life: float | None = None
    calibration: str | None = None
    thinking: bool = True
    port: int = 8081

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
        for key in ("pack_dir", "calibration"):
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
        }
        for key, ok in rules.items():
            if not ok:
                raise ValueError(f"{key} = {getattr(self, key)!r} is not allowed "
                                 f"(kv_bits: {KV_BITS}, eviction: {EVICTION})")
