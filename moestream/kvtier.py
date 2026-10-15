"""SSD tier for the prompt cache (docs/specs/2026-10-14-ssd-kv-tier-design.md).

Turn-end KV snapshots on disk, so a conversation pushed out of RAM, or a server restart, restores its
KV instead of processing the prompt again. One directory per chain (a conversation: each snapshot is
a prefix of the next): full-attention KV appended per turn as token-range segments, sliding-window
state whole per kept snapshot. Every file is written as <name>.tmp, fsynced and renamed; the chain's
manifest.json is replaced last, so a crash loses at most the last turn.
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

FORMAT_VERSION = 1


def log(msg):
    print(f"moeStream kv tier: {msg}", file=sys.stderr)


def fingerprint(pack_dir, kv_bits):
    """Entries are only valid for the same format, pack and KV precision."""
    pack_dir = Path(pack_dir)
    h = hashlib.sha256((pack_dir / "layout.json").read_bytes() + (pack_dir / "config.json").read_bytes())
    return f"v{FORMAT_VERSION} {h.hexdigest()[:16]} kv{kv_bits}"


def write_atomic(path, write_fn):
    """write_fn(f) fills <path>.tmp; then fsync and rename over path. Returns the size."""
    tmp = Path(f"{path}.tmp")
    with open(tmp, "wb") as f:
        write_fn(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path.stat().st_size


def _files(m):
    return {e["file"]: e["bytes"] for e in m["segments"] + m["snapshots"]}


def _load_chain(d):
    """The chain's manifest if every file it names is there at its size; deletes anything else."""
    try:
        m = json.loads((d / "manifest.json").read_text())
        files = _files(m)
        if any(not (d / f).is_file() or (d / f).stat().st_size != n for f, n in files.items()):
            raise ValueError("a named file is missing or the wrong size")
    except (OSError, ValueError, KeyError, TypeError) as e:
        log(f"dropping chain {d.name}: {e}")
        return None
    for f in d.iterdir():
        if f.name != "manifest.json" and f.name not in files:
            f.unlink()
    return m


class Store:
    def __init__(self, dir, fingerprint):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        fmt = self.dir / "FORMAT"
        if not fmt.exists():
            if any(self.dir.iterdir()):  # never wipe a directory this tier did not create
                raise ValueError(f"{self.dir} is not empty and has no FORMAT file: not a kv tier directory")
            write_atomic(fmt, lambda f: f.write(fingerprint.encode()))
        elif fmt.read_text() != fingerprint:
            log(f"format, pack or kv_bits changed ({fmt.read_text()} -> {fingerprint}): wiping {self.dir}")
            shutil.rmtree(self.dir)
            self.dir.mkdir()
            write_atomic(fmt, lambda f: f.write(fingerprint.encode()))
        self.chains = {}
        for d in sorted(self.dir.iterdir()):
            if d.name == "FORMAT":
                continue
            m = _load_chain(d) if d.is_dir() else None
            if m is None:
                shutil.rmtree(d) if d.is_dir() else d.unlink()
            else:
                self.chains[d.name] = m
