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
import struct
import sys
import threading
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import RotatingKVCache

FORMAT_VERSION = 1
SLIDING_EVERY = 8          # besides each chain's latest, keep every 8th turn's sliding state
MAX_WRITE_FAILURES = 3     # in a row, then writes stay off until restart
ST_DTYPE = {mx.bfloat16: "BF16", mx.float16: "F16", mx.float32: "F32", mx.uint32: "U32"}


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


def _digests(tokens, lengths):
    """SHA-256 of tokens[:n] for every n in lengths, in one pass over the prompt."""
    b, h, at, out = np.asarray(tokens, dtype=np.uint32).tobytes(), hashlib.sha256(), 0, {}
    for n in sorted(lengths):
        h.update(b[at * 4 : n * 4])
        out[n], at = h.hexdigest(), n
        h = h.copy()
    return out


def _parts(c):
    """A full-attention layer's arrays [B, H, capacity, X]; the first `offset` tokens are valid.
    lagunaLab's DequantKVCache keeps mlx-lm's QuantizedKVCache (packed, scales, biases) in `q`."""
    return [*c.q.keys, *c.q.values] if hasattr(c, "q") else [c.keys, c.values]


def _host(a):
    """(safetensors dtype, numpy view sharing a's buffer). The new view array also keeps the buffer
    out of MLX's in-place donation, so the cache can move on while the writer reads it."""
    return ST_DTYPE[a.dtype], a.view(mx.uint16 if a.dtype == mx.bfloat16 else a.dtype)


def _snapshot(tokens, cache):
    """On the caller's thread: evaluated numpy views of the cache, for the writer thread."""
    full = [(i, c) for i, c in enumerate(cache) if not isinstance(c, RotatingKVCache)]
    if not full or full[0][1].offset == 0:
        return None
    n = full[0][1].offset
    if n > len(tokens):
        log(f"cache holds {n} tokens but the key has {len(tokens)}: not saved")
        return None
    job = dict(tokens=list(tokens[:n]), n=n, meta={},
               full={f"{i}.{j}": _host(a) for i, c in full for j, a in enumerate(_parts(c))}, sliding={})
    for i, c in enumerate(cache):
        if isinstance(c, RotatingKVCache):
            job["sliding"].update({f"{i}.0": _host(c.keys), f"{i}.1": _host(c.values)})
            job["meta"][str(i)] = json.dumps(c.meta_state)
    mx.eval([a for _, a in job["full"].values()] + [a for _, a in job["sliding"].values()])
    for part in (job["full"], job["sliding"]):
        for k, (st, a) in part.items():
            part[k] = (st, np.array(a, copy=False))
    return job


def _write_tensors(path, tensors, meta=None):
    """safetensors, written one array at a time: extra memory is one array, not the file."""
    head, off = {}, 0
    for k, (st, a) in tensors.items():
        head[k] = {"dtype": st, "shape": list(a.shape), "data_offsets": [off, off + a.nbytes]}
        off += a.nbytes
    if meta:
        head["__metadata__"] = meta
    h = json.dumps(head).encode()
    h += b" " * (-len(h) % 8)

    def body(f):
        f.write(struct.pack("<Q", len(h)) + h)
        for _, a in tensors.values():
            f.write(np.ascontiguousarray(a).data)
    return write_atomic(path, body)


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
        self.pending, self.busy, self.cv = {}, False, threading.Condition()
        self.writing, self.failures, self._seq = True, 0, 0
        threading.Thread(target=self._writer, daemon=True, name="kvtier-writer").start()

    def save(self, tokens, cache):
        """Queue a turn-end snapshot and return at once. A queued, older snapshot of the same
        conversation is dropped: writes never fall behind or block a request."""
        if not self.writing:
            return
        job = _snapshot(tokens, cache)
        if job is None:
            return
        with self.cv:
            for k, j in list(self.pending.items()):
                if j["n"] < job["n"] and job["tokens"][: j["n"]] == j["tokens"]:
                    del self.pending[k]
            self._seq += 1
            self.pending[self._seq] = job
            self.cv.notify_all()

    def flush(self):
        with self.cv:
            self.cv.wait_for(lambda: not self.pending and not self.busy)

    def _writer(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.pending)
                job = self.pending.pop(next(iter(self.pending)))
                self.busy = True
            try:
                self._write(job)
                self.failures = 0
            except Exception as e:
                self.failures += 1
                log(f"snapshot write failed ({e!r}), {self.failures} in a row")
                if self.failures >= MAX_WRITE_FAILURES:
                    self.writing = False
                    log("writes off until restart")
            finally:
                with self.cv:
                    self.busy = False
                    self.cv.notify_all()

    def _chain_for(self, tokens, n):
        """The chain whose latest snapshot is the longest prefix of tokens, or (None, None)."""
        with self.cv:
            latest = {cid: m["snapshots"][-1] for cid, m in self.chains.items()}
        ok = [(s["n"], cid) for cid, s in latest.items() if s["n"] <= n]
        d = _digests(tokens, {k for k, _ in ok})
        k, cid = max(((k, cid) for k, cid in ok if d[k] == latest[cid]["sha"]), default=(0, None))
        return cid, (self.chains[cid] if cid else None)

    def _write(self, job):
        tokens, n = job["tokens"], job["n"]
        cid, m = self._chain_for(tokens, n)
        if m and m["snapshots"][-1]["n"] == n:
            return  # already on disk
        new = m is None
        if new:
            cid, m = os.urandom(6).hex(), {"turns": 0, "segments": [], "snapshots": []}
        d, a = self.dir / cid, (m["segments"][-1]["b"] if m["segments"] else 0)
        seg, sl, written = f"full_{len(m['segments']):03d}.safetensors", f"sliding_{n}.safetensors", []
        d.mkdir(exist_ok=True)
        try:
            nb = _write_tensors(d / seg, {k: (st, x[:, :, a:n]) for k, (st, x) in job["full"].items()})
            written.append(seg)
            ns = _write_tensors(d / sl, job["sliding"], job["meta"])
            written.append(sl)
            turn = m["turns"] + 1
            m2 = {"turns": turn,
                  "segments": m["segments"] + [{"a": a, "b": n, "file": seg, "bytes": nb}],
                  "snapshots": [s for s in m["snapshots"] if s["turn"] % SLIDING_EVERY == 0]
                  + [{"n": n, "sha": _digests(tokens, [n])[n], "turn": turn, "file": sl, "bytes": ns}]}
            write_atomic(d / "manifest.json", lambda f: f.write(json.dumps(m2).encode()))
        except BaseException:
            shutil.rmtree(d, ignore_errors=True) if new else [(d / f).unlink(missing_ok=True) for f in written]
            raise
        for f in set(_files(m)) - set(_files(m2)):
            (d / f).unlink(missing_ok=True)
        with self.cv:
            self.chains[cid] = m2
