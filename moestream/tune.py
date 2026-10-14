"""Self-optimizing runtime: measure, try one change, keep it only if it measured better.

Runs on mlx-lm's generation thread: on_token() after each decode step, idle() when no request
is waiting. Replay trials run in up to 2 low-priority worker processes, so this module stays
free of MLX imports.
"""
import hashlib
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from moestream.budget import allocate
from moestream.store import POLICIES
from tools.calibrate import HALF_LIVES, miss_rate


class Ring:
    """The last `cap` routing rows of one layer, in one preallocated buffer of 2 x cap rows.
    StreamingMoE appends to it through io.trace."""

    def __init__(self, cap):
        self.cap, self.buf, self.n = cap, None, 0

    def append(self, rows):
        rows = rows[-self.cap :]
        if self.buf is None:
            self.buf = np.empty((2 * self.cap, rows.shape[1]), np.uint8)
        if self.n + len(rows) > len(self.buf):  # slide the last cap rows to the front
            self.buf[: self.cap] = self.buf[self.n - self.cap : self.n]
            self.n = self.cap
        self.buf[self.n : self.n + len(rows)] = rows
        self.n += len(rows)

    def rows(self):
        return self.buf[max(0, self.n - self.cap) : self.n].copy() if self.n else None


def replay_rate(rows, slots, policy, half_life, num_experts):
    """Token-weighted simulated miss rate over layers; rows {layer: (T, k) uint8}, slots {layer: n}."""
    n = sum(len(r) for r in rows.values())
    return sum(miss_rate(r, slots[l], half_life, num_experts, policy) * len(r) for l, r in rows.items()) / max(1, n)


def replay_trial(knob, snap):
    """Worker: best candidate for one replay knob and both simulated miss rates; candidate None if none is lower."""
    rows, slots, policy, hl, E = (snap[k] for k in ("rows", "slots", "policy", "half_life", "num_experts"))
    if knob == "policy":
        cands = [dict(policy=p) for p in POLICIES if p != policy]
    elif knob == "half_life":
        cands = [dict(half_life=h) for h in HALF_LIVES if h != hl]
    else:
        curves = {l: [miss_rate(r, g, hl, E, policy) for g in snap["grid"]] for l, r in rows.items()}
        cands = [dict(slots=allocate(curves, snap["grid"], sum(slots.values())))]
    cur = replay_rate(rows, slots, policy, hl, E)
    best, best_rate = None, cur
    for c in cands:
        r = replay_rate(rows, c.get("slots", slots), c.get("policy", policy), c.get("half_life", hl), E)
        if r < best_rate:
            best, best_rate = next(iter(c.values())), r
    return best, cur, best_rate


def ab_keep(pairs, n=8, need=7):
    """pairs: (current seconds, candidate seconds) per matched window pair. None until n pairs."""
    if len(pairs) < n:
        return None
    return sum(c < a for a, c in pairs[:n]) >= need


@dataclass
class Knob:
    name: str
    tier: str                         # "replay": judged by simulation; "live": by A/B timing
    get: Callable
    apply: Callable                   # between tokens; must not exceed memory_gb
    propose: Callable = None          # live knobs: a candidate value, or None
    changes_math: bool = False        # live knobs never do; replay ones may (slot split)
    ready: Callable = lambda: True    # False pauses the knob (e.g. slots shrunk under pressure)


def _plain(o):
    return o.item() if hasattr(o, "item") else str(o)


class Tuner:
    WINDOW = 64       # tokens per A/B window
    WARM = 64         # decode tokens after a break (prefill, idle) that only re-warm the cache: not measured
    AB_TIMEOUT = 600  # seconds without a completed pair before a live trial ends with no decision
    STATS = 256       # tokens per logged stats window
    MIN_GAIN = 0.01   # replay: candidate miss rate must be at least 1% lower

    def __init__(self, knobs, snapshot, dirpath, fingerprint, *, guard=None, pool=None, clock=time.perf_counter,
                 counts=lambda: (0, 0), trial=True, retry_tokens=50_000):
        self.knobs = {k.name: k for k in knobs}
        self.order = list(self.knobs)
        self.snapshot, self.guard, self.pool, self.clock, self.counts = snapshot, guard, pool, clock, counts
        self.dir = Path(dirpath)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fp, self.trial_on, self.retry_tokens = fingerprint, trial, retry_tokens
        self.trial, self.disabled, self.rejected, self.next_i = None, set(), {}, 0
        self.tokens, self.last_tok, self.quiet, self.since = 0, None, False, 0
        self.ab_s = self.ab_n = self.st_s = self.st_n = 0
        self.last_counts, self.window_hit = counts(), 0.0

    # generation thread, per decode token
    def on_token(self):
        if self.quiet:  # the guard's own forward passes are not traffic
            return
        now = self.clock()
        last, self.last_tok = self.last_tok, now
        self.tokens += 1
        if last is None:
            return
        dt = now - last
        self.since += 1
        if self.since > self.WARM:
            self.ab_s += dt; self.ab_n += 1
            if self.ab_n == self.WINDOW:
                self._window(self.ab_s)
                self.ab_s = self.ab_n = 0
        self.st_s += dt; self.st_n += 1
        if self.st_n == self.STATS:
            h, m = self.counts()
            dh, dm = h - self.last_counts[0], m - self.last_counts[1]
            self.last_counts, self.window_hit = (h, m), dh / max(1, dh + dm)
            self.log({"tok_s": self.st_n / max(self.st_s, 1e-9), "hit_rate": self.window_hit, "tokens": self.tokens})
            self.st_s = self.st_n = 0

    def interrupt(self):
        """Prefill or idle: the token stream broke. Drop the partial window and any half pair."""
        self.last_tok = None
        self.ab_s = self.ab_n = self.since = 0
        t = self.trial
        if t and t["phase"] == "ab":
            t["pending"] = None
            first = "old" if len(t["pairs"]) % 2 == 0 else "new"  # A B, then B A: neither arm always runs first
            if t["arm"] != first:
                self._arm(t, first)

    def cancel(self, name):
        """End an in-flight trial of `name` with no decision. Returns the pre-trial value if a candidate is live."""
        t = self.trial
        if t is None or t["knob"] != name:
            return None
        self.trial = None
        self.log({"knob": name, "cancelled": t["phase"]})
        return t["old"] if t["phase"] == "guard" else None

    # generation thread, no request waiting; one bounded slice
    def idle(self):
        self.interrupt()
        g = self.guard
        if g is not None and not g.ready:
            self._guard_step()
            if g.ready:
                self.load_state()
            return
        t = self.trial
        if t is None:
            self._start_next()
        elif t["phase"] == "replay" and t["future"].done():
            self._replay_done(t)
        elif t["phase"] == "ab" and self.clock() - t["last_pair"] > self.AB_TIMEOUT:
            self.trial = None  # traffic too short for pairs: no decision, retried on its next turn
            self._set(t["knob"], t["old"], disable=False)
            self.log({"knob": t["knob"], "cancelled": "ab timeout", "pairs": len(t["pairs"])})
        elif t["phase"] == "guard":
            v = self._guard_step()
            if v is not None:
                self._finish(v["ok"], **v)

    def _guard_step(self):
        self.quiet = True
        try:
            return self.guard.step()
        finally:
            self.quiet = False

    def _start_next(self):
        if not self.trial_on:
            return
        for _ in range(len(self.order)):
            name = self.order[self.next_i]
            self.next_i = (self.next_i + 1) % len(self.order)
            k = self.knobs[name]
            if name in self.disabled or not k.ready() or not self._eligible(name):
                continue
            if k.changes_math and not (self.guard and self.guard.math_ok):
                continue
            old = k.get()
            if k.tier == "replay":
                snap = self.snapshot()
                if snap is None:
                    continue
                try:
                    future = self.pool.submit(replay_trial, name, snap)
                except Exception as e:  # the worker pool died (e.g. a worker was killed): replay knobs stop
                    self.disabled |= {n for n, k in self.knobs.items() if k.tier == "replay"}
                    self.log({"knob": name, "error": repr(e)})
                    continue
                self.trial = dict(knob=name, old=old, new=None, phase="replay", future=future)
            else:
                new = k.propose()
                if new is None or new == old:
                    continue
                self.trial = dict(knob=name, old=old, new=new, phase="ab", arm="old", pending=None, pairs=[],
                                  last_pair=self.clock())
            return

    def _eligible(self, name):
        r = self.rejected.get(name)
        return r is None or self.tokens - r[0] >= self.retry_tokens or abs(self.window_hit - r[1]) > 0.05

    def _window(self, secs):
        t = self.trial
        if t is None or t["phase"] != "ab":
            return
        if t["pending"] is None:   # first arm of the pair just ran; run the other one
            t["pending"] = (t["arm"], secs)
            self._arm(t, "new" if t["arm"] == "old" else "old")
            return
        first, s0 = t["pending"]
        t["pending"] = None
        t["pairs"].append((s0, secs) if first == "old" else (secs, s0))  # always (current, candidate)
        t["last_pair"] = self.clock()
        keep = ab_keep(t["pairs"])  # the next pair starts on the arm this one ended on
        if keep is not None:
            self._finish(keep, pairs=t["pairs"])

    def _arm(self, t, arm):
        if self._switch(t[arm]):
            t["arm"] = arm

    def _replay_done(self, t):
        e = t["future"].exception()
        if e is not None:
            self._finish(False, error=repr(e))
            return
        value, cur, cand = t["future"].result()
        t["new"], nums = value, {"miss_rate": cur, "candidate_miss_rate": cand}
        if value is None or cand > cur * (1 - self.MIN_GAIN):
            self._finish(False, **nums)
        elif self.knobs[t["knob"]].changes_math:
            if self._switch(value):
                t["phase"] = "guard"
                self.guard.start()
        else:
            self._finish(True, **nums)

    def _set(self, name, value, disable=True):
        try:
            self.knobs[name].apply(value)
            return True
        except Exception as e:
            self.log({"knob": name, "value": value, "error": repr(e)})
            if disable:
                self.disabled.add(name)
            return False

    def _switch(self, value):
        """Set the trial's knob; on failure revert it, disable it and end the trial."""
        t = self.trial
        if self._set(t["knob"], value):
            return True
        self._set(t["knob"], t["old"], disable=False)
        self.trial = None
        return False

    def _finish(self, keep, **nums):
        t, self.trial = self.trial, None
        name = t["knob"]
        if keep and not self._set(name, t["new"]):
            keep = False
        if not keep:
            self._set(name, t["old"], disable=False)
            self.rejected[name] = (self.tokens, self.window_hit)
        else:
            self.rejected.pop(name, None)
        self.log({"knob": name, "from": t["old"], "to": t["new"], "kept": keep, **nums})
        self.save()

    def log(self, record):
        with open(self.dir / "log.jsonl", "a") as f:
            f.write(json.dumps({"time": time.time(), **record}, default=_plain) + "\n")

    def save(self):
        values = {n: k.get() for n, k in self.knobs.items()}
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps({"fingerprint": self.fp, "values": values}, default=_plain))
        os.replace(tmp, self.dir / "state.json")  # a crash mid-write never leaves half a state file

    def load_state(self):
        try:
            s = json.loads((self.dir / "state.json").read_text())
        except FileNotFoundError:
            return
        except Exception as e:
            warnings.warn(f"tuning state unreadable ({e!r}); starting from defaults")
            return
        if s.get("fingerprint") != self.fp:
            warnings.warn("tuning state is for a different pack or config; starting from defaults")
            return
        for name, v in s.get("values", {}).items():
            if name in self.knobs and not self._set(name, v, disable=False):
                warnings.warn(f"saved {name} dropped: it no longer applies")


def fingerprint(cfg):
    layout = (Path(cfg.pack_dir) / "layout.json").read_bytes()
    return hashlib.sha256(layout + json.dumps([cfg.memory_gb, cfg.kv_bits, cfg.context_tokens]).encode()).hexdigest()[:16]


def start(model, tok, cfg, prompts=None, probe_tokens=2048):
    """Wire tuning into a loaded engine: knobs, routing traces, quality guard, pressure guard, saved state."""
    from moestream.guard import Guard
    from moestream.knobs import system_knobs
    from moestream.layer import StreamingMoE, stream_stats
    from moestream.pressure import Pressure
    from tools.calibrate import GRID

    moe = [(i, l.mlp) for i, l in enumerate(model.layers) if isinstance(l.mlp, StreamingMoE)]
    E = moe[0][1].io.layout["num_experts"]
    knobs = system_knobs(model, sum(m.io.store.slots for _, m in moe))
    i = next(j for j, k in enumerate(knobs) if k.name == "slots")
    pressure = Pressure(knobs[i], min(16, E))
    knobs[i] = pressure.guarded()

    def snapshot():
        rows = {i: m.io.trace.rows() for i, m in moe}
        if any(r is None or len(r) < 1024 for r in rows.values()):
            return None
        st = moe[0][1].io.store
        return dict(rows=rows, slots={i: m.io.store.slots for i, m in moe}, policy=st.policy,
                    half_life=st.half_life, num_experts=E, grid=[g for g in GRID if g <= E])

    def counts():
        s = stream_stats(model)
        return s["hits"], s["misses"]

    d, guard, pool = Path(cfg.tuning_dir), None, None
    if cfg.tune:
        for _, m in moe:
            m.io.trace = Ring(cfg.trace_tokens)
        if prompts is None:
            from scripts.check_parity import PROMPTS
            prompts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
                       for p in PROMPTS]
        layout = (Path(cfg.pack_dir) / "layout.json").read_bytes()
        key = hashlib.sha256(layout + json.dumps([cfg.kv_bits, prompts, probe_tokens]).encode()).hexdigest()[:16]
        guard = Guard(model, prompts, d / "reference.npz", key, total=probe_tokens)
        pool = ProcessPoolExecutor(2, initializer=os.nice, initargs=(10,))
    tuner = Tuner(knobs, snapshot, d, fingerprint(cfg), guard=guard, pool=pool, counts=counts, trial=cfg.tune)
    pressure.log, pressure.cancel = tuner.log, lambda: tuner.cancel("slots")
    if guard is None or guard.ready:
        tuner.load_state()  # else it waits for the untuned reference to be built (Tuner.idle)

    def hook(decode):
        try:
            if decode:
                tuner.on_token()
                pressure.on_token()
            else:
                tuner.interrupt()
        except Exception as e:  # tuning never stops serving
            moe[-1][1].io.hook = None
            warnings.warn(f"tuning off: {e!r}")

    moe[-1][1].io.hook = hook
    return tuner
