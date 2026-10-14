"""Self-optimizing runtime: measure, try one change, keep it only if it measured better.

Runs on mlx-lm's generation thread: on_token() after each decode step, idle() when no request
is waiting. Replay trials run in up to 2 low-priority worker processes, so this module stays
free of MLX imports.
"""
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from moestream.budget import allocate
from moestream.store import POLICIES
from tools.calibrate import HALF_LIVES, miss_rate


class Ring:
    """The last `cap` routing rows of one layer; StreamingMoE appends to it through io.trace."""

    def __init__(self, cap):
        self.cap, self.parts, self.n = cap, [], 0

    def append(self, rows):
        self.parts.append(rows)
        self.n += len(rows)
        if self.n > 2 * self.cap:  # ponytail: holds up to 2x cap rows between compactions
            self.parts = [self.rows()]
            self.n = len(self.parts[0])

    def rows(self):
        return np.concatenate(self.parts)[-self.cap :] if self.parts else None


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
        self.tokens, self.last_tok, self.quiet = 0, None, False
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
        self.ab_s = self.ab_n = 0
        t = self.trial
        if t and t["phase"] == "ab" and t["pending"] is not None:
            t["pending"] = None
            self._switch(t["old"])

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
                self.trial = dict(knob=name, old=old, new=None, phase="replay",
                                  future=self.pool.submit(replay_trial, name, snap))
            else:
                new = k.propose()
                if new is None or new == old:
                    continue
                self.trial = dict(knob=name, old=old, new=new, phase="ab", pending=None, pairs=[])
            return

    def _eligible(self, name):
        r = self.rejected.get(name)
        return r is None or self.tokens - r[0] >= self.retry_tokens or abs(self.window_hit - r[1]) > 0.05

    def _window(self, secs):
        t = self.trial
        if t is None or t["phase"] != "ab":
            return
        if t["pending"] is None:   # the current arm just ran; run the candidate next
            t["pending"] = secs
            self._switch(t["new"])
            return
        t["pairs"].append((t["pending"], secs))
        t["pending"] = None
        if self._switch(t["old"]):
            keep = ab_keep(t["pairs"])
            if keep is not None:
                self._finish(keep, pairs=t["pairs"])

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
        (self.dir / "state.json").write_text(json.dumps({"fingerprint": self.fp, "values": values}, default=_plain))

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
