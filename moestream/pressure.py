"""Shrink expert slots when macOS runs short of memory; grow back once it has been calm. A guard, not a trial."""
import dataclasses
import re
import subprocess
import time

from moestream.budget import GB, free_bytes


def swap_used():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=True).stdout
    return float(re.search(r"used = ([\d.]+)M", out).group(1)) * 2**20


def shrink(split, floor, frac=0.1):
    s = dict(split)
    for _ in range(max(1, round(sum(s.values()) * frac))):
        l = max(s, key=s.get)
        if s[l] <= floor:
            break
        s[l] -= 1
    return s


class Pressure:
    def __init__(self, knob, floor, *, every=256, low=1 * GB, high=3 * GB, calm_s=300, free=free_bytes,
                 swap=swap_used, clock=time.monotonic, log=print, cancel=lambda: None):
        self.knob, self.floor, self.every, self.low, self.high, self.calm_s = knob, floor, every, low, high, calm_s
        self.free, self.swap, self.clock, self.log, self.cancel = free, swap, clock, log, cancel
        self.n, self.target, self.calm_since, self.last_swap = 0, None, None, swap()

    @property
    def shrunk(self):
        return self.target is not None

    def guarded(self):
        """The slots knob as the tuner sees it: while shrunk, reads and writes go to the restore target,
        so saved state never records a shrunk split and nothing regrows while memory is short."""
        k = self.knob

        def get():
            return dict(self.target) if self.shrunk else k.get()

        def apply(v):
            if self.shrunk:
                self.target = {int(l): int(n) for l, n in v.items()}
            else:
                k.apply(v)

        return dataclasses.replace(k, get=get, apply=apply, ready=lambda: not self.shrunk)

    def on_token(self):
        self.n += 1
        if self.n % self.every:
            return
        free, swap, now = self.free(), self.swap(), self.clock()
        grew, self.last_swap = swap > self.last_swap, swap
        if grew or free < self.low:
            if self.target is None:  # a slot trial in flight ends; its pre-trial split is what comes back
                old = self.cancel()
                self.target = old if old is not None else self.knob.get()
            self.knob.apply(shrink(self.knob.get(), self.floor))
            self.calm_since = None
            self.log({"pressure": "shrink", "free_gb": free / GB, "swap_gb": swap / GB, "slots": sum(self.knob.get().values())})
        elif self.shrunk:
            if free <= self.high:
                self.calm_since = None
            elif self.calm_since is None:
                self.calm_since = now
            elif now - self.calm_since >= self.calm_s:
                self.knob.apply(self.target)
                self.target = None
                self.log({"pressure": "restore", "free_gb": free / GB, "slots": sum(self.knob.get().values())})
