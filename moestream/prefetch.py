"""Background expert reads into reusable staging buffers.

Worker threads only run os.preadv (it releases the GIL). Buffers are handed out and
returned on the main thread, so the free list needs no lock.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_POOL = ThreadPoolExecutor(8)


class Prefetcher:
    def __init__(self, path, stride):
        self.fd = os.open(path, os.O_RDONLY)
        self.stride = stride
        self.pending = {}  # expert -> (buf, future)
        self.free = []
        self.allocated = 0

    def _buf(self):
        if self.free:
            return self.free.pop()
        self.allocated += 1
        return np.empty(self.stride, np.uint8)

    def _read(self, e, buf):
        n = os.preadv(self.fd, [buf], e * self.stride)
        if n != self.stride:
            raise OSError(f"short read: expert {e}, {n}/{self.stride} bytes")
        return buf

    def submit(self, experts):
        for e in experts:
            if e not in self.pending:
                buf = self._buf()
                self.pending[e] = (buf, _POOL.submit(self._read, e, buf))

    def take(self, experts):
        """{expert: staging buffer}, prefetched or read now. Caller must release() each buffer."""
        self.submit(experts)
        out = {}
        for e in experts:
            buf, fut = self.pending.pop(e)
            try:
                out[e] = fut.result()
            except OSError:
                out[e] = self._read(e, buf)  # one synchronous retry; a second failure raises
        return out

    def release(self, buf):
        self.free.append(buf)

    def drop_pending(self):
        """Forget unused predictions and recycle their buffers."""
        for buf, fut in self.pending.values():
            fut.exception()  # ponytail: waits on stragglers (issued a layer ago, usually done); cancel if it shows in profiles
            self.free.append(buf)
        self.pending.clear()
