import numpy as np
import pytest

from moestream.prefetch import Prefetcher

STRIDE = 16384


def _file(tmp_path, n=6):
    p = tmp_path / "layer.bin"
    p.write_bytes(b"".join(bytes([e]) * STRIDE for e in range(n)))
    return p


def test_take_returns_right_bytes_prefetched_or_not(tmp_path):
    pf = Prefetcher(_file(tmp_path), STRIDE)
    pf.submit([3])
    got = pf.take([3, 5])
    assert (got[3] == 3).all() and (got[5] == 5).all() and len(got[3]) == STRIDE
    assert pf.pending == {}


def test_buffers_are_reused_not_grown(tmp_path):
    pf = Prefetcher(_file(tmp_path), STRIDE)
    for i in range(100):
        pf.submit([i % 6])
        for buf in pf.take([i % 6, (i + 1) % 6]).values():
            pf.release(buf)
    assert pf.allocated <= 2


def test_drop_pending_recycles_buffers(tmp_path):
    pf = Prefetcher(_file(tmp_path), STRIDE)
    pf.submit([0, 1, 2])
    pf.drop_pending()
    assert pf.pending == {} and len(pf.free) == 3
    pf.submit([4]); pf.drop_pending()
    assert pf.allocated == 3


def test_short_read_raises_after_retry(tmp_path):
    p = tmp_path / "layer.bin"
    p.write_bytes(b"\0" * (STRIDE + 10))  # expert 1 is truncated
    pf = Prefetcher(p, STRIDE)
    with pytest.raises(OSError, match="short read"):
        pf.take([1])


def test_failed_background_read_retries_once(tmp_path):
    class Flaky(Prefetcher):
        fails = 1

        def _read(self, e, buf):
            if Flaky.fails:
                Flaky.fails -= 1
                raise OSError("transient")
            return super()._read(e, buf)

    pf = Flaky(_file(tmp_path), STRIDE)
    assert (pf.take([2])[2] == 2).all()


def test_free_list_is_capped_after_a_big_batch(tmp_path):
    pf = Prefetcher(_file(tmp_path), STRIDE, keep=2)
    for buf in pf.take([0, 1, 2, 3, 4, 5]).values():  # prefill-sized fill batch
        pf.release(buf)
    assert len(pf.free) == 2
