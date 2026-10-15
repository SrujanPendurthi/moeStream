"""Strict KL guard: the engine against a fixed reference of the untuned engine.

Work runs in bounded slices (one chunk of tokens per step) so it can share mlx-lm's generation
thread with live requests.
"""
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

K = 64
THRESHOLDS = {"mean": 1e-3, "p99": 1e-2, "top1": 0.995}


def _log_softmax(x):
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def reference(logits, k=K):
    """(T, V) logits -> top-k token ids (best first) and their log-probabilities."""
    lp = _log_softmax(logits)
    ids = np.argsort(-lp, axis=-1)[:, :k]
    return ids, np.take_along_axis(lp, ids, -1)


def kl_topk(ref, logits):
    """KL(reference || candidate) on the reference's top-k support, renormalized; and top-1 agreement."""
    ids, ref_lp = ref
    p, q = _log_softmax(ref_lp), _log_softmax(np.take_along_axis(_log_softmax(logits), ids, -1))
    return (np.exp(p) * (p - q)).sum(-1), ids[:, 0] == logits.argmax(-1)


def verdict(kl, top1, th=THRESHOLDS):
    v = {"kl_mean": float(kl.mean()), "kl_p99": float(np.quantile(kl, 0.99)), "top1": float(top1.mean())}
    v["ok"] = v["kl_mean"] <= th["mean"] and v["kl_p99"] <= th["p99"] and v["top1"] >= th["top1"]
    return v


def quiet(v, th=THRESHOLDS):
    """Two runs of the same engine: do they disagree by less than half the threshold?"""
    return v["kl_mean"] <= th["mean"] / 2 and v["kl_p99"] <= th["p99"] / 2 and 1 - v["top1"] <= (1 - th["top1"]) / 2


class Guard:
    def __init__(self, model, prompts, path, key, total=2048, chunk=128, k=K, th=THRESHOLDS):
        self.model, self.path, self.key, self.chunk, self.k, self.th = model, Path(path), key, chunk, k, th
        self.ready = self.math_ok = False
        self._run, self.noise = None, None
        try:
            z = np.load(self.path)
            if str(z["key"]) == key:
                self.seqs, self.ref = z["seqs"], (z["ids"], z["logp"])
                self.math_ok, self.ready = bool(z["math_ok"]), True
                self.noise = json.loads(str(z["noise"])) if "noise" in z else None
        except Exception:  # missing or unreadable: rebuild
            pass
        if not self.ready:
            self._run = self._build(prompts, total)

    def start(self):
        self._run = self._compare()

    def step(self):
        """One slice of work. Returns the verdict when a comparison finishes, else None."""
        try:
            next(self._run)
        except StopIteration as s:
            self._run = None
            return s.value
        except Exception as e:  # out of memory and the like: a failed guard, never a crash
            self._run = None
            if not self.ready:
                self.ready = True  # math_ok stays False: math-changing knobs stay off
                return None
            return {"ok": False, "error": repr(e)}
        return None

    def _build(self, prompts, total):
        per = total // len(prompts)
        seqs = []
        for p in prompts:
            ids, cache = [int(t) for t in p[:per]], self.model.make_cache()
            logits = self.model(mx.array([ids]), cache=cache)[0, -1]
            while len(ids) < per:  # the untuned engine's own greedy continuation
                ids.append(int(mx.argmax(logits)))
                logits = self.model(mx.array([ids[-1:]]), cache=cache)[0, -1]
                if len(ids) % 16 == 0:
                    yield
            seqs.append(ids)
            yield
        self.seqs = np.array(seqs, np.int32)
        parts = []
        for lg in self._chunks():
            parts.append(reference(lg, self.k))
            yield
        self.ref = tuple(np.concatenate(a) for a in zip(*parts))
        v = yield from self._compare()
        self.math_ok, self.noise = quiet(v, self.th), v
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(tmp, key=self.key, seqs=self.seqs, ids=self.ref[0], logp=self.ref[1], math_ok=self.math_ok,
                 noise=json.dumps(v))
        os.replace(tmp, self.path)  # a crash mid-write never leaves a reference that fails to load
        self.ready = True

    def _chunks(self):
        for ids in self.seqs:
            cache = self.model.make_cache()
            for a in range(0, len(ids), self.chunk):
                yield np.array(self.model(mx.array(ids[None, a : a + self.chunk]), cache=cache)[0].astype(mx.float32))

    def _compare(self):
        kls, tops, i = [], [], 0
        for lg in self._chunks():
            kl, top = kl_topk((self.ref[0][i : i + len(lg)], self.ref[1][i : i + len(lg)]), lg)
            kls.append(kl); tops.append(top); i += len(lg)
            yield
        return verdict(np.concatenate(kls), np.concatenate(tops), self.th)
