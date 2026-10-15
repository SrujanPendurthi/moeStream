import mlx.core as mx

from tools.prefill_mem import fit, measure


class Fake:
    """Logits of zeros; records each forward's token count."""
    def __init__(self):
        self.calls = []

    def make_cache(self):
        return []

    def __call__(self, x, cache=None, **kw):
        self.calls.append(x.shape[1])
        return mx.zeros((1, x.shape[1], 8))


def test_measure_prefills_in_steps_of_the_given_size():
    m = Fake()
    row = measure(m, list(range(1000)), 256)
    assert m.calls[:4] == [256, 256, 256, 231]          # mlx-lm keeps the last token for the first decode
    assert row["step"] == 256 and row["tokens"] == 1000 and row["prefill_tok_s"] > 0
    assert abs(row["transient_gb"] - (row["peak_gb"] - row["active_gb"])) < 1e-9


def test_fit_recovers_a_line_per_step():
    rows = [{"step": s, "tokens": n, "transient_gb": 1.5 + k * n / 1000}
            for s, k in ((512, 0.05), (2048, 0.2)) for n in (5000, 15000, 30000)]
    f = fit(rows)
    assert abs(f[512]["gb"] - 1.5) < 1e-9 and abs(f[512]["gb_per_1k_tokens"] - 0.05) < 1e-9
    assert abs(f[2048]["gb_per_1k_tokens"] - 0.2) < 1e-9


def test_sweep_warms_up_before_the_first_measured_point(tmp_path):
    from tools.prefill_mem import sweep

    m = Fake()
    rows = sweep(m, list(range(300)), [128], [500], tmp_path / "out.json", warmup=256)
    assert m.calls[:2] == [128, 127]                     # warm-up prefill, not recorded
    assert [(r["step"], r["tokens"]) for r in rows] == [(128, 500)]
    assert (tmp_path / "out.json").exists()
