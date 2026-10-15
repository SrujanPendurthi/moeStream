import pytest
import json

from moestream.config import EngineConfig
from tools.serve import server_argv


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_argv_from_defaults():
    a = server_argv(EngineConfig(pack_dir="/p", port=9000))
    assert _flag(a, "--model") == "/p" and _flag(a, "--port") == "9000"
    assert _flag(a, "--prompt-cache-bytes") == str(40960 * 136 // 256 * 131072)  # q8, 128k
    assert json.loads(_flag(a, "--chat-template-args")) == {"enable_thinking": True}
    assert (_flag(a, "--temp"), _flag(a, "--top-k"), _flag(a, "--top-p")) == ("1.0", "20", "1.0")
    assert _flag(a, "--max-tokens") == "4096" and _flag(a, "--prompt-cache-size") == "6"


def test_argv_follows_kv_bits_context_and_thinking():
    a = server_argv(EngineConfig(pack_dir="/p", kv_bits=8, context_tokens=1000, thinking=False))
    assert _flag(a, "--prompt-cache-bytes") == str(40960 * 136 // 256 * 1000)
    assert json.loads(_flag(a, "--chat-template-args")) == {"enable_thinking": False}


def test_every_flag_is_one_mlx_lm_server_accepts():
    import inspect

    import mlx_lm.server

    src = inspect.getsource(mlx_lm.server)
    for f in server_argv(EngineConfig(pack_dir="/p"))[::2]:
        assert f'"{f}"' in src, f


def test_one_request_at_a_time_since_the_budget_reserves_one_session():
    a = server_argv(EngineConfig(pack_dir="/p"))
    assert _flag(a, "--decode-concurrency") == "1" and _flag(a, "--prompt-concurrency") == "1"


def test_stored_prompt_caches_are_capped_at_one_session_of_kv():
    from types import SimpleNamespace

    from mlx_lm.models.cache import LRUPromptCache

    from tools.serve import patch_server

    server = SimpleNamespace(LRUPromptCache=LRUPromptCache, load=None,
                             ResponseGenerator=type("RG", (), {"_next_request": lambda self, timeout=None: None}))
    patch_server(server, EngineConfig(pack_dir="/p", kv_bits=8, context_tokens=1000))
    assert server.LRUPromptCache(6).max_bytes == 40960 * 136 // 256 * 1000


def test_loader_loads_once_whatever_model_name_a_client_sends(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr("moestream.tune.start", lambda *a, **k: None)

    from tools.serve import make_loader

    calls = []
    tok = SimpleNamespace(eos_token_ids={2})
    monkeypatch.setattr("moestream.load.load", lambda cfg: calls.append(cfg) or ("model", tok))
    loader = make_loader(EngineConfig(pack_dir="/p", kv_tier=False))
    first = loader("/p")
    assert loader("Laguna-XS-2.1") is first and len(calls) == 1


def test_idle_polls_drive_the_tuner(monkeypatch):
    from types import SimpleNamespace

    from mlx_lm.models.cache import LRUPromptCache

    import tools.serve as serve

    class RG:
        def _next_request(self, timeout=None):
            return self.queued

    server = SimpleNamespace(LRUPromptCache=LRUPromptCache, load=None, ResponseGenerator=RG)
    serve.patch_server(server, EngineConfig(pack_dir="/p"))
    idles = []
    monkeypatch.setattr(serve, "TUNER", [SimpleNamespace(idle=lambda: idles.append(1))])
    rg = RG()
    rg.queued = None
    assert rg._next_request(0.1) is None and idles == [1]
    rg.queued = "req"
    assert rg._next_request(0.1) == "req" and idles == [1]


def test_tuner_error_in_idle_turns_tuning_off_instead_of_killing_the_thread(monkeypatch):
    from types import SimpleNamespace

    from mlx_lm.models.cache import LRUPromptCache

    import tools.serve as serve

    class RG:
        def _next_request(self, timeout=None):
            return None

    serve.patch_server(SimpleNamespace(LRUPromptCache=LRUPromptCache, load=None, ResponseGenerator=RG),
                       EngineConfig(pack_dir="/p"))

    def boom():
        raise OSError("disk full")

    tuners = [SimpleNamespace(idle=boom)]
    monkeypatch.setattr(serve, "TUNER", tuners)
    with pytest.warns(UserWarning, match="tuning off"):
        assert RG()._next_request(0.1) is None
    assert tuners == []


def test_batched_poll_between_tokens_does_not_run_idle_work(monkeypatch):
    from types import SimpleNamespace

    from mlx_lm.models.cache import LRUPromptCache

    import tools.serve as serve

    class RG:
        def _next_request(self, timeout=None):
            return None

    serve.patch_server(SimpleNamespace(LRUPromptCache=LRUPromptCache, load=None, ResponseGenerator=RG),
                       EngineConfig(pack_dir="/p"))
    idles = []
    monkeypatch.setattr(serve, "TUNER", [SimpleNamespace(idle=lambda: idles.append(1))])
    RG()._next_request(None)  # mlx-lm's non-blocking poll while a batch is decoding
    assert idles == []


def _tiered(tmp_path, monkeypatch, **cfg):
    """A packed tiny model, and mlx-lm's prompt cache class as patch_server leaves it."""
    from types import SimpleNamespace

    from mlx_lm.models.cache import LRUPromptCache

    import tools.serve as serve
    from moestream import kvtier
    from moestream.load import load
    from scripts.pack_experts import pack
    from tests.tiny import quantized_model_dir

    d, _ = quantized_model_dir(tmp_path)
    pack(d, tmp_path / "packed")
    model, _ = load(tmp_path / "packed", slots=4)
    c = EngineConfig(pack_dir=str(tmp_path / "packed"), kv_dir=str(tmp_path / "kv"), kv_bits=0,
                     context_tokens=1000, **cfg)
    server = SimpleNamespace(LRUPromptCache=LRUPromptCache, load=None,
                             ResponseGenerator=type("RG", (), {"_next_request": lambda self, timeout=None: None}))
    monkeypatch.setattr(serve, "TIER", [])
    monkeypatch.setattr(kvtier, "MIN_RESTORE_TOKENS", 4)
    serve.patch_server(server, c)
    serve.start_tier(model, c)
    return model, server.LRUPromptCache


def _turn(model, pc, tokens, n=3):
    """One request as mlx-lm serves it: fetch, prefill the rest, greedy-decode n tokens, insert."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache, rest = pc.fetch_nearest_cache("k", tokens)
    cache = cache if cache is not None else make_prompt_cache(model)
    logits, out = model(mx.array([rest]), cache=cache)[:, -1:], []
    seen = [logits]
    for _ in range(n):
        out.append(int(mx.argmax(logits[0, -1]).item()))
        logits = model(mx.array([[out[-1]]]), cache=cache)
        seen.append(logits)
    pc.insert_cache("k", tokens + out, cache)
    return len(rest), out, mx.concatenate(seen, axis=1)


T1, T2 = list(range(100, 130)), list(range(100, 130)) + [5, 6, 7] + list(range(200, 215))


def test_turn_2_after_an_ssd_restore_equals_turn_2_after_a_ram_hit(tmp_path, monkeypatch):
    import mlx.core as mx

    import tools.serve as serve

    model, LRU = _tiered(tmp_path, monkeypatch)
    ram = LRU(6)
    _, out, _ = _turn(model, ram, T1)
    serve.TIER[0][0].flush()
    t2 = T1 + out + T2[30:]
    n_ram, out_ram, logits_ram = _turn(model, ram, t2)
    n_ssd, out_ssd, logits_ssd = _turn(model, LRU(6), t2)    # empty RAM: a restart, or pushed out
    assert n_ram == n_ssd == len(T2) - 30                     # both reuse all of turn 1
    assert out_ssd == out_ram and mx.array_equal(logits_ssd, logits_ram).item()


def test_kv_tier_off_writes_nothing(tmp_path, monkeypatch):
    import tools.serve as serve

    model, LRU = _tiered(tmp_path, monkeypatch, kv_tier=False)
    _turn(model, LRU(6), T1)
    assert serve.TIER == [] and not (tmp_path / "kv").exists()


def test_a_restore_error_falls_back_to_a_full_prefill(tmp_path, monkeypatch):
    import tools.serve as serve

    model, LRU = _tiered(tmp_path, monkeypatch)
    _, out, _ = _turn(model, LRU(6), T1)
    store = serve.TIER[0][0]
    store.flush()
    (torn,) = store.chains
    for f in (tmp_path / "kv").glob("*/full_*"):
        f.write_bytes(b"torn")
    t2 = T1 + out + T2[30:]
    assert _turn(model, LRU(6), t2)[0] == len(t2)   # the chain is dropped: a full prefill
    assert torn not in store.chains and not (tmp_path / "kv" / torn).exists()


def test_shared_machine_mode_drops_the_wired_limit(monkeypatch):
    import mlx.core as mx

    import tools.serve as serve

    limits = []
    monkeypatch.setattr(mx, "set_wired_limit", lambda n: limits.append(("wired", n)) or 0)
    serve.shared_machine()
    mx.set_wired_limit(1 << 34)                       # what mlx-lm does at start and per generation
    assert limits == []


@pytest.mark.parametrize("gb, want", [(1.0, [1 << 30]), (0.5, [1 << 29]), (0, [])])
def test_buffer_pool_cap_comes_from_the_config(monkeypatch, gb, want):
    import mlx.core as mx

    import tools.serve as serve

    limits = []
    monkeypatch.setattr(mx, "set_cache_limit", lambda n: limits.append(n) or 0)
    serve.cap_buffer_pool(gb)
    assert limits == want


def test_each_request_logs_mlx_memory_when_asked(tmp_path, monkeypatch):
    import json

    log = tmp_path / "mem.log"
    monkeypatch.setenv("MOESTREAM_MEMLOG", str(log))
    model, LRU = _tiered(tmp_path, monkeypatch)
    _turn(model, LRU(6), T1)
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["fetch", "insert"]
    assert rows[0]["prompt_tokens"] == len(T1) and rows[0]["reused"] == 0 and rows[1]["tokens"] == len(T1) + 3
    for r in rows:
        assert {"t", "active_gb", "cache_gb", "peak_gb"} <= set(r) and r["active_gb"] > 0
    assert rows[1]["misses"] > 0 and "hits" in rows[1]   # cumulative expert hits/misses, for hit rate


def test_a_tier_error_falls_back_to_a_normal_prefill_and_never_raises(tmp_path, monkeypatch):
    import tools.serve as serve

    monkeypatch.setenv("MOESTREAM_MEMLOG", str(tmp_path / "no" / "such" / "dir" / "mem.log"))
    model, LRU = _tiered(tmp_path, monkeypatch)
    store = serve.TIER[0][0]

    def boom(*a, **k):
        raise KeyError("dtype")

    monkeypatch.setattr(store, "lookup", boom)
    monkeypatch.setattr(store, "save", boom)
    assert _turn(model, LRU(6), T1)[0] == len(T1)   # fetch, prefill, insert: no exception


def test_the_check_script_keeps_its_cache_under_its_results_dir():
    from pathlib import Path

    s = Path("tools/kvtier_check.sh").read_text()
    assert 'KV=$(cd "$OUT" && pwd)/kvcache' in s and "KV=$REPO/kvcache" not in s
