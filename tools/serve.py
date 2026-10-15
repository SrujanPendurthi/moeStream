"""Serve moeStream through mlx-lm's OpenAI-compatible server, configured by moestream.toml.

PYTHONPATH=.:../lagunaLab python3 -m tools.serve [--config moestream.toml] [mlx_lm.server flags]
Loads real weights: run it as
  .gpu/gpu-run moe tools/watch.sh 8 21 python3 -m tools.serve
Flags given on the command line override the ones derived from the config.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

from moestream.budget import kv_bytes_per_token
from moestream.config import EngineConfig

TUNER = []  # the running engine's tuner, once loaded
TIER = []   # (kvtier.Store, model): the SSD tier for the prompt cache, once loaded


def session_kv_bytes(cfg):
    return kv_bytes_per_token(cfg.kv_bits) * cfg.context_tokens


def server_argv(cfg):
    return [
        "--model", str(cfg.pack_dir), "--port", str(cfg.port), "--max-tokens", "4096",
        "--prompt-cache-size", "6",
        "--prompt-cache-bytes", str(session_kv_bytes(cfg)),
        # the memory budget reserves KV for one live session (plus one stored), not one per client
        "--decode-concurrency", "1", "--prompt-concurrency", "1",
        "--chat-template-args", json.dumps({"enable_thinking": cfg.thinking}),
        "--temp", "1.0", "--top-k", "20", "--top-p", "1.0",  # generation_config.json's sampling
    ]


def make_loader(cfg):
    """One engine for the server's lifetime. mlx-lm reloads whenever a request names a model other
    than the startup path (pi sends "Laguna-XS-2.1"); a second load would double memory."""
    loaded = []

    def _load(*a, **k):
        if loaded:
            return loaded[0]
        from moestream.load import load

        model, tok = load(cfg)
        # generation_config.json's eos_token_id is [2, 24]; 24 is "</assistant>". mlx-lm only
        # infers 2, so without this the model never stops at end of turn (as in lagunaLab).
        gen = Path(cfg.pack_dir) / "generation_config.json"
        if gen.exists():
            for eid in json.loads(gen.read_text()).get("eos_token_id", []):
                if eid not in tok.eos_token_ids:
                    tok.add_eos_token(str(eid))
        loaded.append((model, tok))
        start_tier(model, cfg)
        try:
            from moestream.tune import start

            TUNER.append(start(model, tok, cfg))
        except Exception as e:  # tuning never stops serving
            warnings.warn(f"tuning off: {e!r}")
        return loaded[0]

    return _load


def start_tier(model, cfg):
    if not cfg.kv_tier:
        return
    try:
        from moestream.kvtier import Store, fingerprint

        TIER[:] = [(Store(cfg.kv_dir, fingerprint(cfg.pack_dir, cfg.kv_bits), int(cfg.kv_disk_gb * 1e9)), model)]
    except Exception as e:  # the tier never stops serving
        warnings.warn(f"kv tier off: {e!r}")


def patch_server(server, cfg):
    """Swap in the moeStream loader, and cap stored conversation caches at one session of KV.
    mlx-lm 0.31.3 builds LRUPromptCache without max_bytes; --prompt-cache-bytes alone is only
    applied on its batched path, never on the single-request path quantized KV takes. Also runs one
    slice of tuner work whenever the generation thread polls an empty request queue."""
    lru = server.LRUPromptCache
    server.load = make_loader(cfg)
    memlog = os.environ.get("MOESTREAM_MEMLOG")

    def mem(event, **kw):
        """MOESTREAM_MEMLOG=<file>: MLX memory at the start and end of every request, as JSON lines.
        peak_gb at "insert" is the request's peak (reset at each "fetch")."""
        if not memlog:
            return
        import mlx.core as mx

        row = dict(event=event, t=round(time.time(), 2), **kw, active_gb=round(mx.get_active_memory() / 1e9, 3),
                   cache_gb=round(mx.get_cache_memory() / 1e9, 3), peak_gb=round(mx.get_peak_memory() / 1e9, 3))
        with open(memlog, "a") as f:
            f.write(json.dumps(row) + "\n")
        if event == "fetch":
            mx.reset_peak_memory()

    class PromptCache(lru):
        """mlx-lm's RAM prompt cache, plus the SSD tier (moestream/kvtier.py) when it is on."""

        def __init__(self, max_size):
            super().__init__(max_size, max_bytes=session_kv_bytes(cfg))

        def fetch_nearest_cache(self, model, tokens):
            cache, rest = self._fetch(model, tokens)
            mem("fetch", prompt_tokens=len(tokens), reused=len(tokens) - len(rest))
            return cache, rest

        def _fetch(self, model, tokens):
            if TIER:
                store, m = TIER[0]
                # RAM match length without mlx-lm's deep copy, so a restore never sits beside one.
                # Laguna's sliding caches cannot trim, so mlx-lm's "longer entry" branch never applies.
                r = self._trie.search(model, tokens)
                ram = len(tokens) if r.exact is not None else len(r.shorter or [])
                hit = store.lookup(tokens, ram, cfg.context_tokens)
                if hit:
                    from mlx_lm.models.cache import make_prompt_cache

                    from moestream.kvtier import log

                    t, cache = time.perf_counter(), make_prompt_cache(m)
                    if store.restore(hit[1], hit[0], cache):
                        log(f"restored {hit[0]} tokens in {time.perf_counter() - t:.2f} s "
                            f"(RAM had {ram}, {len(tokens) - hit[0]} left to process)")
                        return cache, tokens[hit[0]:]
            return super().fetch_nearest_cache(model, tokens)

        def insert_cache(self, model, tokens, prompt_cache, *, cache_type="assistant"):
            super().insert_cache(model, tokens, prompt_cache, cache_type=cache_type)
            mem("insert", tokens=len(tokens), cache_type=cache_type)
            if TIER and cache_type == "assistant":  # turn ends only; mid-prompt segments are not restorable points
                TIER[0][0].save(tokens, prompt_cache)

    server.LRUPromptCache = PromptCache
    nxt = server.ResponseGenerator._next_request

    def _next_request(self, timeout=None):
        r = nxt(self, timeout)
        if r is None and timeout is not None and TUNER:  # timeout None: polled between batched decode steps
            try:
                TUNER[0].idle()
            except Exception as e:  # raised here, it would end mlx-lm's generation thread and hang the server
                TUNER.clear()
                warnings.warn(f"tuning off: {e!r}")
        return r

    server.ResponseGenerator._next_request = _next_request


def shared_machine():
    """No wired limit (mlx-lm sets one at start and around every generation) and MLX's free-buffer
    pool capped at 1 GB, so the footprint stays near memory_gb while the machine is in use
    (lagunaLab a1ca51e: the pool inflated the footprint past its watchdog)."""
    import mlx.core as mx

    mx.set_cache_limit(1 << 30)
    mx.set_wired_limit = lambda limit: 0


def main():
    import mlx_lm.server as server

    if os.environ.get("MOESTREAM_SHARED") == "1":
        shared_machine()

    import scripts.serve_streaming  # noqa: F401  installs lagunaLab's SSE keepalive filter on the handler

    path = "moestream.toml"
    if "--config" in sys.argv:
        i = sys.argv.index("--config")
        path = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    cfg = EngineConfig.read(path)
    patch_server(server, cfg)
    sys.argv = sys.argv[:1] + server_argv(cfg) + sys.argv[1:]
    server.main()


if __name__ == "__main__":
    main()
