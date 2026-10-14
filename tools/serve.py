"""Serve moeStream through mlx-lm's OpenAI-compatible server, configured by moestream.toml.

PYTHONPATH=.:../lagunaLab python3 -m tools.serve [--config moestream.toml] [mlx_lm.server flags]
Loads real weights: run it as
  .gpu/gpu-run moe tools/watch.sh 8 21 python3 -m tools.serve
Flags given on the command line override the ones derived from the config.
"""
import json
import sys
import warnings
from pathlib import Path

from moestream.budget import kv_bytes_per_token
from moestream.config import EngineConfig

TUNER = []  # the running engine's tuner, once loaded


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
        try:
            from moestream.tune import start

            TUNER.append(start(model, tok, cfg))
        except Exception as e:  # tuning never stops serving
            warnings.warn(f"tuning off: {e!r}")
        return loaded[0]

    return _load


def patch_server(server, cfg):
    """Swap in the moeStream loader, and cap stored conversation caches at one session of KV.
    mlx-lm 0.31.3 builds LRUPromptCache without max_bytes; --prompt-cache-bytes alone is only
    applied on its batched path, never on the single-request path quantized KV takes. Also runs one
    slice of tuner work whenever the generation thread polls an empty request queue."""
    lru = server.LRUPromptCache
    server.load = make_loader(cfg)
    server.LRUPromptCache = lambda max_size: lru(max_size, max_bytes=session_kv_bytes(cfg))
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


def main():
    import mlx_lm.server as server

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
