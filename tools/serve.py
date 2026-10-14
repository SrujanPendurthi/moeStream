"""Serve moeStream through mlx-lm's OpenAI-compatible server, configured by moestream.toml.

PYTHONPATH=.:../lagunaLab python3 -m tools.serve [--config moestream.toml] [mlx_lm.server flags]
Loads real weights: run it as
  .gpu/gpu-run moe tools/watch.sh 8 21 python3 -m tools.serve
Flags given on the command line override the ones derived from the config.
"""
import json
import sys
from pathlib import Path

from moestream.budget import kv_bytes_per_token
from moestream.config import EngineConfig


def server_argv(cfg):
    return [
        "--model", str(cfg.pack_dir), "--port", str(cfg.port), "--max-tokens", "4096",
        "--prompt-cache-size", "6",
        "--prompt-cache-bytes", str(kv_bytes_per_token(cfg.kv_bits) * cfg.context_tokens),
        "--chat-template-args", json.dumps({"enable_thinking": cfg.thinking}),
        "--temp", "1.0", "--top-k", "20", "--top-p", "1.0",  # generation_config.json's sampling
    ]


def make_loader(cfg):
    def _load(*a, **k):
        from moestream.load import load

        model, tok = load(cfg)
        # generation_config.json's eos_token_id is [2, 24]; 24 is "</assistant>". mlx-lm only
        # infers 2, so without this the model never stops at end of turn (as in lagunaLab).
        gen = Path(cfg.pack_dir) / "generation_config.json"
        if gen.exists():
            for eid in json.loads(gen.read_text()).get("eos_token_id", []):
                if eid not in tok.eos_token_ids:
                    tok.add_eos_token(str(eid))
        return model, tok

    return _load


def main():
    import mlx_lm.server as server

    import scripts.serve_streaming  # noqa: F401  installs lagunaLab's SSE keepalive filter on the handler

    path = "moestream.toml"
    if "--config" in sys.argv:
        i = sys.argv.index("--config")
        path = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    cfg = EngineConfig.read(path)
    server.load = make_loader(cfg)
    sys.argv = sys.argv[:1] + server_argv(cfg) + sys.argv[1:]
    server.main()


if __name__ == "__main__":
    main()
