import json

from moestream.config import EngineConfig
from tools.serve import server_argv


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_argv_from_defaults():
    a = server_argv(EngineConfig(pack_dir="/p", port=9000))
    assert _flag(a, "--model") == "/p" and _flag(a, "--port") == "9000"
    assert _flag(a, "--prompt-cache-bytes") == str(40960 * 32768)
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
