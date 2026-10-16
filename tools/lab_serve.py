"""lagunaLab's streaming server (scripts.serve_streaming, read-only) on a given pack, for the pi-task
comparison. Its loader ignores the requested model, but mlx-lm reloads whenever a request names a
model other than the startup one; pi sends "Laguna-XS-2.1", so that is the startup name here.

usage: PYTHONPATH=.:../lagunaLab python3 -m tools.lab_serve --pack <dir> [--slots 64] [--port 8081]
"""
import argparse
import sys


def main(argv=None):
    import scripts.serve_streaming as lab

    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    p.add_argument("--slots", default="64")
    p.add_argument("--port", default="8081")
    a = p.parse_args(argv)
    lab.pack = a.pack
    sys.argv = [sys.argv[0], "--slots", a.slots, "--port", a.port, "--model", "Laguna-XS-2.1",
                "--max-tokens", "4096", "--prompt-cache-size", "6", "--prompt-cache-bytes", str(4 << 30)]
    lab.main()


if __name__ == "__main__":
    main()
