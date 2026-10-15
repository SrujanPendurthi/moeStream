# Serving and pi

## Start the server

```bash
PYTHONPATH=.:../lagunaLab /Users/srujanpendurthi/Projects/Current/.gpu/gpu-run moe \
  tools/watch.sh 4 17 python3 -m tools.serve [--config moestream.toml] [mlx_lm.server flags]
```

- `gpu-run moe` takes the shared GPU lock (one heavy job at a time; lagunaLab has priority). See
  [development.md](development.md).
- `tools/watch.sh 4 17` kills the server if system swap passes 4 GB or its footprint passes 17 GB,
  and prints the peaks when it exits. Use `8 21` only when nobody else is using the machine.
- Loading takes under a minute. The server is ready when the log shows both the banner
  (`moeStream: 3456 slots (6.12 GB), per layer 64-128, half-life 64, eviction lfu, kv 8-bit`) and
  `Starting httpd at 127.0.0.1 on port 8081`.
- Stop it by killing the **Python process's** PID, never with `pkill -f`. The `gpu-run` and `watch.sh`
  command lines contain the same text, and killing the watchdog orphans the server:

  ```bash
  ps -axo pid,command | grep -E "[Pp]ython3? -m tools.serve" | grep -v -e watch.sh -e gpu-run
  ```

## API

mlx-lm's OpenAI-compatible server (`mlx_lm.server`, 0.31.3) with moeStream's loader swapped in.

```bash
curl -s http://127.0.0.1:8081/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Laguna-XS-2.1",
  "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}],
  "max_tokens": 512}'
```

With `thinking = true` the reply's `message.reasoning` holds the model's `<think>` text and
`message.content` the answer. Tool calls use the OpenAI shape; `function.arguments` must be a JSON
**string**, not an object (mlx-lm rejects the request otherwise, with HTTP 404 and
`the JSON object must be str, bytes or bytearray, not dict`).

### What `tools/serve.py` changes in mlx-lm

| Change | Why |
|---|---|
| One engine for the server's lifetime | mlx-lm reloads whenever a request names a model other than the startup path (pi sends `Laguna-XS-2.1`); a second load would double memory |
| Adds token 24 (`</assistant>`) as an end-of-sequence token | `generation_config.json` lists `[2, 24]`; mlx-lm only infers 2, so the model would never stop at end of turn |
| Caps stored conversation caches at one session of KV | mlx-lm builds its prompt cache without a byte cap on the path quantized KV takes; the memory budget reserves room for exactly one stored session |
| One request at a time (decode and prompt concurrency 1) | the budget reserves KV for one live session |
| Sampling from `generation_config.json` | temperature 1.0, top-k 20, top-p 1.0 |
| Chat template `enable_thinking` from `thinking` | |
| Runs one slice of tuner work whenever the generation thread finds no request waiting | the tuner shares mlx-lm's generation thread ([self-tuning.md](self-tuning.md)) |
| lagunaLab's SSE keepalive filter (`scripts.serve_streaming`) | keeps streaming clients connected through long prefills |

Command-line flags after the config override the derived ones (for example `--max-tokens`).

### Prompt cache

mlx-lm reuses the KV of the longest stored prefix of a new request. In a pi session each turn
extends the previous one, so after the first prompt only the new tokens are processed (measured:
1388 tokens on turn 1, then 22-259 per request). Reuse breaks when the earlier messages change, for
example when a client replays a recorded session whose assistant turns differ from what the model
just generated.

### SSD tier

With `kv_tier = true` (the default) every turn-end cache mlx-lm stores in RAM is also written to
`kv_dir` by a background thread (`moestream/kvtier.py`, spec
`docs/specs/2026-10-14-ssd-kv-tier-design.md`). A request whose prompt extends a stored snapshot by
at least 2048 tokens more than the RAM cache covers is restored from disk instead of re-processed:
this is what keeps 2-3 interleaved sessions, and any session after a restart, from re-processing
their whole history. The log shows each restore:

```
moeStream kv tier: restored 41210 tokens in 0.48 s (RAM had 0, 312 left to process)
```

- Per conversation: full-attention KV appended per turn, plus the sliding-window state of the
  latest turn and of every 8th turn. About 87 MB written per turn at q8.
- Files are written as temp files, fsynced and renamed, the manifest last: a crash loses at most
  the last turn. A different pack, `kv_bits` or format version wipes the directory at startup.
- A failed read drops that conversation's snapshots and falls back to a normal prefill; three
  failed writes in a row turn writes off until restart. Neither fails a request.
- Delete `kv_dir` to clear it; the server recreates it.

## pi

The provider entry in `~/.pi/agent/models.json`:

```json
"moestream": {
  "baseUrl": "http://127.0.0.1:8081/v1",
  "api": "openai-completions",
  "apiKey": "none",
  "compat": { "supportsDeveloperRole": false },
  "models": [{ "id": "Laguna-XS-2.1", "reasoning": true, "contextWindow": 131072, "maxTokens": 4096 }]
}
```

Keep `contextWindow` equal to `context_tokens`. Then:

```bash
pi --provider moestream --model Laguna-XS-2.1
```

`pi --list-models` should show `moestream  Laguna-XS-2.1  131.1K  4.1K  yes  no`.

## Verify a deployment

`tools/verify_pi.sh` runs the end-to-end check under the lock and watchdog, with tuning off:

1. Serve the current config; one curl; a 3-turn pi session on a scratch repo with a planted bug
   (run the tests, fix the bug, add an empty-list test); the repo's pytest at the end.
2. One restart each with `eviction = "lru"`, `kv_bits = 0` and `thinking = false`, a curl each.
3. Kill each server by PID and wait for the port to be free before the next.

```bash
WATCH="4 17" tools/verify_pi.sh     # WATCH sets the watchdog limits (default "8 21")
```

Output goes to `results/serve_verify.log`; the summary lines are grepped at the end. The 2026-10-14
results are in the top-level README ("pi check").

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `OSError: [Errno 48] Address already in use` | a previous server still holds port 8081: find its Python PID (above) and kill it |
| decode drops to well under 5 tok/s and the machine is sluggish | macOS is paging: lower `memory_gb`, close heavy apps, check `sysctl vm.swapusage` |
| `kv tier off: ...` warning at load | the tier could not start (for example `kv_dir` is a non-empty directory it did not create); serving goes on without it |
| `aborted: paging` in the log | the watchdog stopped the server at its swap or footprint limit; same fix |
| `memory_gb 14 GB but only 13.8 GB is free` warning at load | free memory is counted before the load; harmless when close, a real risk when far below |
| replies never end | the end-of-turn token is missing: check `generation_config.json` is in the pack |
| HTTP 404 on chat completions | mlx-lm reports generation errors as 404; the response body has the real error |
