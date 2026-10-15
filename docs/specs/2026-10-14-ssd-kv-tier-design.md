# moeStream: SSD tier for the prompt cache

Date: 2026-10-14
Status: approved by the user 2026-10-14

## Goal

When a conversation's KV is no longer in RAM, because other sessions pushed it out of the prompt cache
or the server restarted, restore it from SSD instead of processing the whole prompt again.

Measured case for it (`tools/kv_tier_sim.py`, `results/kv_tier_sim.json`, README section "SSD tier
for the prompt cache"). Over the 11 captured agent sessions (690 turns, 29.3 M prompt tokens), with
the current RAM cap of one session of q8 KV:

- 2 interleaved sessions re-process 3.75 M prompt tokens instead of 0.46 M (about 10 hours against
  78 minutes at 100 tok/s);
- 3 interleaved sessions re-process 12.9 M (about 36 hours);
- with an SSD tier, every workload drops to the 0.46 M floor (first prompts plus each turn's new tool
  output, which must be processed anyway);
- one session at a time gains only after a restart (up to about 80k tokens, about 13 minutes).

The harness, its evals and GRPO rollouts will run sessions side by side, so this comes before them.

Done means: with 3 interleaved captured sessions on real weights, prompt tokens processed are within
10% of the floor, a restart resumes every session from SSD, output after a restore matches output
after a RAM-cache reuse, and peak memory stays inside `memory_gb` plus the known 1.2 GB.

## Not in scope

- Sharing a prefix across different sessions (for example a common system prompt). The sliding-window
  layers cannot rewind, so only exact turn-end states are restorable.
- Restoring from the middle of a turn, or after the history was edited before the last snapshot (that
  falls back to the longest earlier snapshot that still matches).
- Compressing KV on disk beyond its live format.
- Any change to lagunaLab.

## Background: how mlx-lm's prompt cache works today

`mlx_lm.models.cache.LRUPromptCache` (0.31.3) keeps whole per-layer cache lists in a token trie.
`fetch_nearest_cache(tokens)` returns a **deep copy** of the entry whose tokens are the longest prefix
of the prompt, plus the remaining tokens to process. For non-trimmable caches (Laguna's 30 sliding
layers are `RotatingKVCache`, window 512) only an entry that is an exact prefix can be used. After a
request, `insert_cache` stores the cache for prompt plus generated tokens. `tools/serve.py` caps the
cache at one session of KV bytes, and the memory budget reserves KV for two sessions: the live one
(the deep copy) and the stored one.

In live use, each turn's stored entry is an exact prefix of the next turn's prompt; the simulation
checked this for every turn of all 11 sessions through Laguna's chat template.

mlx-lm has `save_prompt_cache` and `load_prompt_cache` (safetensors). They are not used directly:
`load_prompt_cache` rebuilds cache objects by class name from mlx-lm's own module, which does not
know lagunaLab's quantized cache classes, and `mx.load` reads a whole file into new arrays at once.
The format below keeps their idea (per-layer `state` and `meta_state`) and restores into a cache list
made by `model.make_cache()`.

## Design

### Snapshot points

A snapshot is the KV state at the end of an assistant turn: the tokens `insert_cache` is given and
the per-layer cache for them. It is written when the entry is stored in RAM (`insert_cache`), so
every turn end is restorable.

### On-disk layout

```
kvcache/                              kv_dir in moestream.toml
  FORMAT                              format version, pack layout hash, kv_bits
  <chain>/                            one directory per conversation chain
    manifest.json                     the chain's snapshots: tokens count, prefix hash, file names, last use
    full_000.safetensors ...          full-attention layers: KV for token range [a, b), appended per turn
    sliding_<tokens>.safetensors      the 30 sliding layers' whole state at one snapshot (about 63 MB)
```

- **Full-attention KV is append-only.** A turn writes only its new token range for the 10 full layers,
  in the live format (q8 by default: packed values, scales, biases), so a 75k-token chain is about
  1.6 GB on disk however many turns it took.
- **Sliding state is whole per snapshot**, since a rotating buffer cannot be rebuilt from deltas. The
  latest two per chain are kept (amended 2026-10-14, see below), plus one every 8 turns as a fallback
  for histories edited further back. Older ones are deleted when a newer one lands.
- A **chain** is a sequence of snapshots where each is a prefix of the next. A snapshot whose tokens
  extend an existing chain's latest snapshot is appended to it; otherwise it starts a new chain. That
  covers a conversation that branches: the branch gets a new chain from its first unmatched turn.

### Entry identity and invalidation

`FORMAT` holds the format version, a hash of the pack's `layout.json` and config, and `kv_bits`. If
any differs from the running server at startup, the whole `kv_dir` is deleted with a log line.
`memory_gb` and `context_tokens` do not invalidate entries; an entry longer than `context_tokens` is
skipped at lookup.

### Index and lookup

At startup the server reads every `manifest.json` into an in-memory index: for each snapshot, its
token count and the SHA-256 of its token ids. No KV is read.

`fetch_nearest_cache(tokens)`, in order:

1. **RAM:** mlx-lm's own lookup. Its matched prefix length is `r`.
2. **Index:** hash the prompt once, in a single pass, recording the digest at each token count that
   any snapshot has. The longest snapshot whose digest matches is `s`.
3. **SSD,** only if `s - r >= min_restore_tokens` (default 2048) and the snapshot fits under
   `context_tokens`. At 100 tok/s prefill against about 2 GB/s reads, any restore over a few hundred
   tokens wins; 2048 keeps small restores from churning the disk.

Otherwise the RAM result (or nothing) is used as today.

### Restore without a memory spike

The 20 GB run showed that memory over the cap means paging and collapse, so a restore must not add a
second copy of the KV or fill the page cache:

- The restored cache takes the place of the deep copy that a RAM hit would make. It uses the
  already-reserved live-session KV, so no new reservation is needed.
- Files are read one array at a time with `F_NOCACHE` set on the file descriptor (no unified-memory
  page cache), into one reusable host buffer the size of the largest array, then copied into the
  layer's cache arrays. Peak extra memory is one array: about 170 MB for a 78k-token full-attention
  layer at q8.
- Restore runs on the generation thread before prefill, in place of the prefill it saves.

### Writes

- **Asynchronous:** `insert_cache` hands the snapshot to one writer thread and returns. The writer
  evaluates and copies arrays one at a time (the KV arrays are not mutated after insert, since each
  later turn works on a deep copy), so its extra memory is also one array.
- **Crash-safe:** each file is written as `<name>.tmp`, `fsync`ed and renamed. `manifest.json` is
  replaced the same way, **after** the files it names exist. At startup, `.tmp` files and files not
  named by any manifest are deleted. A torn chain therefore loses at most its last turn.
- **Back-pressure:** if a snapshot for a chain is still being written when the next one for the same
  chain arrives, the older one is dropped. Writes never block a request.
- **Volume:** about 87 MB per turn in the simulation (mostly the 63 MB sliding state), 60 GB over
  the 11 sessions. At the M4 SSD's rated endurance (hundreds of TB) this is not a concern for this
  workload; `kv_tier = false` turns writes off.

### Disk cap and pruning

`kv_disk_gb` caps the directory. After each write, while the total is over the cap, delete the chain
with the oldest last use (whole chains: a chain without its early full-attention segments is
useless). Default 10 GB: about six 75k-token conversations at q8, enough for 3 interleaved sessions
plus a restart, and under half the 24 GB currently free.

### Configuration

```toml
kv_tier = true            # SSD tier for the prompt cache
kv_dir = "kvcache"        # relative to this file; git-ignored
kv_disk_gb = 10.0         # LRU prune of whole chains above this
```

`min_restore_tokens` stays a constant until a measurement says otherwise.

### Code

- New `moestream/kvtier.py`: `Tier(dir, fingerprint, cap)` with `lookup(tokens) -> (n, chain)`,
  `restore(chain, n, cache)`, `save(tokens, cache)` (queues), `prune()`; the format and the writer
  thread.
- `tools/serve.py`: the `LRUPromptCache` wrapper it already installs gains the SSD lookup in
  `fetch_nearest_cache` and the save in `insert_cache`.
- `moestream/config.py`: the three keys.

## Errors

- A read fails, or a file is missing or the wrong size: drop that chain from the index and the disk,
  fall back to the RAM result or full prefill. Never fail the request.
- A write fails (disk full, I/O error): log, drop that snapshot, keep serving. Three failures in a
  row turn writes off until restart.
- Fingerprint mismatch at startup: delete `kv_dir`, start empty.
- Unreadable manifest: delete that chain.

## Tests (tiny model, no real weights)

In the existing style: a tiny Laguna with full and sliding layers and a sliding window smaller than
the test prompts, so the rotating buffer wraps.

1. **Round trip, bit-exact:** prefill a prompt, snapshot, restore into a fresh `make_cache()`; the
   next-token logits equal those from the original cache exactly, at `kv_bits` 0 and 8.
2. **Restore then continue equals RAM reuse:** turn 2 decoded after an SSD restore gives the same
   tokens and logits as turn 2 after mlx-lm's RAM-cache reuse.
3. **Append-only chains:** three turns write three full-attention segments; the chain's bytes equal
   the cache's full-attention bytes at turn 3; restoring turn 2 from a 3-turn chain truncates
   correctly.
4. **Sliding state retention:** only the latest two and every 8th sliding snapshot survive; a dead-end
   reply that lands between a lookup and its restore does not cost the turn it branched from.
5. **Lookup:** the longest matching snapshot wins; a match under `min_restore_tokens` reads nothing;
   a branch starts a new chain; an entry longer than `context_tokens` is skipped.
6. **Crash safety:** a `.tmp` file left by a killed writer is ignored and deleted at startup; a
   manifest naming a missing file drops the chain; a corrupt manifest drops the chain.
7. **Invalidation:** a different `kv_bits`, layout hash or format version wipes the directory.
8. **Disk cap:** writing past `kv_disk_gb` deletes whole chains, oldest last use first.
9. **Memory:** restore allocates at most one host buffer of the largest array's size (counted
   through an injected allocator).
10. **Async and back-pressure:** a slow fake writer does not delay `insert_cache`; a second snapshot
    for the same chain replaces a queued one.
11. **Errors:** a failing read falls back to full prefill with correct output; three failing writes
    turn writes off.

## Verification on real weights (later; through the GPU lock and `watch.sh 4 17`)

1. **Interleaved sessions:** 3 captured sessions interleaved turn by turn, sent as live use would
   (the model's own replies kept in history, about 30 turns each), once with `kv_tier = false` and
   once with it on. Record prompt tokens processed, prompt time, restores and their times, peak
   footprint and swap. Pass: with the tier, prompt tokens within 10% of the simulated floor for
   those turns; restores at least 10x faster than the prefill they replace; peak footprint at most
   `memory_gb` + 1.2 GB; no swap growth.
2. **Restart:** stop the server midway (kill by PID), start it again, continue all 3 sessions. Pass:
   each resumes from SSD (log line per restore), no full re-prefill.
3. **Hard kill mid-write:** `kill -9` during a snapshot write, restart. Pass: the server starts, the
   torn snapshot is gone, the chain restores from its previous turn.
4. **Output equality:** for 3 turns, greedy continuation after an SSD restore equals the continuation
   after a RAM hit, token for token.
5. **Disk:** directory size stays under `kv_disk_gb`; total bytes written per turn reported.

## Decisions at review (2026-10-14)

- **Disk cap:** 10 GB default, as proposed.
- **Fallback sliding snapshots every 8 turns:** kept.

## Amendment (2026-10-14, approved by the user)

- **Keep the latest two sliding snapshots per chain**, not one. Task 18 run 2 showed why: a reply that
  nothing continues (a regenerate, an aborted request, or the check's own greedy probe) extends the
  chain and became its only kept snapshot, so the turn the conversation actually continues from lost
  its sliding state, mid-restore, and the request fell back to a 15.6k-token prefill. Cost: 63 MB per
  chain on disk. Two dead ends in a row still lose it.
