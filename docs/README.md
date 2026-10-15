# moeStream documentation

moeStream runs Laguna-XS-2.1, a 33B mixture-of-experts model, on a 24 GB Apple Silicon Mac by keeping
only the experts in use in memory and streaming the rest from SSD. It serves an OpenAI-compatible API
(used by the pi coding agent) and tunes its own cache settings while it serves.

| Page | Read it when you want to |
|---|---|
| [Architecture](architecture.md) | understand how a token flows through the engine, and why each piece exists |
| [Configuration](configuration.md) | change memory, context, KV precision, eviction or tuning; work out what fits |
| [Serving and pi](serving.md) | run the server, connect pi, verify a deployment |
| [Self-tuning runtime](self-tuning.md) | understand what the engine tunes on its own, how it decides, and how to read its log |
| [Tools](tools.md) | record routing, calibrate, benchmark, drive load, watch memory |
| [Prefill memory](prefill-memory.md) | what a prefill allocates, why the study runs stopped, and how the next run is built |
| [Development](development.md) | run tests, follow the shared-GPU rules, know what may be imported from lagunaLab |

Measured results (benchmarks, the speed gate, the pi check) live in the top-level
[README](../README.md) next to the files in `results/` they come from. Design documents are in
[`specs/`](specs/), implementation plans in [`superpowers/plans/`](superpowers/plans/).

## The model in numbers

| | |
|---|---|
| Layers | 40: layer 0 dense, layers 1-39 mixture-of-experts |
| Experts | 256 per MoE layer, 8 routed per token, plus one shared expert |
| Expert weights | 4-bit (group size 64), 1,769,472 bytes per expert; 453 MB per layer, 17.7 GB on SSD in all |
| Resident weights | 1.16 GB: attention, embeddings, routers (8-bit), shared experts, layer 0 |
| Attention | 10 full-attention layers plus sliding-window layers (window 512) |
| KV cache | 40,960 bytes per token at bf16; 21,760 at q8 (the default); 11,520 at q4 |
| Context | up to 262,144 tokens (the model); 131,072 configured |
| Vocabulary | 100,352 tokens |

## Status (2026-10-14)

| Sub-project | State |
|---|---|
| 0. Speed gate | rewritten to what is measured: parity with lagunaLab at 14.4 GB, both MLX buffer pools capped at 1 GB (short decode 1.01x, 16k decode 1.04x, prefill 0.81x); the earlier uncapped numbers are superseded |
| 1. Engine config, server, pi | done: multi-turn pi session with tool calls verified at 14 GB |
| 4. Self-tuning (system level) | built and tested; its real-weight gate is not met, so it ships off (`tune = false`). SSD tier for the prompt cache: verified on real weights, 3.7x less prompt work and 2.8x faster on three interleaved sessions |
| 2. Custom harness and eval | next; needs its own spec |
| 3. Control UI, 6. GRPO training, 5. Refine loop | later |
