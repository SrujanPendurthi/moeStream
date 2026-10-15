"""load(pack_dir) -> (model, tokenizer): Laguna with streamed experts sized to free memory."""
import json
import sys
import warnings
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from laguna_mlx import laguna
from moestream.budget import GB, allocate, cap_budget_bytes, free_bytes, probe, uniform
from moestream.config import EngineConfig
from moestream.layer import StreamingMoE


def _per_layer_slots(layers, E, stride, budget, cal, floor):
    total = int(budget // stride)
    if total < floor * len(layers):
        warnings.warn(f"slot budget {budget / GB:.2f} GB is below the floor ({floor} slots x {len(layers)} layers); "
                      "using the floor anyway")
    if cal is not None:
        grid = [min(E, g) for g in cal["grid"]]
        return allocate({int(l): c for l, c in cal["curves"].items()}, grid, total)
    return uniform(layers, total, min(floor, E), E)


def load(pack_dir, *, slots=None, budget_gb=None, memory_gb=None, calibration=None, half_life=None,
         target_ctx=32768, kv_bits=0, eviction="lfu", floor=16):
    if isinstance(pack_dir, EngineConfig):
        c = pack_dir
        # served: KV for kv_sessions sessions, by default the live one plus one stored conversation
        # cache (tools/serve.py caps those); 1 leaves other conversations to the SSD tier
        model, tok = load(c.pack_dir, memory_gb=c.memory_gb, calibration=c.calibration, half_life=c.half_life,
                          target_ctx=c.kv_sessions * c.context_tokens, kv_bits=c.kv_bits, eviction=c.eviction,
                          floor=floor)
        if not c.lookahead:  # the tuner's per-layer lookahead knob, all off
            for l in model.layers:
                if hasattr(l.mlp, "io"):
                    l.mlp.io.lookahead = False
        return model, tok
    pack_dir = Path(pack_dir)
    config = json.loads((pack_dir / "config.json").read_text())
    layout = json.loads((pack_dir / "layout.json").read_text())
    q = config["quantization"]
    E, stride, layers = layout["num_experts"], layout["stride"], layout["layers"]
    for l in layers:
        f = pack_dir / "experts" / f"layer_{l:02d}.bin"
        if not f.exists() or f.stat().st_size != E * stride:
            raise ValueError(f"{f}: missing or not {E * stride} bytes")

    cal = None
    if calibration is not None and Path(calibration).exists():
        cal = json.loads(Path(calibration).read_text())
        if sorted(int(l) for l in cal["curves"]) != layers:
            warnings.warn(f"calibration {calibration} does not match layers {layers}; using a uniform split")
            cal = None
    half_life = half_life or (cal["half_life"] if cal else 256.0)
    if cal is not None and eviction != "lfu":
        warnings.warn(f"calibration {calibration} was tuned for lfu; using its slot split with {eviction}")

    if slots is not None:
        per = {l: slots for l in layers}
    else:
        resident = (pack_dir / "resident.safetensors").stat().st_size
        if budget_gb is not None:
            budget = budget_gb * GB
        elif memory_gb is not None:
            budget = cap_budget_bytes(memory_gb * GB, resident, target_ctx, kv_bits)
            free = free_bytes()
            if free < memory_gb * GB:
                warnings.warn(f"memory_gb {memory_gb:g} GB but only {free / GB:.1f} GB is free; slot writes slow "
                              "about 17x once macOS pages (known good: 12.8 GB peak, 13.9 tok/s at 128 slots)")
        else:
            budget = probe(resident, target_ctx=target_ctx, kv_bits=kv_bits)
        per = _per_layer_slots(layers, E, stride, budget, cal, floor)

    args = laguna.ModelArgs.from_dict(config)
    model = laguna.Model(args)  # lazy: the full SwitchGLUs it builds are never evaluated
    dtype = getattr(mx, layout["arrays"]["gate_proj.scales"]["dtype"])
    moe = []
    for l in layers:
        model.layers[l].mlp = StreamingMoE(args, l, pack_dir, layout, per[l], half_life,
                                           q["group_size"], q["bits"], dtype, policy=eviction)
        moe.append(model.layers[l].mlp)
    for a, b in zip(moe, moe[1:]):
        a.io.next = b

    def predicate(p, m):
        return q[p] if p in q else hasattr(m, "to_quantized")

    nn.quantize(model, group_size=q["group_size"], bits=q["bits"], class_predicate=predicate)
    model.load_weights(str(pack_dir / "resident.safetensors"), strict=True)
    mx.eval(model.parameters())
    if kv_bits:
        from laguna_mlx.cache import make_quantized_cache

        gs = 64 if args.head_dim % 64 == 0 else 32
        make_quantized_cache(model, bits=kv_bits, group_size=gs)  # bad head_dim fails here, not mid-generation
        model.make_cache = lambda: make_quantized_cache(model, bits=kv_bits, group_size=gs)
    n = sum(per.values())
    print(f"moeStream: {n} slots ({n * stride / GB:.2f} GB), per layer {min(per.values())}-{max(per.values())}, "
          f"half-life {half_life:g}, eviction {eviction}, kv {kv_bits or 16}-bit", file=sys.stderr)
    tok = None
    if (pack_dir / "tokenizer.json").exists():
        from mlx_lm.utils import load_tokenizer
        tok = load_tokenizer(pack_dir)
    return model, tok
