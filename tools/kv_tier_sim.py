"""Would an SSD tier for the prompt (prefix) KV cache pay off? Token-level simulation, no model load.

PYTHONPATH=.:../lagunaLab python3 -m tools.kv_tier_sim [--out results/kv_tier_sim.json]

Each captured session is rendered through the chat template turn by turn, as pi would send it:
turn i's prompt is everything before assistant turn i, and after the turn the server holds the
conversation through that assistant reply. This assumes live use, where the reply in the history is
the one the model generated (a replay of recorded replies would not reuse anything).
The sliding-window layers cannot rewind, so a cache entry is reusable only when it is an exact prefix
of the new prompt (mlx-lm's rule for non-trimmable caches); snapshots are taken at turn ends.
"""
import argparse
import glob
import json
from collections import OrderedDict
from pathlib import Path

from moestream.budget import kv_bytes_per_token

PREFILL_TOK_S = 100.0   # measured 93-98 tok/s at 16k context; slower at 60k+, so this is generous to re-prefill
SSD_BYTES_S = 2e9       # sustained read into preallocated buffers; conservative for the M4's internal SSD
SLIDING_BYTES = 30 * 512 * 4096  # 30 sliding layers x 512-token window, bf16, stored whole in every snapshot


def lcp(a, b):
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


def session_turns(msgs, tools, tok):
    """[(prompt_tokens, held_tokens_after, prev_held_is_prefix)] per assistant turn."""
    out, held = [], None
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or i == 0:
            continue
        p = tok.apply_chat_template(msgs[:i], tools=tools, add_generation_prompt=True, tokenize=True)
        c = tok.apply_chat_template(msgs[: i + 1], tools=tools, tokenize=True)
        out.append((len(p), len(c), held is not None and lcp(held, p) == len(held)))
        held = c
    return out


def simulate(sessions, order, ram_tokens, max_entries=6, ssd_tokens=None, restart_at=None):
    """Replay requests through an LRU RAM prompt cache (one entry per session, capped in tokens and
    entries) and an optional SSD tier holding each session's latest turn-end snapshot.
    order: [(session, turn)]. Returns token counts."""
    ram, ssd = OrderedDict(), OrderedDict()
    tot = dict(prompt=0, from_ram=0, from_ssd=0, recomputed=0, ssd_written=0, snapshots=0)
    for step, (s, i) in enumerate(order):
        if step == restart_at:
            ram.clear()  # the server restarts; the SSD tier survives
        p, c, chained = sessions[s][i]
        reuse = 0
        if chained and s in ram:
            reuse = ram.pop(s)
            tot["from_ram"] += reuse
        elif chained and ssd_tokens is not None and s in ssd:
            reuse = ssd[s]
            tot["from_ssd"] += reuse
        tot["prompt"] += p
        tot["recomputed"] += p - reuse
        ram.pop(s, None)
        ram[s] = c
        while ram and (sum(ram.values()) > ram_tokens or len(ram) > max_entries):
            ram.popitem(last=False)
        if ssd_tokens is not None:
            prev = ssd.pop(s, 0)
            ssd[s] = c
            tot["ssd_written"] += c - prev if chained else c  # full-attention KV is append-only
            tot["snapshots"] += 1
            while ssd and sum(ssd.values()) > ssd_tokens:
                ssd.popitem(last=False)
    return tot


def interleave(sessions, group):
    """Round-robin the turns of `group` sessions at a time, groups one after another."""
    order, ids = [], list(range(len(sessions)))
    for g in range(0, len(ids), group):
        live = {s: 0 for s in ids[g : g + group]}
        while live:
            for s in list(live):
                order.append((s, live[s]))
                live[s] += 1
                if live[s] == len(sessions[s]):
                    del live[s]
    return order


def seconds(t, kv_bits):
    restore = t["from_ssd"] * kv_bytes_per_token(kv_bits) + t.get("restores", 0) * SLIDING_BYTES
    return t["recomputed"] / PREFILL_TOK_S + restore / SSD_BYTES_S


def main():
    from mlx_lm.utils import load_tokenizer
    from replay.session import load_session

    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="../lagunaLab/traces")
    ap.add_argument("--pack", default="../lagunaLab/models/Laguna-XS-2.1-packed")
    ap.add_argument("--out", default="results/kv_tier_sim.json")
    ap.add_argument("--kv-bits", type=int, default=8)
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--ssd-gb", type=float, default=20.0)
    a = ap.parse_args()
    tok = load_tokenizer(Path(a.pack))
    files = sorted(glob.glob(f"{a.traces}/*.jsonl"))
    sessions = [session_turns(*load_session(f), tok) for f in files]
    bpt = kv_bytes_per_token(a.kv_bits)
    ram_tokens = a.context  # tools/serve.py caps stored caches at one session of KV
    ssd_tokens = int(a.ssd_gb * 1e9 / bpt)
    rows = []
    for name, group, restart in [("one session at a time", 1, False), ("one session + restart", 1, True), ("2 interleaved", 2, False),
                                 ("3 interleaved", 3, False), ("3 interleaved + restart", 3, True)]:
        order = interleave(sessions, group)
        r = len(order) // 2 if restart else None
        for tier, ssd in (("ram only", None), ("ram + ssd", ssd_tokens)):
            t = simulate(sessions, order, ram_tokens, ssd_tokens=ssd, restart_at=r)
            rows.append(dict(workload=name, tier=tier, **t, est_prompt_s=round(seconds(t, a.kv_bits)),
                             ssd_written_gb=round(t["ssd_written"] * bpt / 1e9 + t["snapshots"] * SLIDING_BYTES / 1e9, 1)))
    out = dict(sessions=[dict(file=Path(f).name, turns=len(s), final_tokens=s[-1][1]) for f, s in zip(files, sessions)],
               kv_bits=a.kv_bits, ram_cap_tokens=ram_tokens, ssd_cap_gb=a.ssd_gb, prefill_tok_s=PREFILL_TOK_S,
               ssd_bytes_s=SSD_BYTES_S, rows=rows)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f"{r['workload']:26} {r['tier']:10} prompt {r['prompt']:>9} recomputed {r['recomputed']:>9} "
              f"ram {r['from_ram']:>9} ssd {r['from_ssd']:>9}  est {r['est_prompt_s'] / 60:6.1f} min  "
              f"written {r['ssd_written_gb']} GB")


if __name__ == "__main__":
    main()
