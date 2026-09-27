"""How much memory the expert slots get, and how to split it across layers. Pure functions."""
import heapq
import re
import subprocess

GB = 1e9
KV_BYTES_PER_TOKEN = 40 * 1024  # 10 full-attention layers, bf16 K+V (lagunaLab harness spec)


def parse_vm_stat(text):
    page = int(re.search(r"page size of (\d+) bytes", text).group(1))
    pages = {m.group(1): int(m.group(2)) for m in re.finditer(r"Pages (\w+):\s+(\d+)\.", text)}
    return page * sum(pages.get(k, 0) for k in ("free", "inactive", "speculative", "purgeable"))


def slot_budget_bytes(available, working_set, resident, target_ctx=32768, headroom=3 * GB, activations=1 * GB):
    other = resident + KV_BYTES_PER_TOKEN * target_ctx + activations
    return int(max(0, min(available - headroom, working_set) - other))


def probe(resident, **kw):
    import mlx.core as mx

    vm = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    return slot_budget_bytes(parse_vm_stat(vm), mx.device_info()["max_recommended_working_set_size"], resident, **kw)


def uniform(layers, total, floor=16, num_experts=256):
    s = min(num_experts, max(floor, total // len(layers)))
    return {l: s for l in layers}


def allocate(curves, grid, total):
    """Greedy: every layer starts at grid[0]; each grid step goes to the layer whose miss rate drops most per slot."""
    at = {l: 0 for l in curves}
    left = total - grid[0] * len(curves)

    def push(heap, l):
        i = at[l]
        if i + 1 < len(grid):
            cost = grid[i + 1] - grid[i]
            heapq.heappush(heap, (-(curves[l][i] - curves[l][i + 1]) / cost, l, cost))

    heap = []
    for l in curves:
        push(heap, l)
    while heap and left > 0:
        _, l, cost = heapq.heappop(heap)
        if cost > left:
            continue  # ponytail: skips this layer for good; fine at 16-slot grid granularity
        at[l] += 1
        left -= cost
        push(heap, l)
    return {l: grid[i] for l, i in at.items()}
