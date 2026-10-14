import pytest

from moestream.budget import (GB, KV_BYTES_PER_TOKEN, allocate, cap_budget_bytes, kv_bytes_per_token,
                              parse_vm_stat, slot_budget_bytes, uniform)

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   100.
Pages active:                                 999.
Pages inactive:                               20.
Pages speculative:                            3.
Pages throttled:                              0.
Pages wired down:                             777.
Pages purgeable:                              7.
"""


def test_parse_vm_stat_counts_reclaimable_pages():
    assert parse_vm_stat(VM_STAT) == (100 + 20 + 3 + 7) * 16384


def test_budget_is_min_of_free_and_gpu_cap_minus_reservations():
    other = 1 * GB + KV_BYTES_PER_TOKEN * 1000 + 1 * GB
    assert slot_budget_bytes(10 * GB, 20 * GB, 1 * GB, target_ctx=1000) == 10 * GB - 3 * GB - other
    assert slot_budget_bytes(30 * GB, 20 * GB, 1 * GB, target_ctx=1000) == 20 * GB - other
    assert slot_budget_bytes(1 * GB, 20 * GB, 1 * GB) == 0


def test_uniform_clamps_to_floor_and_experts():
    assert uniform([1, 2], 100) == {1: 50, 2: 50}
    assert uniform([1, 2], 4) == {1: 16, 2: 16}
    assert uniform([1, 2], 10**6) == {1: 256, 2: 256}


def test_allocate_favors_steep_curves():
    grid = [16, 32, 48]
    curves = {1: [0.9, 0.2, 0.1], 2: [0.5, 0.45, 0.4]}
    assert allocate(curves, grid, 48) == {1: 32, 2: 16}
    assert allocate(curves, grid, 64) == {1: 48, 2: 16}
    assert allocate(curves, grid, 10**6) == {1: 48, 2: 48}


def test_allocate_below_floor_gives_floor():
    assert allocate({1: [0.5, 0.1], 2: [0.5, 0.1]}, [16, 32], 5) == {1: 16, 2: 16}


def test_allocate_never_exceeds_total_above_floor():
    grid = [16, 32, 64, 128, 256]
    curves = {l: [1 / (1 + i * l) for i in range(5)] for l in range(1, 40)}
    for total in (16 * 39, 5000, 7000):
        assert sum(allocate(curves, grid, total).values()) <= total


def test_kv_bytes_per_token_by_bits():
    assert kv_bytes_per_token(0) == 40960
    assert kv_bytes_per_token(8) == 40960 * 136 // 256
    assert kv_bytes_per_token(4) == 40960 * 72 // 256


def test_cap_budget_subtracts_resident_kv_and_activations():
    assert cap_budget_bytes(20 * GB, 1 * GB, 1000) == 20 * GB - 1 * GB - 40960 * 1000 - 1 * GB


def test_fewer_kv_bits_leave_more_for_slots():
    b = [cap_budget_bytes(20 * GB, 1 * GB, 32768, kv_bits=k) for k in (0, 8, 4)]
    assert b[0] < b[1] < b[2]


def test_cap_below_fixed_costs_raises_listing_each_term():
    with pytest.raises(ValueError, match="resident.*KV.*activations"):
        cap_budget_bytes(2 * GB, 1 * GB, 32768)


def test_probe_budget_counts_kv_bits():
    assert slot_budget_bytes(30 * GB, 20 * GB, 1 * GB, target_ctx=1000, kv_bits=8) == \
        20 * GB - (1 * GB + kv_bytes_per_token(8) * 1000 + 1 * GB)
