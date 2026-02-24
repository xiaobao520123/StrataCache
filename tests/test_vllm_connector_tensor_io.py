from __future__ import annotations


def run() -> None:
    try:
        import torch
    except Exception:
        # Optional test: only runs when torch is installed.
        return

    from stratacache.adapters.vllm.connector_v1 import _gather_by_slots, _scatter_by_slots

    device = torch.device("cpu")

    # Simulate a common kv_layer shape: [2, num_blocks, block_size, num_heads, head_size]
    kv_layer = torch.randn(2, 4, 8, 2, 4, device=device)

    # Build a slot_mapping selecting a few arbitrary token slots.
    # Flattened slots = num_blocks*block_size = 32
    slots = torch.tensor([0, 3, 7, 8, 15, 31], dtype=torch.long)

    gathered = _gather_by_slots(kv_layer, slots)
    assert gathered.shape[0] == 2
    assert gathered.shape[1] == slots.numel()

    # Zero and scatter back, then verify exact match at selected slots.
    kv2 = torch.zeros_like(kv_layer)
    _scatter_by_slots(kv2, slots, gathered)

    flat1 = kv_layer.reshape(2, -1, *kv_layer.shape[3:])
    flat2 = kv2.reshape(2, -1, *kv2.shape[3:])
    assert torch.allclose(flat1[:, slots], flat2[:, slots])

