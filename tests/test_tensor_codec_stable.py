from __future__ import annotations


def run() -> None:
    try:
        import torch
    except Exception:
        return

    from stratacache.adapters.vllm.connector_v1 import _decode_tensor_stable, _encode_tensor_stable

    for dt in (torch.float16, torch.bfloat16, torch.float32):
        t = torch.randn(2, 3, 4, dtype=dt)
        b = _encode_tensor_stable(t)
        out = _decode_tensor_stable(b, device=torch.device("cpu"))
        assert out.shape == t.shape
        assert out.dtype == t.dtype
        assert torch.allclose(out, t.cpu())

