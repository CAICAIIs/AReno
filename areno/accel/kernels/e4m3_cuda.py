"""GPU E4M3 fused dequant-linear via the areno.accel CUDA extension.

Reads the 1-byte uint8 E4M3 weight payload and decodes it to fp16 in-kernel
(the branchless bit-trick in ``areno/accel/csrc/e4m3_linear.cu``) so the weight
bytes pulled from HBM are halved vs the bf16 path — the memory-bandwidth benefit
from RFC 0001. The reference decode math is validated against
``torch.float8_e4m3fn`` in ``tests/test_e4m3_decode_cpu.py``. This is a
standalone Ampere (cc 8.0) decode kernel; it is **not** wired into
``_areno_linear_forward`` (the model FP8 decode path uses the E5M2 Triton kernel
in ``fp8_linear.py`` on A100 and the native ``torch._scaled_mm`` on Hopper/H20).

Forward-only: E4M3 has no backward, so this is a decode/inference operator and
must not be wired into a training graph.
"""

from __future__ import annotations

import torch

from areno.accel._extension import extension as _extension


def quantized_e4m3_linear_cuda(
    x: torch.Tensor,
    w_u8: torch.Tensor,
    scale: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``y = x @ (e4m3(w) * scale)^T`` reading the uint8 E4M3 weight directly.

    Args:
        x: activation, shape (..., K), bf16 (or fp16).
        w_u8: weight as packed E4M3 bytes, shape (N, K), ``torch.uint8``.
        scale: per-tensor scalar (float32), shape ().
        out: optional pre-allocated bf16 output (..., N); reused if given.
    Returns:
        bf16 (..., N); ``out`` if provided.
    """
    if not (x.is_cuda and w_u8.is_cuda and scale.is_cuda):
        raise RuntimeError("quantized_e4m3_linear_cuda requires CUDA inputs")
    if w_u8.dtype != torch.uint8:
        raise RuntimeError(f"quantized_e4m3_linear_cuda weight must be uint8, got {w_u8.dtype}")
    x2 = x.contiguous()
    w2 = w_u8.contiguous()
    s2 = scale.contiguous()
    # Pass the caller's buffer through so the extension writes into it directly and
    # the hot decode path avoids a per-call torch::empty + copy_.
    return _extension().areno_e4m3_linear_forward(x2, w2, s2, out)


def quantize_weight_e4m3(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a bf16 weight to E4M3 and return (uint8 bytes, scalar scale).

    Uses a per-tensor scale so -448, 448 (or symmetric) maps to the grid; matches
    the scale semantics in ``areno.engine.quantization``.
    """
    from areno.engine.quantization import quantize_to_fp8

    fp8, scale = quantize_to_fp8(weight, group_size=-1)
    return fp8.view(torch.uint8).contiguous(), scale.contiguous()


def dequant_e4m3_bf16(w_u8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Materialize a bf16 weight from an E4M3 uint8 payload + scale."""
    return (w_u8.view(torch.float8_e4m3fn).float() * scale.float()).to(torch.bfloat16)
