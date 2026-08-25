"""FP8 weight quantization helpers (W8A16, dequant-forward reference).

These helpers implement the FP8 (E4M3) quantize/dequant math in a dtype-agnostic
way so they run on CPU (for unit tests) as well as on GPU. They are the
correctness reference for the FP8 vertical slice (RFC 0001, M1): the intended
memory-bandwidth speedup comes from a fused FP8-dequant matmul that reads the
1-byte weights directly; the dequant-forward path here establishes numerical
correctness and the exact scale semantics before that kernel lands.

Note on FP8 grids: this module (and ``QuantizedLinear``) uses **E4M3** as the
reference grid. The model's runtime FP8 path in ``fp8_linear.py`` uses **E5M2**,
because on Ampere (A100) Triton only accepts the E5M2 grid and rejects E4M3
(``fp8e4nv``); see RFC 0001 §4.6. The fused E4M3 dequant kernels live in
``areno/accel/csrc/e4m3_linear.cu``. ``compute_fp8_scale``/``quantize_to_fp8``
are dtype-agnostic apart from the constant ``_FP8_E4M3_MAX``=448.

This is deliberately small and dependency-free (only torch).
"""

from __future__ import annotations

import torch

# E4M3 has 3 exponent bits + 4 mantissa bits + sign. Max finite magnitude is
# 448.0; magnitude range is [2^-9, 448]. We clamp to the representable range.
_FP8_E4M3_MAX = 448.0


def compute_fp8_scale(weight: torch.Tensor, group_size: int = -1) -> torch.Tensor:
    """Compute the FP8 scale for `weight`, per-tensor or per-group.

    ``group_size <= 0`` (default) uses a single per-tensor scale. Otherwise the
    weight's last dim is split into groups of ``group_size`` and a scale is
    computed per group. Scale is chosen so the max magnitude in the group maps
    to ``_FP8_E4M3_MAX``; a zero group falls back to 1.0 to avoid inf/nan.
    """
    w = weight.detach().float()
    if group_size is None or group_size <= 0:
        amax = w.abs().amax()
        scale = amax / _FP8_E4M3_MAX
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return scale
    n = w.numel()
    g = int(group_size)
    if n % g != 0:
        raise ValueError(f"weight numel {n} must be divisible by group_size {g}")
    flat = w.reshape(-1, g)
    amax = flat.abs().amax(dim=1, keepdim=True)
    scale = amax / _FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale.reshape(-1)


def quantize_to_fp8(weight: torch.Tensor, group_size: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `weight` (bf16/fp32) to FP8 grid; returns (fp8, scale).

    ``fp8`` is returned in ``torch.float8_e4m3fn`` when this torch build supports
    it; otherwise it is returned as a float tensor already snapped to the FP8
    grid (so the math is testable on CPU). ``scale`` is float32, shaped per the
    group layout (scalar for per-tensor, ``(n/g,)`` for per-group).
    """
    scale = compute_fp8_scale(weight, group_size)
    w = weight.detach().to(torch.float32)
    if group_size is None or group_size <= 0:
        snapped = torch.clamp((w / scale).round(), -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    else:
        low = w.reshape(-1, int(group_size))
        scale2 = scale.view(-1, 1)
        snapped = torch.clamp((low / scale2).round(), -_FP8_E4M3_MAX, _FP8_E4M3_MAX)
        snapped = snapped.reshape_as(w)
    if hasattr(torch, "float8_e4m3fn"):
        try:
            return snapped.to(torch.float8_e4m3fn), scale
        except (RuntimeError, TypeError, ValueError):
            pass
    return snapped, scale


def dequant_fp8(fp8: torch.Tensor, scale: torch.Tensor, group_size: int = -1) -> torch.Tensor:
    """Dequantize FP8 weights back to float; opposite of :func:`quantize_to_fp8`.

    ``scale`` is per-tensor (scalar) or per-group (``(n/g,)``) and broadcasts
    down to the weight shape. Output dtype is float32 (caller casts).
    """
    q = fp8.float()
    if group_size is None or group_size <= 0:
        return q * scale
    return (q.reshape(-1, int(group_size)) * scale.view(-1, 1)).reshape_as(q)
