"""FP8 (W8A16) decode-linear for the areno.accel surface.

Runs ``y = x @ (w_fp8 * scale)`` reading the 1-byte FP8 weight directly — the
memory-bound decode benefit. Two backends share the same W8A16 semantics:

* ``quantized_fp8_scaled_mm`` — Hopper (cc >= 8.9) via ``torch._scaled_mm``, the
  primary path (E4M3 grid, weight read once at 1 byte/elem).
* ``quantized_fp8_linear`` — Triton matmul (A100 fallback). Ampere Triton only
  accepts the E5M2 grid, so the E4M3 payload is converted to E5M2 in-kernel.
  ``num_stages`` pipelining, large ``BLOCK_K`` and the ``.cg`` cache modifier
  tune for the memory-bound decode shape, and the per-tensor scale is applied to
  the accumulator **after** the dot (``(x @ w_fp8) * scale == x @ (w_fp8 * scale)``).

Decode-only: the FP8 grid has no backward, so this must not enter a training graph.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quantized_fp8_linear_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    y_ptr,
    M,
    N,
    K,
    sx,
    sw,
    sy,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    scale = tl.load(scale_ptr)
    for k in range(0, tl.cdiv(K, BK)):
        rk = k * BK + tl.arange(0, BK)
        am = (rm[:, None] < M) & (rk[None, :] < K)
        bm_ = (rn[:, None] < N) & (rk[None, :] < K)
        a = tl.load(x_ptr + rm[:, None] * sx + rk[None, :], mask=am, other=0.0)
        b = tl.load(w_ptr + rn[:, None] * sw + rk[None, :], mask=bm_, other=0.0, cache_modifier=".cg")
        # FP8->fp16 tensor-core dot; per-tensor scale applied once after the dot.
        acc += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))
    acc = acc * scale
    tl.store(y_ptr + rm[:, None] * sy + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def quantized_fp8_linear(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    scale: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    block_m: int = 32,
    block_n: int = 128,
    block_k: int = 128,
    num_stages: int = 4,
    num_warps: int = 4,
) -> torch.Tensor:
    """FP8(E5M2) W8A16 linear forward: ``y = x @ (w_fp8 * scale)``.

    Args:
        x: activation, shape (batch, K), bf16.
        w_fp8: weight in FP8-E5M2, shape (N, K).
        scale: per-tensor scalar scale (float32), shape ().
        out: optional pre-allocated output (batch, N); reused if given, so the
            hot decode path does not pay a per-step ``torch.empty``.
    Returns:
        bf16 (batch, N); ``out`` if provided.
    """
    if not (x.is_cuda and w_fp8.is_cuda and scale.is_cuda):
        raise RuntimeError("quantized_fp8_linear requires CUDA inputs")
    M, K = x.shape
    N, _ = w_fp8.shape
    out_dtype = x.dtype
    y = (
        out
        if (out is not None and out.shape == (M, N) and out.device == x.device)
        else torch.empty((M, N), device=x.device, dtype=out_dtype)
    )
    if w_fp8.dtype != torch.float8_e5m2:
        # Many checkpoints store E4M3; Triton on Ampere only accepts E5M2, so
        # convert to the grid the kernel can consume.
        w_fp8 = w_fp8.to(torch.float8_e5m2)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _quantized_fp8_linear_kernel[grid](
        x,
        w_fp8,
        scale,
        y,
        M,
        N,
        K,
        x.stride(0),
        w_fp8.stride(0),
        y.stride(0),
        block_m,
        block_n,
        block_k,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return y


def dequantize_fp8_weight(w_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Materialize a bf16 weight from an FP8 weight + per-tensor scale."""
    return (w_fp8.float() * scale.to(torch.float32)).to(torch.bfloat16)


def scaled_mm_available() -> bool:
    """``torch._scaled_mm`` (FP8 E4M3) needs Hopper/Ada (compute capability >= 8.9)."""
    if not hasattr(torch, "_scaled_mm") or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) >= (8, 9)


def quantized_fp8_scaled_mm(x: torch.Tensor, w_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Hopper FP8 linear via ``torch._scaled_mm``: ``y = x @ (e4m3(w) * scale)^T``.

    ``x`` is bf16 (M, K); ``w_fp8`` is E4M3 (N, K) with a per-tensor scalar
    ``scale``. ``torch._scaled_mm`` takes mat1 (M, K) and mat2 (K, N), so the
    weight is transposed to (K, N); ``scale_b`` scales the FP8 operand (the bf16
    activation scale is identity). Dequantizing the weights first would defeat the
    memory-bound win, so this reads the bytes directly.
    """
    if not (x.is_cuda and w_fp8.is_cuda and scale.is_cuda):
        raise RuntimeError("quantized_fp8_scaled_mm requires CUDA inputs")
    w_t = w_fp8.t().contiguous()
    return torch._scaled_mm(x, w_t, out_dtype=torch.bfloat16, scale_b=scale)


def mark_fp8_weight(weight: torch.Tensor, *, fp8_dtype=torch.float8_e4m3fn) -> torch.Tensor:
    """Quantize a bf16 weight in place and stash the FP8 payload for the linear hook.

    Sets ``weight._areno_fp8`` (FP8 grid tensor) and ``weight._areno_fp8_scale``
    (per-tensor scale) so ``_areno_linear_forward`` dispatches to the FP8 decode
    path. Returns ``weight``. Default grid is E4M3 (the finer grid used by
    ``torch._scaled_mm`` on Hopper); the A100 Triton fallback converts to E5M2 at
    kernel time. Per-tensor scale only (both kernels read a single scalar).
    """
    from areno.engine.quantization import compute_fp8_scale

    max_v = 57344.0 if fp8_dtype is torch.float8_e5m2 else 448.0
    f = weight.detach().float()
    scale = compute_fp8_scale(f, max_val=max_v)
    weight._areno_fp8 = (f / scale).clamp(-max_v, max_v).to(fp8_dtype)
    weight._areno_fp8_scale = scale
    return weight
