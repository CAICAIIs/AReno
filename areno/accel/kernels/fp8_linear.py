"""Triton FP8 (W8A16) dequant-linear for the areno.accel surface.

Runs ``y = x @ (w_fp8 * scale)`` (``x`` bf16, ``w_fp8`` FP8-E5M2, ``scale`` per-tensor
scalar) with a Triton matmul that reads the FP8 weight directly — the memory-bound
decode benefit from RFC 0001. Tunings reflect the measured best on A100 (~1.62x vs
bf16 at decode shapes): ``num_stages`` pipelining, large ``BLOCK_K``, ``.cg`` on the
streamed weight, and the key trick of applying the per-tensor scale to the
accumulator **after** the dot (since ``(x @ w_fp8) * scale == x @ (w_fp8 * scale)``),
which avoids a per-element dequant in the inner loop.

This is an inference-side (decode) op. A backwards pass is out of scope for now;
training integration is a separate piece.
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


def mark_fp8_weight(weight: torch.Tensor, *, group_size: int = -1, fp8_dtype=torch.float8_e5m2) -> torch.Tensor:
    """Quantize a bf16 weight in place and stash the FP8 payload for the linear hook.

    Sets ``weight._areno_fp8`` (FP8 grid tensor) and ``weight._areno_fp8_scale``
    (per-tensor scale) so ``_areno_linear_forward`` dispatches to
    :func:`quantized_fp8_linear`. Returns ``weight``. ``group_size <= 0`` means a
    single per-tensor scale (the current kernel path).
    """
    f = weight.detach().float()
    max_v = 57344.0 if fp8_dtype is torch.float8_e5m2 else 448.0
    scale = (f.abs().amax() / max_v).to(torch.float32).reshape(())
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = (f / scale).clamp(-max_v, max_v).to(fp8_dtype)
    weight._areno_fp8 = q
    weight._areno_fp8_scale = scale
    return weight
