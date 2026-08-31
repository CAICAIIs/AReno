"""FP8 (W8A16) decode-linear for the areno.accel surface (target: H20 / Hopper).

Two backends implement the dequant-linear ``y = x @ (e4m3(w) * scale)``:

* ``quantized_fp8_scaled_mm`` — Hopper (cc >= 8.9, e.g. H20) via ``torch._scaled_mm``.
  This is the decode fast path: cuBLASLt reads the 1-byte FP8 weight directly
  (~2x, and more accurate, than the Triton path). ``_scaled_mm`` is A8W8 (it
  rejects a bf16 activation), so the activation is also quantized to E4M3 — a
  W8A8 decode, not weight-only.
* ``quantized_fp8_linear`` — Triton matmul, kept as a non-Hopper fallback. It is
  weight-only (bf16 activation); on this stack it measures *slower* than bf16, so
  it is a correctness path rather than a speedup.

The target host for the decode speedup is H20 (Hopper, cc 9.0): on Qwen3-8B MLP
shapes the A8W8 ``_scaled_mm`` path is ~1.6x end-to-end per MLP block (2x per
linear), compared to a bf16 baseline, under CUDA graphs.

Decode-only: FP8 has no backward, so this must not enter a training graph.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def scaled_mm_available() -> bool:
    """``torch._scaled_mm`` (FP8 A8W8) needs Hopper/Ada (compute capability >= 8.9)."""
    if not hasattr(torch, "_scaled_mm") or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) >= (8, 9)


# Largest numel the single-launch activation quantizer can cover. Decode
# activations are tiny (M x K), so a single block computes the amax and quantizes
# in one kernel — measured ~2.2x with the FP8 matmul vs ~1.2x with the torch
# op sequence. Bigger (prefill) tensors fall back to the torch path.
_KQUANT_MAX_NUMEL = 1 << 15


@triton.jit
def _quantize_act_kernel(x_ptr, scale_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < numel
    v = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    amax = tl.max(tl.where(m, v, float("-inf")), axis=0)
    scale = amax / 448.0
    scale = tl.where(scale > 0, scale, 1.0)
    tl.store(scale_ptr, scale)
    tl.store(out_ptr + offs, tl.clamp(v / scale, -448.0, 448.0).to(tl.float8e4nv), mask=m)


def _quantize_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor E4M3 quantize of the activation -> (fp8, scale).

    For small (decode) activations this fuses the amax + scale + quantize into a
    single Triton kernel (the decode fast path); larger tensors use the torch
    op sequence.
    """
    numel = x.numel()
    if numel <= _KQUANT_MAX_NUMEL:
        block = 1024
        while block < numel:
            block *= 2
        out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty((), dtype=torch.float32, device=x.device)
        _quantize_act_kernel[(1,)](x, scale, out, numel, BLOCK=block)
        return out, scale
    amax = x.abs().amax()
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax)).to(torch.float32)
    return (x.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale


def quantized_fp8_scaled_mm(x: torch.Tensor, w_fp8: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """Hopper FP8 decode linear via ``torch._scaled_mm``: ``y = x @ (e4m3(w) * scale)^T``.

    ``x`` is bf16 (M, K) and is quantized to E4M3 here (A8W8); ``w_fp8`` is the
    E4M3 weight (N, K) with per-tensor ``w_scale``. cuBLASLt wants mat2 in a
    column-major (K, N) view, so the weight is transposed with ``w.t()`` (NOT
    ``.contiguous()``). Reads the weight bytes directly — the decode win.
    """
    if not (x.is_cuda and w_fp8.is_cuda and w_scale.is_cuda):
        raise RuntimeError("quantized_fp8_scaled_mm requires CUDA inputs")
    xq, x_scale = _quantize_act(x.contiguous())
    return torch._scaled_mm(xq, w_fp8.t(), x_scale, w_scale, out_dtype=torch.bfloat16)


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


def mark_fp8_weight(weight: torch.Tensor, *, fp8_dtype=torch.float8_e4m3fn) -> torch.Tensor:
    """Quantize a bf16 weight in place and stash the FP8 payload for the linear hook.

    Sets ``weight._areno_fp8`` (FP8 grid tensor) and ``weight._areno_fp8_scale``
    (per-tensor scale) so ``_areno_linear_forward`` dispatches to the FP8 decode
    path. Returns ``weight``. The default grid is E4M3 (finer, and the grid
    ``torch._scaled_mm`` needs on Hopper); the Ampere Triton kernel converts to
    E5M2 as required. Per-tensor scale only.
    """
    from areno.engine.quantization import compute_fp8_scale

    max_v = 57344.0 if fp8_dtype is torch.float8_e5m2 else 448.0
    f = weight.detach().float()
    scale = compute_fp8_scale(f, max_val=max_v)
    weight._areno_fp8 = (f / scale).clamp(-max_v, max_v).to(fp8_dtype)
    weight._areno_fp8_scale = scale
    return weight
