"""FP8 decode-linear for the areno.accel surface (target: H20 / Hopper).

Runs the dequant-linear ``y = x @ (e4m3(w) * scale)`` via ``torch._scaled_mm``
on Hopper (cc >= 8.9, e.g. H20). cuBLASLt reads the 1-byte FP8 weight directly,
which is the decode memory-bound win: on Qwen3-8B MLP shapes this is ~2x per
linear and ~1.6x end-to-end per MLP block under CUDA graphs, vs a bf16 baseline.

``_scaled_mm`` is A8W8 (it rejects a bf16 activation), so the activation is also
quantized to E4M3 — a W8A8 decode, not weight-only. A fused Triton
``_quantize_act`` kernel keeps that per-call quantize cheap. Only applies on
Hopper; elsewhere fp8-marked weights fall through to the bf16 path.

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


def mark_fp8_weight(weight: torch.Tensor, *, fp8_dtype=torch.float8_e4m3fn) -> torch.Tensor:
    """Quantize a bf16 weight in place and stash the FP8 payload for the linear hook.

    Sets ``weight._areno_fp8`` (FP8 grid tensor) and ``weight._areno_fp8_scale``
    (per-tensor scale) so ``_areno_linear_forward`` dispatches to the FP8 decode
    path on Hopper. Returns ``weight``. Per-tensor scale only; the grid is E4M3
    (the ``torch._scaled_mm`` grid).
    """
    from areno.engine.quantization import compute_fp8_scale

    max_v = 57344.0 if fp8_dtype is torch.float8_e5m2 else 448.0
    f = weight.detach().float()
    scale = compute_fp8_scale(f, max_val=max_v)
    weight._areno_fp8 = (f / scale).clamp(-max_v, max_v).to(fp8_dtype)
    weight._areno_fp8_scale = scale
    return weight
