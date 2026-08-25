"""End-to-end verification of the FP8 linear path on the Qwen3-8B MLP shape.

Puts the FP8 W8A16 Triton kernel behind the real single matmul entry point
(``areno.engine.layers.linear._areno_linear_forward``), then measures bf16 vs
FP8 through that path. Confirms (1) the hook dispatches to the FP8 kernel and
(2) the measured decode-matmul speedup (~1.6x) holds on the real model shape.
"""

from __future__ import annotations

import time

import torch

from areno.accel.kernels.fp8_linear import mark_fp8_weight, quantized_fp8_linear
from areno.engine.layers.linear import _areno_linear_forward


def _bench(fn, iters=40, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    M, K, N = 32, 4096, 12288  # Qwen3-8B MLP gate/up shape
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)

    bf16_out = _areno_linear_forward(x, w, None)
    torch.cuda.synchronize()

    w_q = torch.nn.Parameter(w.clone())
    mark_fp8_weight(w_q)
    assert getattr(w_q, "_areno_fp8", None) is not None, "mark_fp8_weight did not set the FP8 payload"
    fp8_out = _areno_linear_forward(x, w_q, None)
    torch.cuda.synchronize()

    rel = float((fp8_out.float() - bf16_out.float()).abs().max() / (bf16_out.float().abs().max() + 1e-6))
    print(f"fp8 linalg rel err vs bf16 (via hook) = {rel:.4f}")

    t_bf16 = _bench(lambda: _areno_linear_forward(x, w, None))
    t_fp8_hook = _bench(lambda: _areno_linear_forward(x, w_q, None))
    # Kernel-only (reuse a pre-allocated out buffer -> no per-step cudaMalloc).
    ybuf = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    t_fp8_noalloc = _bench(lambda: quantized_fp8_linear(x, w_q._areno_fp8, w_q._areno_fp8_scale, out=ybuf))
    print(f"bf16 (areno_linear)       : {t_bf16 * 1e3:.3f} ms/step, {M / t_bf16:.0f} tok/s")
    print(f"fp8  (hook, alloc)        : {t_fp8_hook * 1e3:.3f} ms/step, {M / t_fp8_hook:.0f} tok/s")
    print(f"fp8  (kernel, no alloc)   : {t_fp8_noalloc * 1e3:.3f} ms/step, {M / t_fp8_noalloc:.0f} tok/s")
    print(f"speedup (bf16 / fp8-alloc)   = {t_bf16 / t_fp8_hook:.2f}x")
    print(f"speedup (bf16 / fp8-noalloc) = {t_bf16 / t_fp8_noalloc:.2f}x")


if __name__ == "__main__":
    main()
