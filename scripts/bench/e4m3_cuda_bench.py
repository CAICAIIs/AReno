"""GPU bench + correctness check for the E4M3 fused dequant-GEMM kernel.

Validates the CUDA kernel against the bf16 reference across representative
shapes and times it vs the bf16 ``areno_linear`` cuBLAS path on decode-like
shapes. Run on a free GPU:

    CUDA_VISIBLE_DEVICES=2 python scripts/bench/e4m3_cuda_bench.py
"""

from __future__ import annotations

import torch

from areno.accel import areno_linear
from areno.accel.kernels.e4m3_cuda import (
    dequant_e4m3_bf16,
    quantize_weight_e4m3,
    quantized_e4m3_linear_cuda,
)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.float() - b.float()).abs().max().item()
    denom = b.float().abs().max().item() + 1e-6
    return d / denom


def bench(fn, iters: int = 200, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "need a GPU"

    print("=== correctness vs bf16 reference ===")
    for M, N, K in [(1, 4096, 4096), (1, 12288, 4096), (4, 12288, 4096), (64, 12288, 4096)]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K**0.5)
        w_u8, scale = quantize_weight_e4m3(w)
        out = quantized_e4m3_linear_cuda(x, w_u8, scale)
        dq = dequant_e4m3_bf16(w_u8, scale)
        ref = x @ dq.T
        print(
            f"  M={M:3d} N={N:5d} K={K:5d}: shape={tuple(out.shape)} "
            f"dtype={out.dtype} rel_vs_dequant={rel_err(out, ref):.5f} "
            f"finite={bool(torch.isfinite(out).all())}"
        )

    print("\n=== decode throughput vs bf16 areno_linear ===")
    print(f"{'M':>4} {'N':>6} {'K':>6} {'bf16(ms)':>10} {'e4m3(ms)':>10} {'speedup':>8} {'e4m3GB/s':>9}")
    for M, N, K in [(1, 12288, 4096), (1, 8192, 4096), (4, 12288, 4096), (64, 12288, 4096)]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K**0.5)
        w_u8, scale = quantize_weight_e4m3(w)
        t_bf16 = bench(lambda: areno_linear(x, w, None))
        t_e4m3 = bench(lambda: quantized_e4m3_linear_cuda(x, w_u8, scale))
        mb = N * K * 1 / 1e6  # uint8 weight MB
        gbs = mb * 1e6 / (t_e4m3 * 1e-3 * 1e9)
        print(f"{M:4d} {N:6d} {K:6d} {t_bf16:10.4f} {t_e4m3:10.4f} {t_bf16 / t_e4m3:8.3f}x {gbs:9.1f}")


if __name__ == "__main__":
    main()
