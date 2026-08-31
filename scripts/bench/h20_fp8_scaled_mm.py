"""Validate E4M3 FP8 decode speedup on Hopper (H20) via torch._scaled_mm.

The H20 (cc 9.0) supports the built-in FP8 matmul. For the memory-bound decode
case, the weight is read as fp8_e4m3 (1 byte) so the weight-read bytes halve vs
bf16. This compares `torch._scaled_mm` on FP8 vs the bf16 cuBLAS matmul at the
MLP decode shape, and validates correctness against a bf16 reference.
"""

from __future__ import annotations

import torch

torch.manual_seed(0)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-6))


def best(fn, reps=100, warm=20) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    bv = 1e9
    for _ in range(reps):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        bv = min(bv, s.elapsed_time(e))
    return bv


def quant_fp8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor FP8 E4M3 quantize; returns (fp8, scale)."""
    amax = t.abs().amax()
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    q = (t / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32)


def main() -> None:
    assert torch.cuda.is_available()
    cc = torch.cuda.get_device_capability(0)
    if not hasattr(torch, "_scaled_mm") or cc < (8, 9):
        raise RuntimeError(
            f"torch._scaled_mm (FP8 E4M3 matmul) requires Hopper/Ada (cc >= 8.9); got cc={cc}. "
            "This benchmark is Hopper/H20-only."
        )
    print(f"torch {torch.__version__} cc {cc[0]}.{cc[1]} gpus {torch.cuda.device_count()}")

    for M, N, K in [(1, 12288, 4096), (1, 8192, 4096), (4, 12288, 4096), (64, 12288, 4096)]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / K**0.5

        # --- bf16 baseline (reads 2 bytes/weight) ---
        t_bf16 = best(lambda: torch.mm(x, w.T))

        # --- FP8 W8A8 via _scaled_mm (reads 1 byte/weight) ---
        xq, xs = quant_fp8(x)
        wq, ws = quant_fp8(w)
        # _scaled_mm(a,b): a (M,K) fp8, b (K,N) fp8 -> (M,N). w is (N,K), need w^T
        # (K,N). mat2 must stay a column-major view (w.t(), NOT .contiguous()) or
        # cuBLASLt rejects it.
        wq_t = wq.t()
        sm = torch._scaled_mm if hasattr(torch, "_scaled_mm") else None
        if sm is None:
            print(f"M={M} N={N}: torch._scaled_mm NOT available")
            continue
        # out = a @ b
        out = sm(xq, wq_t, out_dtype=torch.bfloat16, scale_a=xs, scale_b=ws)
        t_fp8 = best(lambda: sm(xq, wq_t, out_dtype=torch.bfloat16, scale_a=xs, scale_b=ws))

        # reference: dequant fp8 weights and matmul in bf16
        dq = (wq.float() * ws).to(torch.bfloat16)
        ref = x @ dq.T
        rel = rel_err(out, ref)
        rel_bf16 = rel_err(out, x @ w.T)

        print(
            f"M={M} N={N} K={K}: bf16={t_bf16:.4f}ms fp8_scaleMM={t_fp8:.4f}ms "
            f"speedup={t_bf16 / t_fp8:.3f}x rel_vs_dq={rel:.4f} rel_vs_bf16={rel_bf16:.4f}"
        )


if __name__ == "__main__":
    main()
