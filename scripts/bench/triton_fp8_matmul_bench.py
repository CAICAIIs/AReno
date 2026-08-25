"""Tuned Triton bf16 vs FP8(W8A16) matmul bench on A100.

Applies SOTA Triton GEMM levers to both kernels so the delta isolates the
weight-precision (2 bytes vs 1 byte) memory effect:
  * num_stages  — pipelined prefetch of the next tile while computing (critical
                  to reach memory bandwidth),
  * larger BLOCK_N / BLOCK_K and num_warps,
  * `.cg` (cache-global) on the large streamed weight load and default on the
    activation, so the weight doesn't thrash L1 and the kernel is bandwidth-,
    not latency-, bound.
Runs the same config sweep for bf16 and fp8 and reports the best of each.
"""

from __future__ import annotations

import time

import torch
import triton
import triton.language as tl


@triton.jit
def _mm_bf16(
    a_ptr, b_ptr, y_ptr, M, N, K, sa, sb, sy, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IS_CG: tl.constexpr
):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        rk = k * BK + tl.arange(0, BK)
        am = (rm[:, None] < M) & (rk[None, :] < K)
        bm_ = (rn[:, None] < N) & (rk[None, :] < K)
        a = tl.load(a_ptr + rm[:, None] * sa + rk[None, :], mask=am, other=0.0)
        b = tl.load(b_ptr + rn[:, None] * sb + rk[None, :], mask=bm_, other=0.0, cache_modifier=".cg" if IS_CG else "")
        acc += tl.dot(a, tl.trans(b))
    tl.store(y_ptr + rm[:, None] * sy + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _mm_fp8(
    a_ptr,
    b_ptr,
    scale_ptr,
    y_ptr,
    M,
    N,
    K,
    sa,
    sb,
    sy,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    IS_CG: tl.constexpr,
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
        a = tl.load(a_ptr + rm[:, None] * sa + rk[None, :], mask=am, other=0.0)
        b = tl.load(b_ptr + rn[:, None] * sb + rk[None, :], mask=bm_, other=0.0, cache_modifier=".cg" if IS_CG else "")
        # Per-tensor scalar scale distributes out of the matmul, so dequantize
        # the tile to fp16 (Ampere tensor cores) and apply the scalar AFTER the
        # dot — no per-element multiply in the inner loop.
        acc += tl.dot(a.to(tl.float16), tl.trans(b.to(tl.float16)))
    acc = acc * scale
    tl.store(y_ptr + rm[:, None] * sy + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


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
    K, N, M = 4096, 12288, 32
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
    scale = (w.abs().amax() / 57344.0).to(torch.float32).reshape(())
    w_fp8 = (w.float() / scale).clamp(-57344.0, 57344.0).to(torch.float8_e5m2)
    y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)

    # correctness
    _mm_fp8[(triton.cdiv(M, 32), triton.cdiv(N, 128))](
        x, w_fp8, scale, y, M, N, K, x.stride(0), w_fp8.stride(0), y.stride(0), 32, 128, 64, True
    )
    torch.cuda.synchronize()
    ref = x.float() @ w.float().T
    rel = float((y.float() - ref).abs().max() / (ref.abs().max() + 1e-6))
    print(f"fp8 output rel err vs bf16 = {rel:.4f}")

    bms = [32, 64]
    bns = [64, 128, 256]
    bks = [64, 128]
    stages = [2, 3, 4]
    warps = [4, 8]

    def run(kernel, use_cg):
        is_fp8 = kernel is _mm_fp8
        bmat = w_fp8 if is_fp8 else w
        bstr = bmat.stride(0)
        best = None
        for BM in bms:
            for BN in bns:
                for BK in bks:
                    for ns in stages:
                        for nw in warps:
                            try:
                                grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
                                if is_fp8:
                                    args = (x, bmat, scale, y, M, N, K, x.stride(0), bstr, y.stride(0))
                                else:
                                    args = (x, bmat, y, M, N, K, x.stride(0), bstr, y.stride(0))
                                kernel[grid](*args, BM, BN, BK, use_cg, num_stages=ns, num_warps=nw)
                                torch.cuda.synchronize()

                                def fn(_k=kernel, _g=grid, _a=args, _BM=BM, _BN=BN, _BK=BK, _cg=use_cg, _ns=ns, _nw=nw):
                                    _k[_g](*_a, _BM, _BN, _BK, _cg, num_stages=_ns, num_warps=_nw)

                                t = _bench(fn)
                                if best is None or t < best[0]:
                                    best = (t, BM, BN, BK, ns, nw)
                            except Exception:
                                continue
        return best

    bf16_best, fp8_best = None, None
    for cg in (False, True):
        b = run(_mm_bf16, cg)
        if b and (bf16_best is None or b[0] < bf16_best[0]):
            bf16_best = b
        f = run(_mm_fp8, cg)
        if f and (fp8_best is None or f[0] < fp8_best[0]):
            fp8_best = f

    if bf16_best and fp8_best:
        bt, fp8t = bf16_best[0], fp8_best[0]
        print(
            f"bf16 best: {bt * 1e3:.3f} ms/step (BM={bf16_best[1]} BN={bf16_best[2]} BK={bf16_best[3]} ns={bf16_best[4]} w={bf16_best[5]})"
        )
        print(
            f"fp8  best: {fp8t * 1e3:.3f} ms/step (BM={fp8_best[1]} BN={fp8_best[2]} BK={fp8_best[3]} ns={fp8_best[4]} w={fp8_best[5]})"
        )
        print(f"speedup (bf16_time/fp8_time) = {bt / fp8t:.2f}x  (fp8 fraction of bf16 = {fp8t / bt:.2f})")
        print(f"weight bytes: bf16={w.numel() * 2}, fp8={w.numel() * 1}; theoretical ceil = 2.00x")
    else:
        print("no valid config for one of the kernels")


if __name__ == "__main__":
    main()
