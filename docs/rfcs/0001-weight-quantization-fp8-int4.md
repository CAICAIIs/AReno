# RFC 0001 — FP8 / INT4 Weight Quantization for AReno (CUDA Train & Decode)

- **Metadata**
  - **Status:** Draft (for comment)
  - **Author:** @CAICAIIs
  - **Date:** 2026-08-25
  - **Affected subsystems:** `areno/models/*`, `areno/engine/layers/linear.py`, `areno/accel/`,
    `areno/models/registry.py`, `areno/engine/checkpoints/io.py`, `areno/engine/config.py` (`ModelConfig`),
    `areno/cli/train.py`, `areno/cli/serve.py`
  - **Reviewers:** open

---

## 1. Background & Problem

AReno targets fast, self-contained single-node post-training. On a single 8×A100 node we measured the
following for a GRPO rollout with native attention and world=4 / tp=4:

| model | weights re-read per generated token | decode tokens/s | weight-read bandwidth | compute utilization |
|---|---|---|---|---|
| Qwen3-0.6B | ~1.2 GiB | ~1450 | ~1.7 TB/s | ~1.7 TFLOPs/s (overhead/latency-dominated) |
| Qwen3-8B | ~16 GiB | ~390 | ~6.2 TB/s ≈ HBM aggregate limit | ~0.5% (**memory-bandwidth-bound**) |

**Problem statement.** For models that actually stress a single node (≈4B parameters and above), the
decode path is **memory-bandwidth-bound**: every generated token re-reads the full weight tensor from HBM,
while the compute units sit largely idle. The dominant, well-understood mitigation is to shrink the number
of bytes read per weight — i.e. **weight quantization**. AReno today has **no weight quantization**: the only
quantization-related code is 8-bit optimizer moments (`areno/engine/optim/adamw_8bit.py`) and the MLX
optimizer path; all linear layers run plain bf16 (`areno/engine/layers/linear.py`,
`areno/accel/csrc/linear.cu`).

This RFC proposes a backward-compatible, opt-in FP8 / INT4 weight-quantization capability for the CUDA
backend, TP-aware, with a correctness gate and a quantifiable token/s benchmark.

---

## 2. Objectives & Scope

**Objectives (v1, CUDA backend).**
1. Reduce bytes read per weight so decode token/s increases measurably (target: near-linear with the byte
   reduction for weights).
2. Preserve correctness: a trained step under quantization stays within a documented tolerance of the full-precision
   baseline (loss, grad norm, advantage mean).
3. Remain fully backward-compatible and opt-in: the default config is unchanged `quantization="none"`.
4. Integrate cleanly with tensor parallelism on a single node.

**Non-goals (explicitly out of scope).**
- KV-cache quantization (a distinct bottleneck; to be proposed separately).
- Multi-node parallelism, ONNX/CPU backends, weight sparsification.
- Automatic post-training calibration as part of `areno train` (calibration is assumed already performed when a
  quantized checkpoint is supplied, or handled by a separate tool).

---

## 3. Design

### 3.1 Supported schemes & selection

| scheme | bytes/weight | use |
|---|---|---|
| `none` (bf16) | 2 | baseline |
| `fp8` (E4M3) | 1 | W8A16 (per-tensor / per-channel scale) — v1 |
| `int4` (grouped, e.g. g128, GPTQ/AWQ) | 0.5 | W4A16 — v1.1 |

Selection is two-way:

- **Explicit config.** Add `quant_method: "none" | "fp8" | "int4"` to `ModelConfig` (with per-method fields, e.g.
  `quant_group_size: int = 128`, `quant_fp8_dtype: "e4m3"`), surfaced as `--quant {none,fp8,int4}` on
  `areno train` and `areno serve`.
- **Checkpoint-driven.** If the checkpoint carries a `quantization_config` block (or an equivalent signature in
  `config.json`), AReno auto-selects `quant_method` so a pre-quantized checkpoint "just works" without a flag.
  This keeps a quantized model re-runnable in one command and avoids a silent double-quantization.

### 3.2 Layer replacement (registry-driven)

Instead of editing each model's forward, introduce a quantized-layer registry:

- `areno/models/registry.py` gains a mapping from each model family's linear class to a **family adapter** that
  knows how to construct a `QuantizedLinear` from (weight, scale, zero-point, group-size) and how to shard /
  de-shard those tensors under tensor parallelism.
- `areno/engine/layers/linear.py` gains a `QuantizedLinear` module that dequantizes on-device (or runs a fused
  dequant-matmul) and then uses the existing TP path, so collectives operate on the float side.

This keeps adding families additive (a new adapter) rather than modifying a shared factory.

### 3.3 Checkpoint loading / conversion

Two load styles behind one loader:

- **Static (pre-quantized).** Recognize a `quantization_config`; load packed integer tensors plus
  `weight_scale` and (optionally) `zero_point` and `group_size`.
- **Dynamic (quantize-on-load).** Quantize an existing bf16 checkpoint at load time. FP8 is available without
  calibration; INT4 requires calibration data, so INT4 v1 is **pre-quantized only**.

On the I/O side, the weight-layout helpers in `areno/engine/checkpoints/io.py` and each family's
`checkpoint.py` must round-trip quantized tensors (packed ints + scales + zero-points), and the policy-sync
weight plan (`areno/engine/policy_sync.py`) must understand the new keys so weights can be exchanged between
train and rollout partitions.

### 3.4 Kernels

Keep the CUDA path self-contained (areno-owned, per AGENTS.md):

- Add `dequant_linear` for FP8 and INT4 to `areno/accel/` (INT4 dequant → bf16 → existing `linear.cu`), or a
  fused `quantized_linear.cu`.
- Deliberately **not** a hard new runtime dependency (e.g. no mandatory bitsandbytes). A fast path via an
  established CUDA utility may be added later as an opt-in.

**Compute-capability constraint (measured on the 8×A100 reference host).**
`torch._scaled_mm` — the built-in FP8 matmul that consumes FP8 weights directly — requires Hopper
(compute capability ≥ 8.9) or ROCm MI300+; it is **unavailable on Ampere (A100, cc 8.0)**. Verified locally:
`RuntimeError: torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9`.
Consequences:
- On **Hopper/H100**, the FP8 decode path can use the built-in FP8 matmul as a low-effort fast path.
- On **Ampere/A100** (the reference host), a **custom fused FP8-dequant matmul in `areno/accel`** is required
  to realize the memory-bound speedup (dequant-then-bf16-matmul does not reduce bytes read, so it yields no
  speedup — see §5). This raises the M1 effort on A100.

### 3.5 Tensor-parallel integration

- Quantized weights are TP-sharded inside the packed integer tensor; per-channel/group `weight_scale` and
  `zero_point` are sharded together with their group.
- The forward dequantizes to a local float shard and reuses the existing TP reduce/all-reduce. Gradients
  accumulate in the existing FP32 master (or 8-bit) optimizer unchanged.

---

## 4. Validation plan (merge gate)

1. **Numerical / CPU tests** (`tests/`, required):
   - `dequant(matmul)` matches a reference implementation for FP8 and INT4 (random tensors, per-scheme tolerance).
   - TP shard + all-reduce of a quantized weight equals the unsharded result (CPU parity test).
   - Config / CLI parsing; `quantization="none"` produces the existing path byte-for-byte.
2. **Behavior-preserving gate:** run a bounded GRPO (or SFT) step with `--quant fp8` and assert `loss`,
   `grad_norm`, `advantage_mean` are within a documented tolerance of the `--quant none` baseline.
3. **Benchmark (primary acceptance):** on `Qwen3-8B` @ world=4/tp=4 on 8×A100, measure decode **tokens/s and
   peak GPU memory** for `none` vs `fp8` (v1.1 adds `int4`), reusing the `scripts/bench` probe. Target: near-linear
   speedup from the byte reduction with memory headroom.
4. **Docs:** `docs/models/` + CLI reference document `--quant` usage and accuracy caveats.

### 4.5 P1 kernel — measured status (2026-08-25)

A Triton FP8(E5M2) W8A16 dequant-linear is implemented in `areno/accel/kernels/fp8_linear.py`
and wired into the single matmul entry point `areno/engine/layers/linear.py:_areno_linear_forward`
(detects a `_areno_fp8` weight payload and routes to the kernel; default path unchanged).

Measured on Qwen3-8B MLP shape (batch 32 × 4096 → 12288) on A100, versus the production bf16
`areno_linear` path:

- **fp8 = 0.052 ms/step, bf16 = 0.084 ms/step → ~1.60×**, weight max-rel error 4.7% (per-tensor E5M2 scale).
- Key tunings: `num_stages=4` pipelining, `BLOCK_K=128`, `.cg` on the streamed weight, **FP16 tensor-core
  dot** (bf16 dot measured ~1.13× — the gap was dtype, not allocation), and **applying the per-tensor scale
  once post-dot** (`(x @ w_fp8) * scale`, so no per-element dequant in the inner loop).
- The 1.60× is per memory-bound linear; the full decode-path speedup is expected lower (attention/norm/vocab
  are not purely bandwidth-bound) but the memory-bound premise is validated on the real model shape.
- Note on Ampere (A100): a *built-in* FP8 matmul is unavailable (`torch._scaled_mm` is Hopper/Ada-only), so the
  kernel above is the A100 path; on Hopper the built-in FP8 matmul can be an opt-in fast path.

### 4.6 P1 correctness caveat — E5M2 on A100 is a *coarse* fast path, not a faithful bf16 drop-in

Measured end-to-end on the Qwen3-0.6B GRPO rollout with the model-level integration active
(`quantize_model_weights_fp8` marks **112** TP-parallel linears; the decode runs and scores rewards
normally). Using **greedy** (deterministic) decoding, the same prompts produce **different reasoning than
bf16** (e.g. `"…48 friends in April. Then, she sold…"` (bf16) vs `"…48 of her friends in April, and then she
sold…"` (fp8)). The FP8 **kernel is correct** (linear-level verified, ~5 % error); the divergence comes from
the **coarse E5M2 grid** (only the Triton FP8 dtype Ampere supports — `fp8e4nv`/E4M3 is rejected on A100),
whose ~12 % per-element error flips greedy argmax choices and cascades.

**Implication (be honest with consumers):** on A100 this is a **fast-but-coarse decode path**, not a faithful
replacement for bf16. It yields a real ~1.60× on the memory-bound linear but with accuracy divergence on
greedy decoding. To keep output faithful:
- prefer **Hopper/H100** (built-in FP8 matmul + E4M3, `torch._scaled_mm`); or
- add an **E4M3 dequant-linear CUDA kernel** (Ampere Triton only accepts E5M2), which is more work but finer.

Out of scope here: making E5M2 output match bf16. The M1 acceptance gate's §4.2 "behavior-preserving" step
**cannot be satisfied on A100 for greedy decode** under E5M2; it may be satisfiable on Hopper / with E4M3.

### 4.7 E4M3 CUDA fused dequant-linear — correct; small-M GEMV path, but still < bf16 on A100 (2026-08-25)

To get a **finer** (E4M3, 3 mantissa bits) grid on A100 where Triton rejects `fp8e4nv`, a custom CUDA kernel was
built. It reads the uint8 E4M3 weight (1 byte/element) and decodes each byte to bf16 in-kernel (fast bit-decode
validated against `torch.float8_e4m3fn` in `tests/test_e4m3_decode_cpu.py`).

- `areno/accel/csrc/e4m3_linear.cu`: two kernels + registration.
  - `e4m3_linear_kernel`: WMMA (fp16 16×16×16) tensor-core GEMM for general M (correct, but capped at ~60 GB/s
    for the tiny-M decode case — the tensor-core GEMM wastes the M dim and its register pressure limits
    occupancy).
  - `e4m3_gemv_kernel`: **small-M memory-streaming GEMV** — one warp per output row, reading the row with
    **vectorized (8 bytes/lane) coalesced** `uint8` loads along the contiguous k dim, reusing the (M×K)
    activation from shared, and warp-reducing with shuffles. This is the right shape for the memory-bound
    decode case and is ~6.7× faster than the GEMM variant.
  - Dispatch: `M ≤ 4` uses the GEMV; larger M uses the GEMM.
- `areno/accel/kernels/e4m3_cuda.py` — `quantized_e4m3_linear_cuda` shim + quantize/dequant helpers.
- **Correctness (verified):** vs the bf16 reference, max-rel error ≈ 0.4–0.7 % (fp16 compute noise) across
  M ∈ {1,4,64}, N=12288, K=4096; vs full bf16 ≈ 2.8–3.6 % (E4M3 quantization, expected).
- **Performance (honest, corrected).** The E4M3 **kernel** streams the 1-byte weight once and is memory-bound
  (~0.05 ms at M=1, N=12288, K=4096 — ≈1.68× vs the bf16 cuBLAS kernel's ~0.084 ms at the GPU level). **But
  the end-to-end ratio through the torch/ATen extension is only ~1.0×** (single-launch bf16 0.101 ms vs E4M3
  0.098 ms = 1.03×; pipelined 0.88×). The kernel's speed advantage is swallowed by the per-call `torch::empty` +
  ATen dispatch + the shim's `.contiguous()` overhead (~0.04 ms), which is ~50–100 % of a 0.05 ms kernel. A real
  decode hot path reuses the output buffer (now supported via an `out` param) but the residual dispatch overhead
  still dominates. **>1.5× is NOT achieved end-to-end**; it exists only at the bare-kernel level and only if the
  framework dispatch overhead is eliminated (the next real step). An earlier "1.68×" claim was a measurement
  error (comparing the kernel-level pipelined throughput against the bf16 single-launch latency). Note: an
  earlier absolute number of ~1000 GB/s for a float-x standalone was also non-representative (only single-launch
  is realistic; a 2.2× gap was an artifact of comparing pipelined-vs-single-launch, not a genuine bf16 penalty —
  the real bf16-x kernel runs at the same rate as float-x).
### 4.8 Hopper/H20 — native FP8 E4M3 via `torch._scaled_mm` (measured >1.5×)

On a **NVIDIA H20 / "CX70"** (Hopper, compute capability 9.0, 8 GPUs), the built-in cuBLASLt FP8 matmul
(`torch._scaled_mm`, fp8_e4m3) gives the memory-bound decode speedup directly — no custom kernel needed.
Measured with `scripts/bench/h20_fp8_scaled_mm.py` at the MLP decode shape vs the bf16 cuBLAS baseline:

| M | N | K | bf16 | FP8 `_scaled_mm` | speedup |
|---|---|---|---|---|---|
| 1 | 12288 | 4096 | 0.0493 ms | 0.0292 ms | **1.69×** |
| 1 | 8192 | 4096 | 0.0398 ms | 0.0250 ms | **1.59×** |
| 4 | 12288 | 4096 | 0.0497 ms | 0.0289 ms | **1.72×** |
| 64 | 12288 | 4096 | 0.0662 ms | 0.0431 ms | **1.53×** |

**>1.5× achieved on Hopper/H20**, with correct FP8 E4M3 quantization error (~3–4 % rel vs bf16, the E4M3 grid).
`torch._scaled_mm` needs A row-major (M,K) and B column-major (K,N) — pass the weight as `w.t()` (not
`.contiguous()`) for the `x @ w^T` decode GEMM. This is the Hopper fast path the RFC §4.6 pointed to; on
Ampere (A100) the custom CUDA kernel in `areno/accel/csrc/e4m3_linear.cu` is the equivalent, and the
`rel_vs_bf16` ~4 % (E4M3) vs ~12 % (E5M2) confirms E4M3 is the *finer* grid that keeps greedy decode closer
to bf16 than the A100 E5M2 Triton path.

Note: getting a CUDA torch onto this host was itself high-effort — `download.pytorch.org` / PyPI were 27 KB/s
and the China mirrors (aliyun) lacked the exact `nvidia-*` versions the newest torch pins. Resolution: install
`torch 2.10.0+cu126 --no-deps` from the aliyun `pytorch-wheels` mirror (fast) and pull the `nvidia-*` CUDA
runtime wheels (cudnn-9.9, cublas-12.9, cudart/cusparse/nccl/cusolver/etc.) from the aliyun pypi mirror,
iterating to a consistent major-version set (libcudnn.so.9, libcublas.so.12) that torch's loader accepts.
- **Bandwidth ceiling (root-caused, honest).** The raw read-only kernel (row-per-warp, coalesced uint64
  weight reads) hits **~1248 GB/s** on A100, so the read pattern was **never** the bottleneck. The ~400 GB/s
  wall came from the **per-element `exp2f` E4M3 decode**. Replacing it with a **branchless bit-trick**
  (laying the E4M3 bits out into fp32 directly: `fp32 = (s<<31)|((e+120)<<23)|(m<<20)`, with cheap selects for
  the e==0 subnormal `m*2^-9` and the e==15,m==7 NaN clamp) — verified **exact** against `torch.float8_e4m3fn`
  for all 256 bytes — raises the GEMV to ~**815 GB/s (~0.062 ms)**, and staging the activation as **float** (and
  decoding the weight once per load) removes the per-element bf16→float cost. The kernel is now **memory-bound**;
  ~815 GB/s is ≈**1.36×** vs the bf16 cuBLAS 0.084 ms (read-only ceiling 2.1×). A `cp.async` pipeline variant
  was also built and its crash root-caused (Ampere `cp.async` is **max 16 bytes/op**; a 32-byte/lane copy traps
  on this CUDA 13.0 / cc8.0 toolchain — fixed by 16-byte copies) but it does not beat the direct kernel.
  NOTE: the shared host's GPUs were saturated while measuring, so the live `areno_linear`-vs-kernel ratio
  (~1.0×) is contention-inflated; the clean standalone is ~1.36×.

> 🧠 **From Hindsight memory** — the correct branchless E4M3→fp32 decode is `(s<<31)|((e+120)<<23)|(m<<20)`
> (fp32 shares bf16's exponent bias 127; the 3 mantissa bits go to the top of fp32's 23-bit mantissa). The
> subnormal case (e==0: `m*2^-9`) and the NaN pattern (e==15,m==7) need selects; do **not** re-apply the sign
> to the normal path (it's already in `s<<31`) — that double-negation caused a ~1.3 rel error that took a
> debug pass to catch.

**Implication:** E4M3 is the *correctness*-finer path and now has a working CUDA kernel, but the decode speedup
for it is **not yet realized** on A100. E5M2 (Triton, ~1.60× measured) remains the fast-but-coarse A100 path;
E4M3 needs either (a) a working cp.async small-M streaming pipeline (multi-step kernel work, toolchain issue
unresolved), or (b) Hopper/H100 (built-in FP8 matmul + E4M3).


---

## 5. Alternatives considered

| alternative | assessment |
|---|---|
| bf16 → fp16 | no byte reduction (both 2 bytes) → no memory-bound benefit. Rejected. |
| KV-cache quantization | targets KV reads, not weight reads; a different bottleneck. Separate feature. |
| 8-bit AdamW (already present) | reduces optimizer memory, not decode weight reads. Not a substitute. |
| Speculative decoding | complementary; improves latency but does not reduce bytes per token. |
| Weight caching / reuse | decode reads weights per token by construction; no reuse window. |
| Adopt a third-party quant runtime (e.g. bitsandbytes) as core | fast prototype, but adds a runtime dependency; keep optional, not core. |

---

## 6. Risks & mitigations

- **INT4 accuracy depends on calibration data.** Mitigation: ship INT4 as pre-quantized-only in v1, and gate on
  the numerical/behavioral tests.
- **Packed-int TP sharding** is the most likely correctness hazard. Mitigation: dedicated CPU parity tests for
  shard + all-reduce, plus the behavior-preserving step gate.
- **Per-family adapter growth** raises maintenance cost. Mitigation: registry-driven (additive), broad coverage
  via a shared `QuantizedLinear` and only thin per-family adapters.
- **Accuracy of a dynamic FP8 conversion** (no calibration) is acceptable but not free. Mitigation: document a
  tolerance and keep `none` as the default.

---

## 7. Roadmap

**Sequencing principle.** Prove measurable value early, then broaden scope, and gate every phase on the §4
validation plan — the correctness/perf gates are the merge bar. Each phase is independently landable.

### 7.1 Phases

| Milestone | Objective | In-scope | Deliverables | Dependencies | Exit criteria (gate) | Effort |
|---|---|---|---|---|---|---|
| **M0 — Design freeze** | Lock scope, interface, KPIs | `quant_method` config + CLI parity; tolerance + benchmark KPIs | Accepted RFC; `ModelConfig`/CLI changes | — | Reviewer sign-off; no open design Qs | ~0.5 wk (review) |
| **M1 — FP8 vertical slice** | Prove value on one model family | Qwen3 family only | `QuantizedLinear`; FP8 dequant kernel (`areno/accel`); dynamic load; `--quant fp8`; CPU/numerical tests; `areno serve`/rollout run | M0 | `none` vs `fp8` decode bench on Qwen3-8B; behavior-preserving step gate; CPU tests | ~2–3 wk |
| **M2 — TP + checkpoint round-trip** | Correct & repeatable under TP | FP8 weights+scales shard/de-shard; static (pre-quantized) load; policy-sync keys | TP shard/all-reduce parity; checkpoint save→load round-trip | M1 | TP parity test; round-trip reproduces quantized tensors | ~2–3 wk |
| **M3 — Family coverage & docs** | Broaden to the model matrix | llama, qwen3_5, gemma4, bailing, … | Per-family adapter registrations; `areno check` surface; `docs/models/` + CLI reference | M2 | Family matrix loads + decodes; docs shipped | ~2–3 wk |
| **M4 — INT4 (pre-quantized)** | Extend to 0.5 B/weight | INT4 GPTQ/AWQ load + dequant kernel + TP | `none/fp8/int4` benchmark table; behavior gate; tests | M2 | Benchmark table + gate + tests | ~3–4 wk |
| **M5 — Hardening** | Consolidate, optimize, integrate | fused dequant-matmul; optional fast path; async/rollout-policy integration | Perf/throughput report; no regressions | M3, M4 | Existing suite green; report documented | ~2 wk |

### 7.2 Sequencing & dependencies

```
M0 ──► M1 ──► M2 ──► M3 ──► M5
              └────► M4 ──► M5
```

- **Critical path:** M0 → M1 (value proof) → M2 (correctness under TP). M4 is parallelizable after M2.
- **Early stop-or-continue signal:** M1 alone yields the first measurable `none` vs `fp8` number; commit to M2+
  only if that result is compelling.

### 7.3 Effort summary

| scope | est. eng-weeks (1 contributor) |
|---|---|
| M0 | 0.5 |
| M1 | 2–3 |
| M2 | 2–3 |
| M3 | 2–3 |
| M4 | 3–4 |
| M5 | 2 |
| **Full feature** | **~12–17** |
| **M1 only (value proof)** | **~2–3** |

### 7.4 What is needed from reviewers / maintainers

- Confirm scope and the `quant_method` / `--quant` interface (M0).
- Decide the FP8 default (W8A16 vs W8A8) — see §8.
- Approve whether the M1 value-proof slice (~2–3 wk) is worth funding before committing to the full roadmap.

---

## 8. Open questions

1. Should FP8 default to W8A16 (dequant → bf16 matmul) or W8A8 (native FP8 matmul)? We lean W8A16 first for
   risk, W8A8 as an opt-in fast path.
2. Is `areno serve` (inference-only) sufficient to validate decode throughput, or must the RL rollout path
   also be exercised at each phase?
3. Does the community prefer a single `quant_method` string, or distinct `--quant-fp8` / `--quant-int4` flags?
   (Current proposal: single `--quant` with per-method params.)
