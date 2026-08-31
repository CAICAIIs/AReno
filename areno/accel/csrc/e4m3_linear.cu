// E4M3 (FP8, 3 mantissa bits) fused dequant-linear for AReno on Ampere / A100.
//
// Computes:  y(M,N) = x(M,K) @ (dequant_e4m3(w(N,K)) * scale)^T
// where the weight is stored as uint8 E4M3 (1 byte/element) and the per-tensor
// scalar `scale` is applied after the dot. Reading the byte-packed weight halves
// the weight bytes pulled from HBM vs the bf16 `areno_linear` path, which is the
// memory-bandwidth benefit from RFC 0001 (theoretical ~2x, decode-bound).
//
// Forward-only: E4M3 has no backward, so this is a decode/inference operator.
// Two kernels serve different shapes:
//   * e4m3_gemv_kernel — the small-M (M<=4) memory-bound decode path: one warp
//     streams a weight row with 32-byte-coalesced reads, decoding each E4M3 byte
//     inline (branchless bit-trick) and reducing with warp shuffles. No tensor
//     cores; the weight (not the activation) is the streamed operand.
//   * e4m3_linear_kernel — the general-M fallback (M>4): a WMMA (fp16 16x16x16)
//     tensor-core GEMM that stages the decoded weight as fp16 in shared memory.
// Both multiply the fp32 accumulator by `scale` before writing bf16.
//
// Decode reference (validated against torch.float8_e4m3fn on CPU):
//   sign = (b>>7)&1; e = (b>>3)&0xF; m = b&0x7
//   e==0 (subnormal):  (-1)^s * m * 2^-9
//   1<=e<=15 (normal): (-1)^s * 2^(e-7) * (1 + m/8); max finite 448 (e=15,m=6)
//   e==15 && m==7 is NaN in torch; weights are clamped at 448 so it cannot
//   occur in a valid E4M3 payload, but we clamp it to 448 defensively.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <type_traits>

namespace areno_accel {

using namespace nvcuda;

// One WMMA 16x16x16 fp16 tensor-core tile.
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
// Block tile. 8 warps (256 threads) cover a 2x4 grid of 16x16 sub-tiles.
constexpr int kBM = 32;
constexpr int kBN = 64;
constexpr int kBK = 16;  // WMMA K-step; the GEMM variant is the M>4 fallback
constexpr int kThreads = 256;  // (kBM/16)*(kBN/16) = 2*4 = 8 warps

typedef wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major> WmmaA;
typedef wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, __half, wmma::row_major> WmmaB;
typedef wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> WmmaC;

__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
  // Branchless-ish fast decode: E4M3 -> fp32 by laying the bits out directly.
  // E4M3 value = (-1)^s * 2^(e-7) * (1 + m/8), bias 7. fp32 shares the exponent
  // bias 127 with bf16, so the normal case is just fp32_exp = e-7+127 = e+120 and
  // the 3 mantissa bits placed at the top of fp32's 23-bit mantissa (<<20).
  // Subnormal (e==0) and the NaN pattern (e==15,m==7) are handled with cheap
  // selects, avoiding exp2f in the hot decode loop.
  const uint32_t s = (b >> 7) & 1;
  const uint32_t e = (b >> 3) & 0xF;
  const uint32_t m = b & 0x7;
  const uint32_t fbits_normal = (s << 31) | ((e + 120) << 23) | (m << 20);
  const float v_normal = __uint_as_float(fbits_normal);  // sign already encoded
  const float v_sub = (s ? -1.0f : 1.0f) * (static_cast<float>(m) * 0.001953125f);  // m * 2^-9
  // Subnormal (e==0) takes the sign; NaN (e==15,m==7) clamps to max finite.
  return (e == 0) ? v_sub : ((e == 15 && m == 7) ? 448.0f : v_normal);
}

template <typename T>
__device__ __forceinline__ __half to_half(T v);

template <>
__device__ __forceinline__ __half to_half<c10::Half>(c10::Half v) {
  return __float2half(static_cast<float>(v));
}
template <>
__device__ __forceinline__ __half to_half<c10::BFloat16>(c10::BFloat16 v) {
  return __float2half(static_cast<float>(v));
}
template <>
__device__ __forceinline__ __half to_half<float>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ __half to_half<double>(double v) {
  return __float2half(static_cast<float>(v));
}

// WMMA GEMM fallback for M > 4 (correct but not tuned for tiny M). The decode
// fast path for M <= 4 is e4m3_gemv_kernel below.
template <typename T>
__global__ void e4m3_linear_kernel(
    const T* __restrict__ x,            // (M, K) row-major
    const uint8_t* __restrict__ w,      // (N, K) row-major, uint8 E4M3
    const float* __restrict__ scale,    // per-tensor scalar
    c10::BFloat16* __restrict__ y,      // (M, N) bf16 row-major
    int M,
    int N,
    int K,
    int ldx,
    int ldw,
    int ldy) {
  const int n0 = blockIdx.x * kBN;
  const int m0 = blockIdx.y * kBM;
  const int tid = threadIdx.x;
  const int warp = tid / 32;

  // Sub-tile owned by this warp (4x4 grid of 16x16 tiles per block).
  const int wm = warp / (kBN / kWmmaN);
  const int wn = warp % (kBN / kWmmaN);

  __shared__ __half A_smem[kBM][kBK];
  __shared__ __half B_smem[kBK][kBN];
  __shared__ float C_smem[kBM][kBN];

  WmmaA a_frag;
  WmmaB b_frag;
  WmmaC c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += kBK) {
    // Load A tile (kBM x kBK): x[m][k] -> fp16. Threads index the contiguous k
    // dim fastest, so a warp reads consecutive addresses (coalesced).
#pragma unroll
    for (int i = tid; i < kBM * kBK; i += kThreads) {
      const int r = i / kBK;
      const int kk = i % kBK;
      const int gm = m0 + r;
      const int gk = k0 + kk;
      if (gm < M && gk < K) {
        A_smem[r][kk] = to_half(x[gm * ldx + gk]);
      } else {
        A_smem[r][kk] = __float2half(0.0f);
      }
    }
    // Load B tile (kBN x kBK): decode uint8 E4M3 w[n][k] -> fp16, into B_smem[k][n].
    // Threads index the contiguous k dim fastest so the weight bytes stream in
    // coalesced (this is the memory-bound path; the earlier stride-N indexing
    // fetched one cache line per element).
#pragma unroll
    for (int i = tid; i < kBK * kBN; i += kThreads) {
      const int kk = i % kBK;  // k index (contiguous)
      const int c = i / kBK;   // n index
      const int gk = k0 + kk;
      const int gn = n0 + c;
      if (gn < N && gk < K) {
        B_smem[kk][c] = __float2half(e4m3_to_float(w[gn * ldw + gk]));
      } else {
        B_smem[kk][c] = __float2half(0.0f);
      }
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kBK; kk += kWmmaK) {
      wmma::load_matrix_sync(a_frag, &A_smem[wm * kWmmaM][kk], kBK);
      wmma::load_matrix_sync(b_frag, &B_smem[kk][wn * kWmmaN], kBN);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    __syncthreads();
  }

  // Apply the per-tensor scalar scale to the fp32 accumulator, then store.
  const float s = *scale;
#pragma unroll
  for (int i = 0; i < c_frag.num_elements; ++i) {
    c_frag.x[i] *= s;
  }
  wmma::store_matrix_sync(&C_smem[wm * kWmmaM][wn * kWmmaN], c_frag, kBN, wmma::mem_row_major);
  __syncthreads();

#pragma unroll
  for (int i = tid; i < kBM * kBN; i += kThreads) {
    const int r = i / kBN;
    const int c = i % kBN;
    const int gm = m0 + r;
    const int gn = n0 + c;
    if (gm < M && gn < N) {
      y[gm * ldy + gn] = static_cast<c10::BFloat16>(C_smem[r][c]);
    }
  }
}

// Small-M / memory-bound decode path. Each warp owns one output row `n` and
// streams w[n,:] with 32-byte-coalesced reads along the contiguous k dim, reusing
// the (kM x K) activation staged in shared, then warp-reduces with shuffles. This
// avoids the tensor-core GEMM's M-waste and register pressure on tiny M. kM is
// the batch of rows (== M).
constexpr int kWarps = 8;         // warps per block -> 256 threads
constexpr int kGemvThreads = kWarps * 32;
constexpr int kGemvMaxM = 4;      // small-M decode path; larger M uses the WMMA GEMM fallback


template <typename T, int kM, bool kFloatXs>
__global__ void e4m3_gemv_kernel(
    const T* __restrict__ x,        // (kM, K) row-major, kM == M
    const uint8_t* __restrict__ w,  // (N, K) uint8, contiguous k
    const float* __restrict__ scale,
    c10::BFloat16* __restrict__ y,  // (kM, N)
    int N,
    int K,
    int ldw,
    int ldy) {
  extern __shared__ char smem_raw[];
  // Small M (decode): stage the activation as float so the inner loop has no
  // bf16->float cast (the biggest compute cost at M=1). Larger M keeps bf16 x so
  // shared is small and occupancy stays high (float x at M>=3 needs >48KB and
  // drops occupancy, which regresses M=4).
  using XS_t = typename std::conditional<kFloatXs, float, T>::type;
  XS_t* const xs = reinterpret_cast<XS_t*>(smem_raw);  // [kM * K]
  for (int i = threadIdx.x; i < kM * K; i += kGemvThreads) {
    xs[i] = static_cast<XS_t>(x[i]);
  }
  __syncthreads();

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int n = blockIdx.x * kWarps + warp;
  if (n >= N) return;

  const uint8_t* __restrict__ wrow = w + static_cast<int64_t>(n) * ldw;
  float acc[kM];
#pragma unroll
  for (int m = 0; m < kM; ++m) acc[m] = 0.0f;

  // Vectorized phase: each lane handles 8 consecutive k per step (a uint64 load
  // of the uint8 weight), giving a full 256-byte coalesced access per warp (2
  // cache lines) and 8x the memory-level parallelism of the scalar loop. Only
  // safe when the row base is 8-byte aligned (K % 8 == 0). Scalar fallback all K.
  if ((K & 7) == 0) {
    const int k8 = K >> 3;
#pragma unroll 4
    for (int g = lane; g < k8; g += 32) {
      const int k = g << 3;
      const uint64_t wq = *reinterpret_cast<const uint64_t*>(wrow + k);
      const float wv0 = e4m3_to_float(static_cast<uint8_t>(wq & 0xFF));
      const float wv1 = e4m3_to_float(static_cast<uint8_t>((wq >> 8) & 0xFF));
      const float wv2 = e4m3_to_float(static_cast<uint8_t>((wq >> 16) & 0xFF));
      const float wv3 = e4m3_to_float(static_cast<uint8_t>((wq >> 24) & 0xFF));
      const float wv4 = e4m3_to_float(static_cast<uint8_t>((wq >> 32) & 0xFF));
      const float wv5 = e4m3_to_float(static_cast<uint8_t>((wq >> 40) & 0xFF));
      const float wv6 = e4m3_to_float(static_cast<uint8_t>((wq >> 48) & 0xFF));
      const float wv7 = e4m3_to_float(static_cast<uint8_t>((wq >> 56) & 0xFF));
#pragma unroll
      for (int m = 0; m < kM; ++m) {
        const XS_t* const xm = xs + static_cast<int64_t>(m) * K + k;
        acc[m] += static_cast<float>(xm[0]) * wv0 + static_cast<float>(xm[1]) * wv1 +
                  static_cast<float>(xm[2]) * wv2 + static_cast<float>(xm[3]) * wv3 +
                  static_cast<float>(xm[4]) * wv4 + static_cast<float>(xm[5]) * wv5 +
                  static_cast<float>(xm[6]) * wv6 + static_cast<float>(xm[7]) * wv7;
      }
    }
  } else {
    for (int k = lane; k < K; k += 32) {
      const float wv = e4m3_to_float(wrow[k]);
#pragma unroll
      for (int m = 0; m < kM; ++m) {
        acc[m] += xs[static_cast<int64_t>(m) * K + k] * wv;
      }
    }
  }

  const float s = *scale;
#pragma unroll
  for (int m = 0; m < kM; ++m) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc[m] += __shfl_down_sync(0xffffffffu, acc[m], off);
    }
    if (lane == 0) {
      y[static_cast<int64_t>(m) * ldy + n] = static_cast<c10::BFloat16>(acc[m] * s);
    }
  }
}

}  // namespace areno_accel

// Launch the small-M GEMV (direct-load kernel). No cudaFuncSetAttribute: the
// float-xs (M<=2) and bf16-xs (M>=3) shared sizes are all < 48 KB, so the
// default dynamic-shared limit suffices.
template <typename T, int kM, bool kFloatXs>
void areno_launch_e4m3_gemv(
    const T* x,
    const uint8_t* w,
    const float* scale,
    c10::BFloat16* y,
    int N,
    int K,
    int ldw,
    int ldy,
    cudaStream_t stream,
    size_t shmem_direct,
    dim3 grid,
    dim3 block) {
  areno_accel::e4m3_gemv_kernel<T, kM, kFloatXs><<<grid, block, shmem_direct, stream>>>(
      x, w, scale, y, N, K, ldw, ldy);
}

torch::Tensor areno_e4m3_linear_forward_cuda(
    torch::Tensor input,
    torch::Tensor w_u8,
    torch::Tensor scale,
    c10::optional<torch::Tensor> out) {
  TORCH_CHECK(input.is_cuda(), "areno_e4m3_linear input must be CUDA");
  TORCH_CHECK(w_u8.is_cuda(), "areno_e4m3_linear weight must be CUDA");
  TORCH_CHECK(scale.is_cuda(), "areno_e4m3_linear scale must be CUDA");
  TORCH_CHECK(w_u8.scalar_type() == at::kByte, "areno_e4m3_linear weight must be uint8 (E4M3 bytes)");
  TORCH_CHECK(scale.scalar_type() == at::kFloat, "areno_e4m3_linear scale must be float32");
  TORCH_CHECK(input.dim() >= 2, "areno_e4m3_linear input must have at least 2 dims");
  TORCH_CHECK(w_u8.dim() == 2, "areno_e4m3_linear weight must be 2D");
  TORCH_CHECK(input.size(-1) == w_u8.size(1), "areno_e4m3_linear input/weight K mismatch");
  TORCH_CHECK(w_u8.is_contiguous(), "areno_e4m3_linear weight must be contiguous");
  TORCH_CHECK(input.is_contiguous(), "areno_e4m3_linear input must be contiguous");
  TORCH_CHECK(scale.numel() == 1, "areno_e4m3_linear scale must be scalar");

  auto out_shape = input.sizes().vec();
  out_shape.back() = w_u8.size(0);
  // Reuse a caller-provided bf16 CUDA output buffer when it already matches, so
  // the hot decode path does not pay a per-token torch::empty.
  torch::Tensor output;
  if (out.has_value() && out->is_cuda() && out->is_contiguous() &&
      out->scalar_type() == at::kBFloat16 && out->sizes() == out_shape) {
    output = out.value();
  } else {
    output = torch::empty(out_shape, input.options().dtype(at::kBFloat16));
  }

  int64_t K = input.size(-1);
  int64_t M = input.numel() / K;
  int64_t N = w_u8.size(0);
  int ldx = static_cast<int>(input.stride(-2));
  int ldw = static_cast<int>(w_u8.stride(0));
  int ldy = static_cast<int>(output.stride(-2));

  const at::cuda::OptionalCUDAGuard guard(device_of(input));
  auto stream = at::cuda::getCurrentCUDAStream();

  // Small-M decode path uses the memory-streaming GEMV kernel (coalesced weight
  // reads, no M-waste); larger M falls back to the WMMA GEMM kernel (correct but
  // not tuned for tiny M). Both are forward-only (E4M3).
  const bool use_gemv = (M >= 1 && M <= areno_accel::kGemvMaxM);
  const uint8_t* wptr = static_cast<const uint8_t*>(w_u8.data_ptr());
  c10::BFloat16* yptr = output.data_ptr<c10::BFloat16>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, input.scalar_type(), "areno_e4m3_linear", [&] {
        const scalar_t* xptr = input.data_ptr<scalar_t>();
        if (use_gemv) {
          // Small M (decode) stages x as float (no bf16->float cast in the inner
          // loop, the biggest compute cost); M>=3 keeps bf16 x so shared stays
          // small and M=4 occupancy is not lost.
          dim3 grid(static_cast<unsigned>((N + areno_accel::kWarps - 1) / areno_accel::kWarps), 1);
          dim3 block(areno_accel::kGemvThreads);
          const int Ni = static_cast<int>(N);
          const int Ki = static_cast<int>(K);
          const size_t shmem1 = static_cast<size_t>(K) * sizeof(float);          // M=1 float
          const size_t shmem2 = 2 * static_cast<size_t>(K) * sizeof(float);      // M=2 float
          const size_t shmem3 = 3 * static_cast<size_t>(K) * sizeof(scalar_t);   // M=3 bf16
          const size_t shmem4 = 4 * static_cast<size_t>(K) * sizeof(scalar_t);   // M=4 bf16
          switch (M) {
            case 1:
              areno_launch_e4m3_gemv<scalar_t, 1, true>(xptr, wptr, scale.data_ptr<float>(), yptr, Ni, Ki, ldw, ldy,
                                                        stream, shmem1, grid, block);
              break;
            case 2:
              areno_launch_e4m3_gemv<scalar_t, 2, true>(xptr, wptr, scale.data_ptr<float>(), yptr, Ni, Ki, ldw, ldy,
                                                        stream, shmem2, grid, block);
              break;
            case 3:
              areno_launch_e4m3_gemv<scalar_t, 3, false>(xptr, wptr, scale.data_ptr<float>(), yptr, Ni, Ki, ldw, ldy,
                                                         stream, shmem3, grid, block);
              break;
            case 4:
              areno_launch_e4m3_gemv<scalar_t, 4, false>(xptr, wptr, scale.data_ptr<float>(), yptr, Ni, Ki, ldw, ldy,
                                                         stream, shmem4, grid, block);
              break;
          }
        } else {
          dim3 grid(static_cast<unsigned>((N + areno_accel::kBN - 1) / areno_accel::kBN),
                    static_cast<unsigned>((M + areno_accel::kBM - 1) / areno_accel::kBM));
          dim3 block(areno_accel::kThreads);
          areno_accel::e4m3_linear_kernel<scalar_t><<<grid, block, 0, stream>>>(
              xptr, wptr, scale.data_ptr<float>(), yptr, static_cast<int>(M), static_cast<int>(N),
              static_cast<int>(K), ldx, ldw, ldy);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
