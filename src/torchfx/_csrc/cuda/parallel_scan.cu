#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "torchfx/parallel_scan.h"

// Parallel prefix scan for biquad IIR filtering.
//
// The biquad recurrence y[n] = f[n] - a1*y[n-1] - a2*y[n-2] is reformulated
// as a 3x3 matrix recurrence:
//
//   s[n] = M[n] * s[n-1]
//   where s = [y[n], y[n-1], 1]^T
//   and   M = [-a1, -a2, f[n]; 1, 0, 0; 0, 0, 1]
//
// The scan operator is 3x3 matrix multiplication (associative).
// We use a Blelloch-style up-sweep/down-sweep for work efficiency.
//
// Optimization: Since the bottom row is always [0, 0, 1], we only store
// and multiply the 2x3 top portion (6 elements per matrix).

namespace torchfx {

// A "reduced" 3x3 matrix where row 2 = [0, 0, 1].
// We store rows 0 and 1 as 6 doubles.
struct Mat3x3 {
  double m[6];
  // Layout: m[0]=r0c0, m[1]=r0c1, m[2]=r0c2,
  //         m[3]=r1c0, m[4]=r1c1, m[5]=r1c2
};

__device__ __forceinline__ Mat3x3 mat_identity() {
  Mat3x3 I;
  I.m[0] = 1.0; I.m[1] = 0.0; I.m[2] = 0.0;
  I.m[3] = 0.0; I.m[4] = 1.0; I.m[5] = 0.0;
  return I;
}

// Multiply A * B where both have implicit row 2 = [0, 0, 1].
// Result also has implicit row 2 = [0, 0, 1].
__device__ __forceinline__ Mat3x3 mat_mul(const Mat3x3& A, const Mat3x3& B) {
  // Full 3x3 with implicit third row [0,0,1]:
  // R[0][j] = A[0][0]*B[0][j] + A[0][1]*B[1][j] + A[0][2]*B[2][j]
  // R[1][j] = A[1][0]*B[0][j] + A[1][1]*B[1][j] + A[1][2]*B[2][j]
  // where B[2][j] = {0, 0, 1}
  Mat3x3 R;
  R.m[0] = A.m[0]*B.m[0] + A.m[1]*B.m[3];
  R.m[1] = A.m[0]*B.m[1] + A.m[1]*B.m[4];
  R.m[2] = A.m[0]*B.m[2] + A.m[1]*B.m[5] + A.m[2];

  R.m[3] = A.m[3]*B.m[0] + A.m[4]*B.m[3];
  R.m[4] = A.m[3]*B.m[1] + A.m[4]*B.m[4];
  R.m[5] = A.m[3]*B.m[2] + A.m[4]*B.m[5] + A.m[5];
  return R;
}

// Extract y[n] = P[0]*state[0] + P[1]*state[1] + P[2]
// where state = [y[-1], y[-2]] and the implicit 1 contributes P[2].
__device__ __forceinline__ double extract_y(const Mat3x3& P, double y_m1, double y_m2) {
  return P.m[0]*y_m1 + P.m[1]*y_m2 + P.m[2];
}

// ============================================================
// Block-level Hillis-Steele inclusive prefix scan in shared memory
// ============================================================

constexpr int BLOCK_SIZE = 512;

__global__ void prefix_scan_phase1(
    const double* __restrict__ f,        // [C, T] forcing function
    double* __restrict__ y,              // [C, T] output
    Mat3x3* __restrict__ block_agg,      // [C, num_blocks] per-block aggregate
    const double* __restrict__ state,    // [C, 2] initial state {y[-1], y[-2]}
    double a1, double a2,
    int T, int num_blocks) {

  // Each block processes BLOCK_SIZE samples for one channel.
  const int channel = blockIdx.y;
  const int block_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int global_n = block_id * BLOCK_SIZE + tid;

  __shared__ Mat3x3 sdata[2 * BLOCK_SIZE];

  // Build per-sample matrix or identity if out of range
  Mat3x3 my_mat;
  if (global_n < T) {
    double fn = f[channel * T + global_n];
    my_mat.m[0] = -a1; my_mat.m[1] = -a2; my_mat.m[2] = fn;
    my_mat.m[3] = 1.0; my_mat.m[4] = 0.0; my_mat.m[5] = 0.0;
  } else {
    my_mat = mat_identity();
  }

  // Hillis-Steele inclusive prefix scan
  int ping = 0;
  sdata[tid] = my_mat;
  __syncthreads();

  for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
    int pong = 1 - ping;
    if (tid >= offset) {
      sdata[pong * BLOCK_SIZE + tid] = mat_mul(sdata[ping * BLOCK_SIZE + tid], sdata[ping * BLOCK_SIZE + tid - offset]);
    } else {
      sdata[pong * BLOCK_SIZE + tid] = sdata[ping * BLOCK_SIZE + tid];
    }
    ping = pong;
    __syncthreads();
  }

  // Store the block aggregate (last thread's prefix = product of all matrices in block)
  if (tid == BLOCK_SIZE - 1) {
    block_agg[channel * num_blocks + block_id] = sdata[ping * BLOCK_SIZE + tid];
  }

  // For the first block, we can compute y directly using initial state
  if (block_id == 0 && global_n < T) {
    double y_m1 = state[channel * 2 + 0];
    double y_m2 = state[channel * 2 + 1];
    double yn = extract_y(sdata[ping * BLOCK_SIZE + tid], y_m1, y_m2);
    y[channel * T + global_n] = yn;
  }

  // Other blocks need the inter-block prefix (computed in phase 2)
  // Store the intra-block prefix for later use in phase 3
  // We reuse y[] to temporarily store f[n] for non-first blocks
  // Actually, we store the prefix matrix in block_agg and recompute in phase 3
}

__global__ void prefix_scan_phase2(
    Mat3x3* __restrict__ block_agg,  // [C, num_blocks] -- modified in-place
    int num_blocks) {
  // Sequential scan over block aggregates (num_blocks is typically small).
  // Each channel is handled by one thread.
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  // (caller ensures channel < C)

  for (int i = 1; i < num_blocks; ++i) {
    int idx = channel * num_blocks + i;
    int prev = channel * num_blocks + i - 1;
    block_agg[idx] = mat_mul(block_agg[idx], block_agg[prev]);
  }
}

__global__ void prefix_scan_phase3(
    const double* __restrict__ f,
    double* __restrict__ y,
    const Mat3x3* __restrict__ block_agg,
    const double* __restrict__ state,
    double a1, double a2,
    int T, int num_blocks) {

  // For blocks > 0: recompute intra-block scan and apply inter-block prefix.
  const int channel = blockIdx.y;
  const int block_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int global_n = block_id * BLOCK_SIZE + tid;

  if (block_id == 0) return;  // Already computed in phase 1
  if (global_n >= T) return;

  __shared__ Mat3x3 sdata[2 * BLOCK_SIZE];

  // Rebuild per-sample matrix
  double fn = f[channel * T + global_n];
  Mat3x3 my_mat;
  my_mat.m[0] = -a1; my_mat.m[1] = -a2; my_mat.m[2] = fn;
  my_mat.m[3] = 1.0; my_mat.m[4] = 0.0; my_mat.m[5] = 0.0;

  // Intra-block Hillis-Steele scan (same as phase 1)
  int ping = 0;
  sdata[tid] = my_mat;
  __syncthreads();

  for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
    int pong = 1 - ping;
    if (tid >= offset) {
      sdata[pong * BLOCK_SIZE + tid] = mat_mul(sdata[ping * BLOCK_SIZE + tid], sdata[ping * BLOCK_SIZE + tid - offset]);
    } else {
      sdata[pong * BLOCK_SIZE + tid] = sdata[ping * BLOCK_SIZE + tid];
    }
    ping = pong;
    __syncthreads();
  }

  // Compose with inter-block prefix: P_total = intra_prefix * inter_prefix
  // inter_prefix for block k = block_agg[k-1] (inclusive prefix of all previous blocks)
  Mat3x3 inter_prefix = block_agg[channel * num_blocks + block_id - 1];
  Mat3x3 total_prefix = mat_mul(sdata[ping * BLOCK_SIZE + tid], inter_prefix);

  double y_m1 = state[channel * 2 + 0];
  double y_m2 = state[channel * 2 + 1];
  double yn = extract_y(total_prefix, y_m1, y_m2);
  y[channel * T + global_n] = yn;
}

// ============================================================
// Sequential kernel for short signals (T < PARALLEL_SCAN_THRESHOLD)
// ============================================================

__global__ void sequential_biquad_kernel(
    const double* __restrict__ f,
    double* __restrict__ y,
    const double* __restrict__ state,
    double a1, double a2,
    int T) {

  const int channel = blockIdx.x;
  double y_m1 = state[channel * 2 + 0];
  double y_m2 = state[channel * 2 + 1];

  for (int n = 0; n < T; ++n) {
    double fn = f[channel * T + n];
    double yn = fn - a1 * y_m1 - a2 * y_m2;
    y[channel * T + n] = yn;
    y_m2 = y_m1;
    y_m1 = yn;
  }
}

// ============================================================
// Public API
// ============================================================

torch::Tensor compute_forcing(const torch::Tensor& x, const torch::Tensor& b) {
  // f[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2]
  // Implemented as 1D convolution with kernel [b2, b1, b0] (flipped for conv).
  auto b_flip = b.flip(0).reshape({1, 1, 3}).to(x.dtype());
  auto x_2d = x.unsqueeze(1);  // [C, 1, T]
  auto padded = torch::nn::functional::pad(x_2d,
    torch::nn::functional::PadFuncOptions({2, 0}));
  auto f = torch::conv1d(padded, b_flip.expand({x.size(0), 1, 3}),
    /*bias=*/{}, /*stride=*/1, /*padding=*/0, /*dilation=*/1, /*groups=*/x.size(0));
  return f.squeeze(1);  // [C, T]
}

std::tuple<torch::Tensor, torch::Tensor> parallel_biquad_scan(
    const torch::Tensor& f,
    double a1,
    double a2,
    const torch::Tensor& state) {

  // Caller already provides float64 tensors; just ensure contiguity.
  auto f_cont = f.contiguous();
  auto state_cont = state.contiguous();

  const int64_t C = f_cont.size(0);
  const int64_t T = f_cont.size(1);

  auto y = torch::empty({C, T}, f_cont.options());

  const double* f_ptr = f_cont.data_ptr<double>();
  double* y_ptr = y.data_ptr<double>();
  const double* state_ptr = state_cont.data_ptr<double>();

  if (T <= 2048) {
    // Sequential kernel for short signals
    sequential_biquad_kernel<<<C, 1>>>(f_ptr, y_ptr, state_ptr, a1, a2, T);
  } else {
    const int num_blocks = (T + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate block aggregates
    auto block_agg = torch::empty({C * num_blocks * 6}, torch::dtype(torch::kFloat64).device(f.device()));
    Mat3x3* agg_ptr = reinterpret_cast<Mat3x3*>(block_agg.data_ptr<double>());

    // Phase 1: intra-block scan + store aggregates
    dim3 grid1(num_blocks, C);
    prefix_scan_phase1<<<grid1, BLOCK_SIZE>>>(
        f_ptr, y_ptr, agg_ptr, state_ptr, a1, a2, T, num_blocks);

    // Phase 2: scan over block aggregates (sequential, one thread per channel)
    const int threads_p2 = std::min(static_cast<int>(C), 256);
    const int blocks_p2 = (C + threads_p2 - 1) / threads_p2;
    prefix_scan_phase2<<<blocks_p2, threads_p2>>>(agg_ptr, num_blocks);

    // Phase 3: finalize blocks > 0
    dim3 grid3(num_blocks, C);
    prefix_scan_phase3<<<grid3, BLOCK_SIZE>>>(
        f_ptr, y_ptr, agg_ptr, state_ptr, a1, a2, T, num_blocks);
  }

  // Extract updated state: [y[T-1], y[T-2]]
  auto y_last = y.index({torch::indexing::Slice(), -1}).unsqueeze(1);  // [C, 1]
  auto y_prev = (T >= 2) ?
      y.index({torch::indexing::Slice(), -2}).unsqueeze(1) :
      torch::zeros({C, 1}, y.options());
  auto new_st = torch::cat({y_last, y_prev}, 1);  // [C, 2]

  return std::make_tuple(y, new_st);
}

}  // namespace torchfx
