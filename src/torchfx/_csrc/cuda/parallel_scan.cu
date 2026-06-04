#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
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
// We use a Blelloch (work-efficient) up-sweep/down-sweep prefix scan.
//
// Optimization: Since the bottom row is always [0, 0, 1], we only store
// and multiply the 2x3 top portion (6 elements per matrix).
//
// Kernels are templated on ``scalar_t`` (float or double): the host entry
// points dispatch on the input tensor's dtype via AT_DISPATCH_FLOATING_TYPES,
// so a float32 input runs the FP32 path natively instead of being upcast to
// float64. The scalar coefficients arrive as ``double`` (host constants) and
// are cast to ``scalar_t`` at launch.

namespace torchfx {

// A "reduced" 3x3 matrix where row 2 = [0, 0, 1].
// We store rows 0 and 1 as 6 elements.
// Layout: m[0]=r0c0, m[1]=r0c1, m[2]=r0c2,
//         m[3]=r1c0, m[4]=r1c1, m[5]=r1c2
template <typename scalar_t>
struct Mat3x3 {
  scalar_t m[6];
};

template <typename scalar_t>
__device__ __forceinline__ Mat3x3<scalar_t> mat_identity() {
  Mat3x3<scalar_t> I;
  I.m[0] = scalar_t(1); I.m[1] = scalar_t(0); I.m[2] = scalar_t(0);
  I.m[3] = scalar_t(0); I.m[4] = scalar_t(1); I.m[5] = scalar_t(0);
  return I;
}

// Multiply A * B where both have implicit row 2 = [0, 0, 1].
// Result also has implicit row 2 = [0, 0, 1].
template <typename scalar_t>
__device__ __forceinline__ Mat3x3<scalar_t> mat_mul(const Mat3x3<scalar_t>& A, const Mat3x3<scalar_t>& B) {
  Mat3x3<scalar_t> R;
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
template <typename scalar_t>
__device__ __forceinline__ scalar_t extract_y(const Mat3x3<scalar_t>& P, scalar_t y_m1, scalar_t y_m2) {
  return P.m[0]*y_m1 + P.m[1]*y_m2 + P.m[2];
}

// ============================================================
// Blelloch work-efficient inclusive prefix scan
// ============================================================

constexpr int BLOCK_SIZE = 512;
constexpr int LOG2_BLOCK = 9;  // log2(512)

// Blelloch inclusive prefix scan in shared memory.
// Input: sdata[tid] contains element for this thread.
// Output: sdata[tid] contains inclusive prefix product M[tid]*...*M[0].
// original: the element at this thread's position (saved before scan modifies sdata).
// Returns: the block's total product (valid for all threads after final sync).
template <typename scalar_t>
__device__ Mat3x3<scalar_t> blelloch_inclusive_scan(Mat3x3<scalar_t>* sdata, const Mat3x3<scalar_t>& original, int tid) {

  // ---- Up-sweep (reduce) ----
  for (int d = 0; d < LOG2_BLOCK; d++) {
    int stride = 2 << d;      // 2, 4, 8, ..., BLOCK_SIZE
    int half = 1 << d;        // 1, 2, 4, ..., BLOCK_SIZE/2
    int num_active = BLOCK_SIZE >> (d + 1);
    if (tid < num_active) {
      int right = (tid + 1) * stride - 1;
      int left = right - half;
      sdata[right] = mat_mul<scalar_t>(sdata[right], sdata[left]);
    }
    __syncthreads();
  }

  // Save block aggregate (total product) before down-sweep clobbers it
  Mat3x3<scalar_t> block_total = sdata[BLOCK_SIZE - 1];

  // ---- Set root to identity for exclusive scan ----
  if (tid == 0) {
    sdata[BLOCK_SIZE - 1] = mat_identity<scalar_t>();
  }
  __syncthreads();

  // ---- Down-sweep ----
  for (int d = LOG2_BLOCK - 1; d >= 0; d--) {
    int stride = 2 << d;
    int half = 1 << d;
    int num_active = BLOCK_SIZE >> (d + 1);
    if (tid < num_active) {
      int right = (tid + 1) * stride - 1;
      int left = right - half;
      Mat3x3<scalar_t> temp = sdata[left];
      sdata[left] = sdata[right];
      sdata[right] = mat_mul<scalar_t>(temp, sdata[right]);
    }
    __syncthreads();
  }

  // ---- Convert exclusive → inclusive ----
  // exclusive[i] = M[i-1]*...*M[0], inclusive[i] = M[i] * exclusive[i]
  sdata[tid] = mat_mul<scalar_t>(original, sdata[tid]);
  __syncthreads();

  return block_total;
}

template <typename scalar_t>
__global__ void prefix_scan_phase1(
    const scalar_t* __restrict__ f,        // [C, T] forcing function
    scalar_t* __restrict__ y,              // [C, T] output
    Mat3x3<scalar_t>* __restrict__ block_agg,  // [C, num_blocks] per-block aggregate
    const scalar_t* __restrict__ state,    // [C, 2] initial state {y[-1], y[-2]}
    scalar_t a1, scalar_t a2,
    int T, int num_blocks) {

  const int channel = blockIdx.y;
  const int block_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int global_n = block_id * BLOCK_SIZE + tid;

  __shared__ Mat3x3<scalar_t> sdata[BLOCK_SIZE];

  // Build per-sample matrix or identity if out of range
  Mat3x3<scalar_t> my_mat;
  if (global_n < T) {
    scalar_t fn = f[channel * T + global_n];
    my_mat.m[0] = -a1; my_mat.m[1] = -a2; my_mat.m[2] = fn;
    my_mat.m[3] = scalar_t(1); my_mat.m[4] = scalar_t(0); my_mat.m[5] = scalar_t(0);
  } else {
    my_mat = mat_identity<scalar_t>();
  }

  sdata[tid] = my_mat;
  __syncthreads();

  Mat3x3<scalar_t> block_total = blelloch_inclusive_scan<scalar_t>(sdata, my_mat, tid);

  // Store block aggregate
  if (tid == 0) {
    block_agg[channel * num_blocks + block_id] = block_total;
  }

  // For the first block, compute y directly using initial state
  if (block_id == 0 && global_n < T) {
    scalar_t y_m1 = state[channel * 2 + 0];
    scalar_t y_m2 = state[channel * 2 + 1];
    scalar_t yn = extract_y<scalar_t>(sdata[tid], y_m1, y_m2);
    y[channel * T + global_n] = yn;
  }
}

template <typename scalar_t>
__global__ void prefix_scan_phase2(
    Mat3x3<scalar_t>* __restrict__ block_agg,  // [C, num_blocks] -- modified in-place
    int num_blocks) {
  // Sequential scan over block aggregates (num_blocks is typically small).
  // Each channel is handled by one thread.
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  // (caller ensures channel < C)

  for (int i = 1; i < num_blocks; ++i) {
    int idx = channel * num_blocks + i;
    int prev = channel * num_blocks + i - 1;
    block_agg[idx] = mat_mul<scalar_t>(block_agg[idx], block_agg[prev]);
  }
}

template <typename scalar_t>
__global__ void prefix_scan_phase3(
    const scalar_t* __restrict__ f,
    scalar_t* __restrict__ y,
    const Mat3x3<scalar_t>* __restrict__ block_agg,
    const scalar_t* __restrict__ state,
    scalar_t a1, scalar_t a2,
    int T, int num_blocks) {

  // For blocks > 0: recompute intra-block scan and apply inter-block prefix.
  const int channel = blockIdx.y;
  const int block_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int global_n = block_id * BLOCK_SIZE + tid;

  if (block_id == 0) return;  // Already computed in phase 1
  if (global_n >= T) return;

  __shared__ Mat3x3<scalar_t> sdata[BLOCK_SIZE];

  // Rebuild per-sample matrix
  scalar_t fn = f[channel * T + global_n];
  Mat3x3<scalar_t> my_mat;
  my_mat.m[0] = -a1; my_mat.m[1] = -a2; my_mat.m[2] = fn;
  my_mat.m[3] = scalar_t(1); my_mat.m[4] = scalar_t(0); my_mat.m[5] = scalar_t(0);

  sdata[tid] = my_mat;
  __syncthreads();

  blelloch_inclusive_scan<scalar_t>(sdata, my_mat, tid);

  // Compose with inter-block prefix: P_total = intra_prefix * inter_prefix
  // inter_prefix for block k = block_agg[k-1] (inclusive prefix of all previous blocks)
  Mat3x3<scalar_t> inter_prefix = block_agg[channel * num_blocks + block_id - 1];
  Mat3x3<scalar_t> total_prefix = mat_mul<scalar_t>(sdata[tid], inter_prefix);

  scalar_t y_m1 = state[channel * 2 + 0];
  scalar_t y_m2 = state[channel * 2 + 1];
  scalar_t yn = extract_y<scalar_t>(total_prefix, y_m1, y_m2);
  y[channel * T + global_n] = yn;
}

// ============================================================
// Sequential kernel for short signals (T < PARALLEL_SCAN_THRESHOLD)
// Each thread handles one channel; multiple channels per block for occupancy.
// ============================================================

constexpr int SEQ_THREADS_PER_BLOCK = 128;

template <typename scalar_t>
__global__ void sequential_biquad_kernel(
    const scalar_t* __restrict__ f,
    scalar_t* __restrict__ y,
    const scalar_t* __restrict__ state,
    scalar_t a1, scalar_t a2,
    int T, int C) {

  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= C) return;

  scalar_t y_m1 = state[channel * 2 + 0];
  scalar_t y_m2 = state[channel * 2 + 1];

  for (int n = 0; n < T; ++n) {
    scalar_t fn = f[channel * T + n];
    scalar_t yn = fn - a1 * y_m1 - a2 * y_m2;
    y[channel * T + n] = yn;
    y_m2 = y_m1;
    y_m1 = yn;
  }
}

// ============================================================
// Fused sequential kernel: FIR forcing + IIR recurrence in ONE launch.
// Folds the forcing pass into the sequential biquad recurrence so the small-T
// branch is a single kernel (was forcing_kernel + sequential_biquad_kernel).
// Each thread handles one channel; multiple channels per block for occupancy.
// ============================================================
template <typename scalar_t>
__global__ void fused_sequential_kernel(
    const scalar_t* __restrict__ x,     // [C, T]
    const scalar_t* __restrict__ sx,    // [C, 2] = {x[-1], x[-2]}
    const scalar_t* __restrict__ sy,    // [C, 2] = {y[-1], y[-2]}
    scalar_t b0, scalar_t b1, scalar_t b2,
    scalar_t a1, scalar_t a2,
    scalar_t* __restrict__ y,           // [C, T]
    int T, int C) {

  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= C) return;

  scalar_t x_m1 = sx[channel * 2 + 0];
  scalar_t x_m2 = sx[channel * 2 + 1];
  scalar_t y_m1 = sy[channel * 2 + 0];
  scalar_t y_m2 = sy[channel * 2 + 1];

  for (int n = 0; n < T; ++n) {
    const scalar_t xn = x[channel * T + n];
    const scalar_t fn = b0 * xn + b1 * x_m1 + b2 * x_m2;  // forcing, inline
    const scalar_t yn = fn - a1 * y_m1 - a2 * y_m2;       // recurrence
    y[channel * T + n] = yn;
    x_m2 = x_m1; x_m1 = xn;
    y_m2 = y_m1; y_m1 = yn;
  }
}

// ============================================================
// Custom CUDA kernel for 3-tap FIR with state prepend.
// Fuses the cat + conv1d into a single kernel launch, eliminating
// ~5 intermediate tensor operations and cuDNN dispatch overhead.
// ============================================================
template <typename scalar_t>
__global__ void forcing_kernel(
    const scalar_t* __restrict__ x,     // [C, T]
    const scalar_t* __restrict__ sx,    // [C, 2] = {x[-1], x[-2]}
    scalar_t b0, scalar_t b1, scalar_t b2,
    scalar_t* __restrict__ f,           // [C, T]
    int T, int total_elements) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  const int c = idx / T;
  const int n = idx % T;

  const scalar_t xn = x[c * T + n];
  scalar_t xn_1, xn_2;

  if (n >= 2) {
    xn_1 = x[c * T + n - 1];
    xn_2 = x[c * T + n - 2];
  } else if (n == 1) {
    xn_1 = x[c * T + 0];
    xn_2 = sx[c * 2 + 0];  // x[-1]
  } else {  // n == 0
    xn_1 = sx[c * 2 + 0];  // x[-1]
    xn_2 = sx[c * 2 + 1];  // x[-2]
  }

  f[c * T + n] = b0 * xn + b1 * xn_1 + b2 * xn_2;
}

// ============================================================
// Public API (host)
// ============================================================

void compute_forcing_into(
    const torch::Tensor& x,
    double b0, double b1, double b2,
    const torch::Tensor& state_x,
    torch::Tensor& f_out) {
  // f[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] with state prepend for n=0,1.
  auto x_cont = x.contiguous();
  auto sx_cont = state_x.contiguous();

  const int64_t C = x_cont.size(0);
  const int64_t T = x_cont.size(1);
  const int total = static_cast<int>(C * T);

  constexpr int THREADS = 256;
  const int blocks = (total + THREADS - 1) / THREADS;

  // Launch on the current stream (not the default stream) so the kernels are
  // recorded under torch.cuda.graph capture, which runs on a side stream.
  const auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x_cont.scalar_type(), "compute_forcing", [&] {
    forcing_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
        x_cont.data_ptr<scalar_t>(),
        sx_cont.data_ptr<scalar_t>(),
        static_cast<scalar_t>(b0), static_cast<scalar_t>(b1), static_cast<scalar_t>(b2),
        f_out.data_ptr<scalar_t>(),
        static_cast<int>(T), total);
  });
}

torch::Tensor compute_forcing(
    const torch::Tensor& x,
    double b0, double b1, double b2,
    const torch::Tensor& state_x) {
  auto f = torch::empty({x.size(0), x.size(1)}, x.options());
  compute_forcing_into(x, b0, b1, b2, state_x, f);
  return f;
}

void parallel_biquad_scan_into(
    const torch::Tensor& f,
    double a1,
    double a2,
    const torch::Tensor& state,
    int threshold,
    torch::Tensor& y_out,
    torch::Tensor& block_agg) {

  // Caller provides matching-dtype tensors; just ensure contiguity.
  auto f_cont = f.contiguous();
  auto state_cont = state.contiguous();

  const int64_t C = f_cont.size(0);
  const int64_t T = f_cont.size(1);

  // Launch on the current stream (not the default stream) so the kernels are
  // recorded under torch.cuda.graph capture, which runs on a side stream.
  const auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(f_cont.scalar_type(), "parallel_biquad_scan", [&] {
    const scalar_t* f_ptr = f_cont.data_ptr<scalar_t>();
    scalar_t* y_ptr = y_out.data_ptr<scalar_t>();
    const scalar_t* state_ptr = state_cont.data_ptr<scalar_t>();
    const scalar_t a1s = static_cast<scalar_t>(a1);
    const scalar_t a2s = static_cast<scalar_t>(a2);

    if (T <= threshold) {
      // Sequential kernel for short signals — multiple channels per block
      const int seq_blocks = (C + SEQ_THREADS_PER_BLOCK - 1) / SEQ_THREADS_PER_BLOCK;
      sequential_biquad_kernel<scalar_t><<<seq_blocks, SEQ_THREADS_PER_BLOCK, 0, stream>>>(
          f_ptr, y_ptr, state_ptr, a1s, a2s, static_cast<int>(T), static_cast<int>(C));
    } else {
      const int num_blocks = (T + BLOCK_SIZE - 1) / BLOCK_SIZE;

      // Caller-provided block aggregates: C * num_blocks reduced 3x3 matrices
      // (6 scalars each), in the input dtype so the FP32 path stays FP32.
      Mat3x3<scalar_t>* agg_ptr = reinterpret_cast<Mat3x3<scalar_t>*>(block_agg.data_ptr<scalar_t>());

      // Phase 1: intra-block scan + store aggregates
      dim3 grid1(num_blocks, C);
      prefix_scan_phase1<scalar_t><<<grid1, BLOCK_SIZE, 0, stream>>>(
          f_ptr, y_ptr, agg_ptr, state_ptr, a1s, a2s, static_cast<int>(T), num_blocks);

      // Phase 2: scan over block aggregates (sequential, one thread per channel)
      const int threads_p2 = std::min(static_cast<int>(C), 256);
      const int blocks_p2 = (C + threads_p2 - 1) / threads_p2;
      prefix_scan_phase2<scalar_t><<<blocks_p2, threads_p2, 0, stream>>>(agg_ptr, num_blocks);

      // Phase 3: finalize blocks > 0
      dim3 grid3(num_blocks, C);
      prefix_scan_phase3<scalar_t><<<grid3, BLOCK_SIZE, 0, stream>>>(
          f_ptr, y_ptr, agg_ptr, state_ptr, a1s, a2s, static_cast<int>(T), num_blocks);
    }
  });
  // No state extraction here — the caller reads the new state from ``y_out``.
}

std::tuple<torch::Tensor, torch::Tensor> parallel_biquad_scan(
    const torch::Tensor& f,
    double a1,
    double a2,
    const torch::Tensor& state,
    int threshold) {
  const int64_t C = f.size(0);
  const int64_t T = f.size(1);
  auto y = torch::empty({C, T}, f.options());
  const int num_blocks = (static_cast<int>(T) + BLOCK_SIZE - 1) / BLOCK_SIZE;
  auto block_agg = torch::empty({C * num_blocks * 6}, f.options());
  parallel_biquad_scan_into(f, a1, a2, state, threshold, y, block_agg);

  // Extract updated state: [y[T-1], y[T-2]].
  auto y_last = y.index({torch::indexing::Slice(), -1}).unsqueeze(1);  // [C, 1]
  auto y_prev = (T >= 2) ?
      y.index({torch::indexing::Slice(), -2}).unsqueeze(1) :
      torch::zeros({C, 1}, y.options());
  auto new_st = torch::cat({y_last, y_prev}, 1);  // [C, 2]
  return std::make_tuple(y, new_st);
}

// Fused per-section SOS path: forcing folded into the scan.
//
// Replaces the per-section (compute_forcing_into + parallel_biquad_scan_into) pair
// with a single fused launch:
//   - T <= threshold : fused_sequential_kernel (forcing inline)        -- 1 launch.
//   - T  > threshold : single-pass decoupled-look-back scan (inline)   -- 1 launch
//                      (NOT YET WIRED — falls back to the 3-phase path).
// Reads state_x (for forcing) and state_y (recurrence init); writes y_out. The
// caller updates the state tails in place afterwards (unchanged from the 3-phase
// path). All launches are on the current stream for CUDA-graph safety.
void fused_sos_scan_into(
    const torch::Tensor& x,
    double b0, double b1, double b2,
    double a1, double a2,
    const torch::Tensor& state_x,
    const torch::Tensor& state_y,
    int threshold,
    torch::Tensor& y_out,
    torch::Tensor& f_scratch,
    torch::Tensor& block_agg) {

  auto x_c = x.contiguous();
  auto sx = state_x.contiguous();
  auto sy = state_y.contiguous();
  const int64_t C = x_c.size(0);
  const int64_t T = x_c.size(1);
  const auto stream = c10::cuda::getCurrentCUDAStream();

  if (T <= threshold) {
    const int blocks = (static_cast<int>(C) + SEQ_THREADS_PER_BLOCK - 1) / SEQ_THREADS_PER_BLOCK;
    AT_DISPATCH_FLOATING_TYPES(x_c.scalar_type(), "fused_sequential", [&] {
      fused_sequential_kernel<scalar_t><<<blocks, SEQ_THREADS_PER_BLOCK, 0, stream>>>(
          x_c.data_ptr<scalar_t>(), sx.data_ptr<scalar_t>(), sy.data_ptr<scalar_t>(),
          static_cast<scalar_t>(b0), static_cast<scalar_t>(b1), static_cast<scalar_t>(b2),
          static_cast<scalar_t>(a1), static_cast<scalar_t>(a2),
          y_out.data_ptr<scalar_t>(), static_cast<int>(T), static_cast<int>(C));
    });
  } else {
    // Increment B will replace this with the single-pass fused_scan_kernel.
    compute_forcing_into(x_c, b0, b1, b2, sx, f_scratch);
    parallel_biquad_scan_into(f_scratch, a1, a2, sy, threshold, y_out, block_agg);
  }
}

}  // namespace torchfx
