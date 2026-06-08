#include <torch/torch.h>
#include <omp.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <tuple>
#include <vector>

// CPU implementation of biquad Direct Form 1.
// Vectorized across channels, sequential across time.
// This is significantly faster than the Python for-loop.

template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> biquad_forward_cpu_impl(
    const torch::Tensor& x,       // [C, T]
    const torch::Tensor& b,       // [3]
    scalar_t a1,
    scalar_t a2,
    const torch::Tensor& state_x, // [C, 2]
    const torch::Tensor& state_y) {

  auto sx = state_x.clone();
  auto sy = state_y.clone();

  const int64_t C = x.size(0);
  const int64_t T = x.size(1);

  auto y = torch::empty_like(x);

  auto x_ptr = x.accessor<scalar_t, 2>();
  auto y_ptr = y.accessor<scalar_t, 2>();
  auto sx_ptr = sx.accessor<scalar_t, 2>();
  auto sy_ptr = sy.accessor<scalar_t, 2>();
  auto b_ptr = b.accessor<scalar_t, 1>();

  const scalar_t b0 = b_ptr[0];
  const scalar_t b1 = b_ptr[1];
  const scalar_t b2 = b_ptr[2];

  #pragma omp parallel for schedule(static) if(C > 1)
  for (int64_t c = 0; c < C; ++c) {
    scalar_t sx0 = sx_ptr[c][0];  // x[n-1]
    scalar_t sx1 = sx_ptr[c][1];  // x[n-2]
    scalar_t sy0 = sy_ptr[c][0];  // y[n-1]
    scalar_t sy1 = sy_ptr[c][1];  // y[n-2]

    for (int64_t n = 0; n < T; ++n) {
      const scalar_t xn = x_ptr[c][n];
      const scalar_t yn = b0 * xn + b1 * sx0 + b2 * sx1 - a1 * sy0 - a2 * sy1;
      y_ptr[c][n] = yn;

      sx1 = sx0;
      sx0 = xn;
      sy1 = sy0;
      sy0 = yn;
    }

    sx_ptr[c][0] = sx0;
    sx_ptr[c][1] = sx1;
    sy_ptr[c][0] = sy0;
    sy_ptr[c][1] = sy1;
  }

  return std::make_tuple(y, sx, sy);
}

template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sos_forward_cpu_impl(
    const torch::Tensor& x,       // [C, T]
    const torch::Tensor& sos,     // [K, 6]
    const torch::Tensor& state_x, // [K, C, 2]
    const torch::Tensor& state_y) {

  auto sx = state_x.clone();
  auto sy = state_y.clone();

  const int64_t K = sos.size(0);
  const int64_t C = x.size(0);
  const int64_t T = x.size(1);

  auto sos_acc = sos.accessor<scalar_t, 2>();

  struct SosCoeffs {
    scalar_t b0;
    scalar_t b1;
    scalar_t b2;
    scalar_t a1;
    scalar_t a2;
  };
  std::vector<SosCoeffs> coeffs(K);
  for (int64_t s = 0; s < K; ++s) {
    coeffs[s] = {
        sos_acc[s][0],
        sos_acc[s][1],
        sos_acc[s][2],
        sos_acc[s][4],
        sos_acc[s][5],
    };  // sos[s][3] is a0 = 1
  }

  auto y = torch::empty_like(x);
  auto x_ptr = x.accessor<scalar_t, 2>();
  auto y_ptr = y.accessor<scalar_t, 2>();
  auto sx_acc = sx.accessor<scalar_t, 3>();  // [K, C, 2]
  auto sy_acc = sy.accessor<scalar_t, 3>();  // [K, C, 2]

  static constexpr int64_t STACK_MAX = 16;

  #pragma omp parallel for schedule(static) if(C > 1)
  for (int64_t c = 0; c < C; ++c) {
    scalar_t stack_sx0[STACK_MAX], stack_sx1[STACK_MAX],
             stack_sy0[STACK_MAX], stack_sy1[STACK_MAX];

    std::vector<scalar_t> heap_sx0, heap_sx1, heap_sy0, heap_sy1;
    scalar_t *sec_sx0, *sec_sx1, *sec_sy0, *sec_sy1;
    if (K <= STACK_MAX) {
      sec_sx0 = stack_sx0;
      sec_sx1 = stack_sx1;
      sec_sy0 = stack_sy0;
      sec_sy1 = stack_sy1;
    } else {
      heap_sx0.resize(K);
      heap_sx1.resize(K);
      heap_sy0.resize(K);
      heap_sy1.resize(K);
      sec_sx0 = heap_sx0.data();
      sec_sx1 = heap_sx1.data();
      sec_sy0 = heap_sy0.data();
      sec_sy1 = heap_sy1.data();
    }

    for (int64_t s = 0; s < K; ++s) {
      sec_sx0[s] = sx_acc[s][c][0];
      sec_sx1[s] = sx_acc[s][c][1];
      sec_sy0[s] = sy_acc[s][c][0];
      sec_sy1[s] = sy_acc[s][c][1];
    }

    for (int64_t n = 0; n < T; ++n) {
      scalar_t val = x_ptr[c][n];

      for (int64_t s = 0; s < K; ++s) {
        const auto& co = coeffs[s];
        const scalar_t yn =
            co.b0 * val + co.b1 * sec_sx0[s] + co.b2 * sec_sx1[s] - co.a1 * sec_sy0[s] -
            co.a2 * sec_sy1[s];
        sec_sx1[s] = sec_sx0[s];
        sec_sx0[s] = val;
        sec_sy1[s] = sec_sy0[s];
        sec_sy0[s] = yn;
        val = yn;
      }

      y_ptr[c][n] = val;
    }

    for (int64_t s = 0; s < K; ++s) {
      sx_acc[s][c][0] = sec_sx0[s];
      sx_acc[s][c][1] = sec_sx1[s];
      sy_acc[s][c][0] = sec_sy0[s];
      sy_acc[s][c][1] = sec_sy1[s];
    }
  }

  return std::make_tuple(y, sx, sy);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> biquad_forward_cpu(
    const torch::Tensor& x,         // [C, T]
    const torch::Tensor& b,         // [3]
    double a1,
    double a2,
    const torch::Tensor& state_x,   // [C, 2]
    const torch::Tensor& state_y) { // [C, 2]

  TORCH_CHECK(x.dim() == 2, "biquad_forward_cpu: x must be [C, T]");
  TORCH_CHECK(b.numel() == 3, "biquad_forward_cpu: b must have 3 coefficients");
  TORCH_CHECK(x.is_floating_point(), "biquad_forward_cpu: x must be floating-point");

  const auto exec_dtype = (x.scalar_type() == torch::kFloat64) ? torch::kFloat64 : torch::kFloat32;

  const auto x_exec = x.to(exec_dtype).contiguous();
  const auto b_exec = b.to(exec_dtype).contiguous();
  const auto sx_exec = state_x.to(exec_dtype).contiguous();
  const auto sy_exec = state_y.to(exec_dtype).contiguous();

  if (exec_dtype == torch::kFloat32) {
    return biquad_forward_cpu_impl<float>(
        x_exec,
        b_exec,
        static_cast<float>(a1),
        static_cast<float>(a2),
        sx_exec,
        sy_exec);
  }

  return biquad_forward_cpu_impl<double>(x_exec, b_exec, a1, a2, sx_exec, sy_exec);
}

// Cache-blocked cross-channel SIMD SOS cascade (Roadmap Epic 4.6 / #24).
//
// The DF1 SOS recurrence is serial in time and across sections but INDEPENDENT
// across channels, with coefficients shared by all channels. The scalar kernel
// parallelises channels over OpenMP threads; once C > cores each thread processes
// several channels SERIALLY (the time the throughput goes linear in C).
//
// This path packs a group of W channels into SIMD lanes: it transposes a small
// [W, B] tile into channel-contiguous [B, W] (kept in L1 — unlike the reverted F1
// attempt that transposed the whole [C, T] and went memory-bound), runs the
// recurrence with the per-time-step channel loop auto-vectorising (#pragma omp simd),
// and transposes the output tile back. OpenMP runs over channel groups so cores AND
// SIMD lanes are both used. Engaged only when C > num_threads (see the dispatcher).
template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sos_forward_cpu_simd_impl(
    const torch::Tensor& x,       // [C, T]
    const torch::Tensor& sos,     // [K, 6]
    const torch::Tensor& state_x, // [K, C, 2]
    const torch::Tensor& state_y) {

  auto sx = state_x.clone();
  auto sy = state_y.clone();

  const int64_t K = sos.size(0);
  const int64_t C = x.size(0);
  const int64_t T = x.size(1);

  auto sos_acc = sos.accessor<scalar_t, 2>();
  std::vector<scalar_t> cb0(K), cb1(K), cb2(K), ca1(K), ca2(K);
  for (int64_t s = 0; s < K; ++s) {
    cb0[s] = sos_acc[s][0];
    cb1[s] = sos_acc[s][1];
    cb2[s] = sos_acc[s][2];
    ca1[s] = sos_acc[s][4];
    ca2[s] = sos_acc[s][5];
  }

  auto y = torch::empty_like(x);
  const scalar_t* xp = x.data_ptr<scalar_t>();  // [C, T] contiguous
  scalar_t* yp = y.data_ptr<scalar_t>();
  auto sx_acc = sx.accessor<scalar_t, 3>();  // [K, C, 2]
  auto sy_acc = sy.accessor<scalar_t, 3>();

  // Group width = one SIMD vector (4 f32 in a 128-bit NEON/SSE register). Keeping it
  // narrow maximises the number of channel groups, so OpenMP fills the cores even at
  // moderate C (e.g. C=8 on 4 cores -> 2 groups -> 2 cores, not 1). The inner channel
  // loop auto-vectorises to one vector op per section per time step.
  constexpr int64_t W = 4;
  constexpr int64_t B = 256;  // time block; the W*B tile stays in L1

  #pragma omp parallel for schedule(static) if (C > W)
  for (int64_t c0 = 0; c0 < C; c0 += W) {
    const int64_t cw = std::min<int64_t>(W, C - c0);

    // Per-group, channel-contiguous section state [K][W] (unused lanes stay 0).
    std::vector<scalar_t> ssx0(K * W, 0), ssx1(K * W, 0), ssy0(K * W, 0), ssy1(K * W, 0);
    for (int64_t s = 0; s < K; ++s) {
      for (int64_t c = 0; c < cw; ++c) {
        ssx0[s * W + c] = sx_acc[s][c0 + c][0];
        ssx1[s * W + c] = sx_acc[s][c0 + c][1];
        ssy0[s * W + c] = sy_acc[s][c0 + c][0];
        ssy1[s * W + c] = sy_acc[s][c0 + c][1];
      }
    }

    std::vector<scalar_t> tile(B * W);

    for (int64_t bt = 0; bt < T; bt += B) {
      const int64_t bb = std::min<int64_t>(B, T - bt);
      if (cw < W) std::memset(tile.data(), 0, sizeof(scalar_t) * bb * W);

      // Load + transpose [cw, bb] -> tile[bb][W] (channel-contiguous per time step).
      for (int64_t c = 0; c < cw; ++c) {
        const scalar_t* xrow = xp + (c0 + c) * T + bt;
        for (int64_t n = 0; n < bb; ++n) tile[n * W + c] = xrow[n];
      }

      // Recurrence over the tile; the channel loop (constant W) auto-vectorises.
      for (int64_t n = 0; n < bb; ++n) {
        scalar_t* row = &tile[n * W];
        for (int64_t s = 0; s < K; ++s) {
          scalar_t* __restrict px0 = &ssx0[s * W];
          scalar_t* __restrict px1 = &ssx1[s * W];
          scalar_t* __restrict py0 = &ssy0[s * W];
          scalar_t* __restrict py1 = &ssy1[s * W];
          const scalar_t b0 = cb0[s], b1 = cb1[s], b2 = cb2[s], a1 = ca1[s], a2 = ca2[s];
          #pragma omp simd
          for (int64_t c = 0; c < W; ++c) {
            const scalar_t yn = b0 * row[c] + b1 * px0[c] + b2 * px1[c] - a1 * py0[c] - a2 * py1[c];
            px1[c] = px0[c];
            px0[c] = row[c];
            py1[c] = py0[c];
            py0[c] = yn;
            row[c] = yn;  // cascade into the next section / final output
          }
        }
      }

      // Transpose tile[bb][cw] -> y[c0..][bt..].
      for (int64_t c = 0; c < cw; ++c) {
        scalar_t* yrow = yp + (c0 + c) * T + bt;
        for (int64_t n = 0; n < bb; ++n) yrow[n] = tile[n * W + c];
      }
    }

    for (int64_t s = 0; s < K; ++s) {
      for (int64_t c = 0; c < cw; ++c) {
        sx_acc[s][c0 + c][0] = ssx0[s * W + c];
        sx_acc[s][c0 + c][1] = ssx1[s * W + c];
        sy_acc[s][c0 + c][0] = ssy0[s * W + c];
        sy_acc[s][c0 + c][1] = ssy1[s * W + c];
      }
    }
  }

  return std::make_tuple(y, sx, sy);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sos_forward_cpu(
    const torch::Tensor& x,         // [C, T]
    const torch::Tensor& sos,       // [K, 6]
    const torch::Tensor& state_x,   // [K, C, 2]
    const torch::Tensor& state_y) { // [K, C, 2]

  TORCH_CHECK(x.dim() == 2, "sos_forward_cpu: x must be [C, T]");
  TORCH_CHECK(sos.dim() == 2 && sos.size(1) == 6, "sos_forward_cpu: sos must be [K, 6]");
  TORCH_CHECK(x.is_floating_point(), "sos_forward_cpu: x must be floating-point");

  const auto exec_dtype = (x.scalar_type() == torch::kFloat64) ? torch::kFloat64 : torch::kFloat32;

  const auto x_exec = x.to(exec_dtype).contiguous();
  const auto sos_exec = sos.to(exec_dtype).contiguous();
  const auto sx_exec = state_x.to(exec_dtype).contiguous();
  const auto sy_exec = state_y.to(exec_dtype).contiguous();

  // Cross-channel SIMD helps only once channels outnumber cores (each thread then
  // runs several channels serially). Below that the scalar path is already optimal,
  // and the tile transposes would be pure overhead. TORCHFX_NO_SIMD=1 forces scalar;
  // TORCHFX_FORCE_SIMD=1 forces the SIMD path at any C (for testing/benchmarking).
  const char* no_simd = std::getenv("TORCHFX_NO_SIMD");
  const char* force_simd = std::getenv("TORCHFX_FORCE_SIMD");
  const int64_t C = x_exec.size(0);
  // omp_get_max_threads() is a runtime call; guard it so the extension still links
  // where OpenMP is unavailable (e.g. MSVC/cibuildwheel, which doesn't link the
  // runtime — only the #pragma omp directives, which it harmlessly ignores). Without
  // OpenMP the SIMD path still vectorises, just single-threaded, so gate on 1.
#ifdef _OPENMP
  const int64_t nthreads = static_cast<int64_t>(omp_get_max_threads());
#else
  const int64_t nthreads = 1;
#endif
  const bool use_simd =
      (no_simd == nullptr || no_simd[0] != '1') &&
      ((force_simd != nullptr && force_simd[0] == '1') || C > nthreads);

  if (exec_dtype == torch::kFloat32) {
    return use_simd ? sos_forward_cpu_simd_impl<float>(x_exec, sos_exec, sx_exec, sy_exec)
                    : sos_forward_cpu_impl<float>(x_exec, sos_exec, sx_exec, sy_exec);
  }

  return use_simd ? sos_forward_cpu_simd_impl<double>(x_exec, sos_exec, sx_exec, sy_exec)
                  : sos_forward_cpu_impl<double>(x_exec, sos_exec, sx_exec, sy_exec);
}
