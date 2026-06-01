#include <torch/torch.h>
#include <omp.h>
#include <algorithm>
#include <cstdlib>
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

// Cross-channel SIMD SOS cascade.
//
// The DF1 recurrence is serial in time and across sections but INDEPENDENT across
// channels. The scalar kernel above parallelises channels across OpenMP threads but
// each thread is scalar. Here we instead make the channel axis the inner, contiguous,
// auto-vectorisable loop: one AVX2/NEON vector op advances a whole tile of channels'
// recurrence per instruction. This needs channels contiguous, so the signal is
// transposed to [T, C]; OpenMP then runs over channel TILES so cores and SIMD lanes
// are both used. The win grows with channels-per-core, so it pays off most at high
// channel counts and on few-core edge devices (e.g. the Raspberry Pi 5).
template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sos_forward_cpu_simd_impl(
    const torch::Tensor& x,       // [C, T]
    const torch::Tensor& sos,     // [K, 6]
    const torch::Tensor& state_x, // [K, C, 2]
    const torch::Tensor& state_y) {

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

  // Channel-contiguous layout so the per-time-step channel loop vectorises.
  auto xt = x.t().contiguous();                 // [T, C]
  auto yt = torch::empty({T, C}, x.options());  // [T, C]
  auto new_sx = state_x.clone();                // [K, C, 2]
  auto new_sy = state_y.clone();

  const scalar_t* xt_p = xt.data_ptr<scalar_t>();
  scalar_t* yt_p = yt.data_ptr<scalar_t>();
  scalar_t* sx_p = new_sx.data_ptr<scalar_t>();
  scalar_t* sy_p = new_sy.data_ptr<scalar_t>();

  constexpr int64_t TILE = 8;  // channels per OpenMP task (SIMD-friendly)

  #pragma omp parallel for schedule(static) if (C > TILE)
  for (int64_t c0 = 0; c0 < C; c0 += TILE) {
    const int64_t cw = std::min<int64_t>(TILE, C - c0);

    // Per-tile, channel-contiguous section state: [K][cw].
    std::vector<scalar_t> sx0(K * cw), sx1(K * cw), sy0(K * cw), sy1(K * cw), in(cw);
    for (int64_t s = 0; s < K; ++s) {
      for (int64_t c = 0; c < cw; ++c) {
        const int64_t base = (s * C + (c0 + c)) * 2;
        sx0[s * cw + c] = sx_p[base + 0];
        sx1[s * cw + c] = sx_p[base + 1];
        sy0[s * cw + c] = sy_p[base + 0];
        sy1[s * cw + c] = sy_p[base + 1];
      }
    }

    for (int64_t n = 0; n < T; ++n) {
      const scalar_t* xrow = xt_p + n * C + c0;  // [cw] contiguous
      for (int64_t c = 0; c < cw; ++c) in[c] = xrow[c];

      for (int64_t s = 0; s < K; ++s) {
        const scalar_t B0 = cb0[s], B1 = cb1[s], B2 = cb2[s], A1 = ca1[s], A2 = ca2[s];
        scalar_t* __restrict px0 = &sx0[s * cw];
        scalar_t* __restrict px1 = &sx1[s * cw];
        scalar_t* __restrict py0 = &sy0[s * cw];
        scalar_t* __restrict py1 = &sy1[s * cw];
        scalar_t* __restrict pin = in.data();
        #pragma omp simd
        for (int64_t c = 0; c < cw; ++c) {  // channels are independent -> vectorises
          const scalar_t v = pin[c];
          const scalar_t yn = B0 * v + B1 * px0[c] + B2 * px1[c] - A1 * py0[c] - A2 * py1[c];
          px1[c] = px0[c];
          px0[c] = v;
          py1[c] = py0[c];
          py0[c] = yn;
          pin[c] = yn;
        }
      }

      scalar_t* yrow = yt_p + n * C + c0;
      for (int64_t c = 0; c < cw; ++c) yrow[c] = in[c];
    }

    for (int64_t s = 0; s < K; ++s) {
      for (int64_t c = 0; c < cw; ++c) {
        const int64_t base = (s * C + (c0 + c)) * 2;
        sx_p[base + 0] = sx0[s * cw + c];
        sx_p[base + 1] = sx1[s * cw + c];
        sy_p[base + 0] = sy0[s * cw + c];
        sy_p[base + 1] = sy1[s * cw + c];
      }
    }
  }

  auto y = yt.t().contiguous();  // [C, T]
  return std::make_tuple(y, new_sx, new_sy);
}

// Minimum channel count for the cross-channel SIMD path. Below this the scalar
// OpenMP-over-channels kernel is used (when channels <= cores it already saturates
// the cores, and the SIMD path's transpose does not pay off). Tunable via the
// TORCHFX_SIMD_MIN_CHANNELS env var (read once) for benchmarking / per-device tuning.
static int64_t simd_min_channels() {
  static const int64_t v = []() -> int64_t {
    const char* e = std::getenv("TORCHFX_SIMD_MIN_CHANNELS");
    if (e != nullptr && e[0] != '\0') {
      const long long parsed = std::atoll(e);
      if (parsed > 0) return static_cast<int64_t>(parsed);
    }
    return 16;
  }();
  return v;
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

  // Use the cross-channel SIMD path at high channel counts; otherwise the scalar
  // OpenMP-over-channels kernel (which already saturates the cores when C is small).
  const bool use_simd = x_exec.size(0) >= simd_min_channels();

  if (exec_dtype == torch::kFloat32) {
    return use_simd ? sos_forward_cpu_simd_impl<float>(x_exec, sos_exec, sx_exec, sy_exec)
                    : sos_forward_cpu_impl<float>(x_exec, sos_exec, sx_exec, sy_exec);
  }

  return use_simd ? sos_forward_cpu_simd_impl<double>(x_exec, sos_exec, sx_exec, sy_exec)
                  : sos_forward_cpu_impl<double>(x_exec, sos_exec, sx_exec, sy_exec);
}
