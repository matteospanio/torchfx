#include <torch/torch.h>
#include <omp.h>
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

  if (exec_dtype == torch::kFloat32) {
    return sos_forward_cpu_impl<float>(x_exec, sos_exec, sx_exec, sy_exec);
  }

  return sos_forward_cpu_impl<double>(x_exec, sos_exec, sx_exec, sy_exec);
}
