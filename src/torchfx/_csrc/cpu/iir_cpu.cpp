#include <torch/torch.h>
#include <omp.h>
#include <tuple>
#include <vector>

// CPU implementation of biquad Direct Form 1.
// Vectorized across channels, sequential across time.
// This is significantly faster than the Python for-loop.

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> biquad_forward_cpu(
    const torch::Tensor& x,         // [C, T]
    const torch::Tensor& b,         // [3]
    double a1,
    double a2,
    const torch::Tensor& state_x,   // [C, 2]
    const torch::Tensor& state_y) { // [C, 2]

  auto x_f64 = x.to(torch::kFloat64).contiguous();
  auto b_acc = b.to(torch::kFloat64).contiguous();
  auto sx = state_x.to(torch::kFloat64).clone();
  auto sy = state_y.to(torch::kFloat64).clone();

  const int64_t C = x_f64.size(0);
  const int64_t T = x_f64.size(1);

  auto y = torch::empty_like(x_f64);

  auto x_ptr = x_f64.accessor<double, 2>();
  auto y_ptr = y.accessor<double, 2>();
  auto sx_ptr = sx.accessor<double, 2>();
  auto sy_ptr = sy.accessor<double, 2>();

  const double b0 = b_acc.accessor<double, 1>()[0];
  const double b1 = b_acc.accessor<double, 1>()[1];
  const double b2 = b_acc.accessor<double, 1>()[2];

  #pragma omp parallel for schedule(static) if(C > 1)
  for (int64_t c = 0; c < C; ++c) {
    double sx0 = sx_ptr[c][0];  // x[n-1]
    double sx1 = sx_ptr[c][1];  // x[n-2]
    double sy0 = sy_ptr[c][0];  // y[n-1]
    double sy1 = sy_ptr[c][1];  // y[n-2]

    for (int64_t n = 0; n < T; ++n) {
      double xn = x_ptr[c][n];
      double yn = b0 * xn + b1 * sx0 + b2 * sx1 - a1 * sy0 - a2 * sy1;
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sos_forward_cpu(
    const torch::Tensor& x,         // [C, T]
    const torch::Tensor& sos,       // [K, 6]
    const torch::Tensor& state_x,   // [K, C, 2]
    const torch::Tensor& state_y) { // [K, C, 2]

  auto x_f64 = x.to(torch::kFloat64).contiguous();
  auto sos_f64 = sos.to(torch::kFloat64).contiguous();
  auto sx = state_x.to(torch::kFloat64).clone();
  auto sy = state_y.to(torch::kFloat64).clone();

  const int64_t K = sos_f64.size(0);
  const int64_t C = x_f64.size(0);
  const int64_t T = x_f64.size(1);

  auto sos_acc = sos_f64.accessor<double, 2>();

  // Pre-extract coefficients for all sections.
  struct SosCoeffs { double b0, b1, b2, a1, a2; };
  std::vector<SosCoeffs> coeffs(K);
  for (int64_t s = 0; s < K; ++s) {
    coeffs[s] = {sos_acc[s][0], sos_acc[s][1], sos_acc[s][2],
                 sos_acc[s][4], sos_acc[s][5]};  // sos[s][3] is a0 = 1
  }

  // Single output buffer — no intermediate allocations.
  auto y = torch::empty_like(x_f64);
  auto x_ptr = x_f64.accessor<double, 2>();
  auto y_ptr = y.accessor<double, 2>();

  // Accessor helpers for state tensors (per-section, per-channel).
  auto sx_acc = sx.accessor<double, 3>();  // [K, C, 2]
  auto sy_acc = sy.accessor<double, 3>();  // [K, C, 2]

  // Fused loop: for each channel, process all K sections per sample.
  // This keeps all section states in registers/L1 and eliminates K-1
  // intermediate tensor allocations.
  //
  // Use stack arrays for K <= 16 (covers up to order-32 Butterworth),
  // falling back to heap allocation for higher orders.
  static constexpr int64_t STACK_MAX = 16;

  #pragma omp parallel for schedule(static) if(C > 1)
  for (int64_t c = 0; c < C; ++c) {
    // Load per-section state into local arrays.
    double stack_sx0[STACK_MAX], stack_sx1[STACK_MAX],
           stack_sy0[STACK_MAX], stack_sy1[STACK_MAX];

    // Heap fallback for very high order filters.
    std::vector<double> heap_sx0, heap_sx1, heap_sy0, heap_sy1;
    double *sec_sx0, *sec_sx1, *sec_sy0, *sec_sy1;
    if (K <= STACK_MAX) {
      sec_sx0 = stack_sx0; sec_sx1 = stack_sx1;
      sec_sy0 = stack_sy0; sec_sy1 = stack_sy1;
    } else {
      heap_sx0.resize(K); heap_sx1.resize(K);
      heap_sy0.resize(K); heap_sy1.resize(K);
      sec_sx0 = heap_sx0.data(); sec_sx1 = heap_sx1.data();
      sec_sy0 = heap_sy0.data(); sec_sy1 = heap_sy1.data();
    }

    for (int64_t s = 0; s < K; ++s) {
      sec_sx0[s] = sx_acc[s][c][0];
      sec_sx1[s] = sx_acc[s][c][1];
      sec_sy0[s] = sy_acc[s][c][0];
      sec_sy1[s] = sy_acc[s][c][1];
    }

    for (int64_t n = 0; n < T; ++n) {
      double val = x_ptr[c][n];

      for (int64_t s = 0; s < K; ++s) {
        const auto& co = coeffs[s];
        double yn = co.b0 * val + co.b1 * sec_sx0[s] + co.b2 * sec_sx1[s]
                  - co.a1 * sec_sy0[s] - co.a2 * sec_sy1[s];
        sec_sx1[s] = sec_sx0[s];
        sec_sx0[s] = val;
        sec_sy1[s] = sec_sy0[s];
        sec_sy0[s] = yn;
        val = yn;
      }

      y_ptr[c][n] = val;
    }

    // Write back state.
    for (int64_t s = 0; s < K; ++s) {
      sx_acc[s][c][0] = sec_sx0[s];
      sx_acc[s][c][1] = sec_sx1[s];
      sy_acc[s][c][0] = sec_sy0[s];
      sy_acc[s][c][1] = sec_sy1[s];
    }
  }

  return std::make_tuple(y, sx, sy);
}
