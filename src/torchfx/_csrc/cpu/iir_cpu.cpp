#include <torch/torch.h>
#include <tuple>

// CPU implementation of biquad Direct Form 1.
// Vectorized across channels, sequential across time.
// This is significantly faster than the Python for-loop.

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> biquad_forward_cpu(
    const torch::Tensor& x,         // [C, T]
    const torch::Tensor& b,         // [3]
    const torch::Tensor& a,         // [3]
    const torch::Tensor& state_x,   // [C, 2]
    const torch::Tensor& state_y) { // [C, 2]

  auto x_f64 = x.to(torch::kFloat64).contiguous();
  auto b_acc = b.to(torch::kFloat64).contiguous();
  auto a_acc = a.to(torch::kFloat64).contiguous();
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
  const double a1 = a_acc.accessor<double, 1>()[1];
  const double a2 = a_acc.accessor<double, 1>()[2];

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

  // Process each SOS section sequentially; within each section, loop over time.
  auto section_input = x_f64;

  for (int64_t s = 0; s < K; ++s) {
    const double b0 = sos_acc[s][0];
    const double b1 = sos_acc[s][1];
    const double b2 = sos_acc[s][2];
    // sos[s][3] is a0 = 1 (normalized)
    const double a1 = sos_acc[s][4];
    const double a2 = sos_acc[s][5];

    auto out = torch::empty_like(section_input);
    auto in_ptr = section_input.accessor<double, 2>();
    auto out_ptr = out.accessor<double, 2>();
    auto sx_s = sx[s];
    auto sy_s = sy[s];
    auto sx_ptr = sx_s.accessor<double, 2>();
    auto sy_ptr = sy_s.accessor<double, 2>();

    for (int64_t c = 0; c < C; ++c) {
      double sx0 = sx_ptr[c][0];
      double sx1 = sx_ptr[c][1];
      double sy0 = sy_ptr[c][0];
      double sy1 = sy_ptr[c][1];

      for (int64_t n = 0; n < T; ++n) {
        double xn = in_ptr[c][n];
        double yn = b0 * xn + b1 * sx0 + b2 * sx1 - a1 * sy0 - a2 * sy1;
        out_ptr[c][n] = yn;

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

    section_input = out;
  }

  return std::make_tuple(section_input, sx, sy);
}
