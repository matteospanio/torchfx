#include <torch/torch.h>
#include "torchfx/parallel_scan.h"
#include "torchfx/biquad_kernel.h"

namespace torchfx {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> biquad_forward_cuda(
    const torch::Tensor& x,          // [C, T]
    double b0, double b1, double b2,  // numerator coefficients
    double a1,                        // denominator coefficient a1
    double a2,                        // denominator coefficient a2
    const torch::Tensor& state_x,    // [C, 2]
    const torch::Tensor& state_y) {  // [C, 2]

  TORCH_CHECK(x.is_cuda(), "biquad_forward_cuda: input must be on CUDA");
  TORCH_CHECK(x.dim() == 2, "biquad_forward_cuda: input must be [C, T]");

  auto x_f64 = x.contiguous();
  auto sx = state_x.contiguous();
  auto sy = state_y.contiguous();

  auto C = x_f64.size(0);
  auto T = x_f64.size(1);

  // Step 1: Compute forcing function with fused state prepend — single kernel.
  auto f = compute_forcing(x_f64, b0, b1, b2, sx);  // [C, T]

  // Step 2: Parallel scan to solve y[n] = f[n] - a1*y[n-1] - a2*y[n-2]
  auto [y, new_state_y] = parallel_biquad_scan(f, a1, a2, sy);

  // Step 3: Update state_x from the last 2 input samples.
  // Use narrow + flip for minimal kernel launches.
  torch::Tensor new_state_x;
  if (T >= 2) {
    // x[:, -2:] reversed to get [x[-1], x[-2]]
    new_state_x = x_f64.narrow(1, T - 2, 2).flip(1).contiguous();
  } else if (T == 1) {
    new_state_x = torch::cat({
        x_f64.narrow(1, 0, 1),
        sx.narrow(1, 0, 1)
    }, 1);
  } else {
    new_state_x = sx;
  }

  return std::make_tuple(y, new_state_x, new_state_y);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sos_forward_cuda(
    const torch::Tensor& x,          // [C, T]
    const torch::Tensor& sos,        // [K, 6] (device)
    const torch::Tensor& sos_cpu_in, // [K, 6] (CPU) — pre-supplied to avoid GPU sync
    const torch::Tensor& state_x,    // [K, C, 2]
    const torch::Tensor& state_y) {  // [K, C, 2]

  TORCH_CHECK(x.is_cuda(), "sos_forward_cuda: input must be on CUDA");
  TORCH_CHECK(x.dim() == 2, "sos_forward_cuda: input must be [C, T]");
  TORCH_CHECK(sos.dim() == 2 && sos.size(1) == 6, "sos_forward_cuda: sos must be [K, 6]");

  auto sos_f64 = sos.contiguous();
  auto new_sx = state_x.clone();
  auto new_sy = state_y.clone();

  const int64_t K = sos_f64.size(0);

  // Use the pre-supplied CPU copy — no GPU→CPU sync needed.
  auto sos_cpu = sos_cpu_in.contiguous();

  auto section_input = x;

  // Process each SOS section sequentially, using parallel scan within each.
  for (int64_t s = 0; s < K; ++s) {
    // Extract all coefficients from CPU copy — no GPU sync needed.
    const double b0 = sos_cpu[s][0].item<double>();
    const double b1 = sos_cpu[s][1].item<double>();
    const double b2 = sos_cpu[s][2].item<double>();
    const double a1 = sos_cpu[s][4].item<double>();
    const double a2 = sos_cpu[s][5].item<double>();

    auto sx_s = new_sx[s];  // [C, 2]
    auto sy_s = new_sy[s];  // [C, 2]

    auto [y_s, nsx_s, nsy_s] = biquad_forward_cuda(
        section_input, b0, b1, b2, a1, a2, sx_s, sy_s);

    new_sx[s] = nsx_s;
    new_sy[s] = nsy_s;
    section_input = y_s;
  }

  return std::make_tuple(section_input, new_sx, new_sy);
}

}  // namespace torchfx
