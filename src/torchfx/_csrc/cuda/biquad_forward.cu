#include <torch/torch.h>
#include "torchfx/parallel_scan.h"
#include "torchfx/biquad_kernel.h"

namespace torchfx {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> biquad_forward_cuda(
    const torch::Tensor& x,          // [C, T]
    const torch::Tensor& b,          // [3] = {b0, b1, b2}
    const torch::Tensor& a,          // [3] = {1, a1, a2}
    const torch::Tensor& state_x,    // [C, 2]
    const torch::Tensor& state_y) {  // [C, 2]

  TORCH_CHECK(x.is_cuda(), "biquad_forward_cuda: input must be on CUDA");
  TORCH_CHECK(x.dim() == 2, "biquad_forward_cuda: input must be [C, T]");

  // Caller (_ops.py) already provides float64 tensors; just ensure contiguity.
  auto x_f64 = x.contiguous();
  auto b_f64 = b.contiguous();
  auto a_f64 = a.contiguous();
  auto sy = state_y.contiguous();

  // Extract coefficients on CPU to avoid implicit cudaDeviceSynchronize.
  // a is a tiny 3-element tensor, so the D2H copy is negligible.
  auto a_cpu = a_f64.cpu();
  const double a1 = a_cpu.data_ptr<double>()[1];
  const double a2 = a_cpu.data_ptr<double>()[2];

  // Step 1: Compute forcing function f[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2]
  auto sx = state_x.contiguous();
  auto C = x_f64.size(0);
  auto T = x_f64.size(1);

  // Prepend state_x[:, ::-1] (x[-2], x[-1]) to x for correct boundary
  auto x_prepended = torch::cat({sx.flip(1), x_f64}, /*dim=*/1);  // [C, T+2]

  // Compute forcing on prepended signal
  auto f_full = compute_forcing(x_prepended, b_f64);  // [C, T+2]

  // The first 2 samples of f_full correspond to n=-2 and n=-1 (not needed).
  // Actually, compute_forcing pads with zeros on the left for the conv,
  // so f_full[n] = b0*x_prepended[n] + b1*x_prepended[n-1] + b2*x_prepended[n-2].
  // At n=0: f[0] = b0*x[-2] + b1*pad + b2*pad -- not right.
  //
  // Better approach: compute_forcing already handles left padding with zeros.
  // We prepended [x[-2], x[-1]] so:
  //   f_full[0] = b0*x[-2] + b1*0 + b2*0   (from padding)
  //   f_full[1] = b0*x[-1] + b1*x[-2] + b2*0
  //   f_full[2] = b0*x[0]  + b1*x[-1] + b2*x[-2]  -- this is what we want for n=0
  // So we take f_full[:, 2:]
  auto f = f_full.slice(/*dim=*/1, /*start=*/2);  // [C, T]

  // Step 2: Parallel scan to solve y[n] = f[n] - a1*y[n-1] - a2*y[n-2]
  // Initial state for y is state_y = [y[-1], y[-2]]
  auto [y, new_state_y] = parallel_biquad_scan(f, a1, a2, sy);

  // Step 3: Update state_x from the last 2 input samples
  auto new_state_x = torch::empty_like(sx);
  if (T >= 2) {
    new_state_x.index_put_({torch::indexing::Slice(), 0}, x_f64.index({torch::indexing::Slice(), -1}));
    new_state_x.index_put_({torch::indexing::Slice(), 1}, x_f64.index({torch::indexing::Slice(), -2}));
  } else if (T == 1) {
    new_state_x.index_put_({torch::indexing::Slice(), 0}, x_f64.index({torch::indexing::Slice(), 0}));
    new_state_x.index_put_({torch::indexing::Slice(), 1}, sx.index({torch::indexing::Slice(), 0}));
  } else {
    new_state_x = sx;
  }

  return std::make_tuple(y, new_state_x, new_state_y);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sos_forward_cuda(
    const torch::Tensor& x,          // [C, T]
    const torch::Tensor& sos,        // [K, 6]
    const torch::Tensor& state_x,    // [K, C, 2]
    const torch::Tensor& state_y) {  // [K, C, 2]

  TORCH_CHECK(x.is_cuda(), "sos_forward_cuda: input must be on CUDA");
  TORCH_CHECK(x.dim() == 2, "sos_forward_cuda: input must be [C, T]");
  TORCH_CHECK(sos.dim() == 2 && sos.size(1) == 6, "sos_forward_cuda: sos must be [K, 6]");

  // Caller (_ops.py) already provides float64 tensors on the correct device.
  auto sos_f64 = sos.contiguous();
  auto new_sx = state_x.clone();
  auto new_sy = state_y.clone();

  const int64_t K = sos_f64.size(0);

  // Copy the small SOS matrix to CPU once to avoid per-section .item() GPU syncs.
  auto sos_cpu = sos_f64.cpu();

  auto section_input = x;

  // Process each SOS section sequentially, using parallel scan within each.
  for (int64_t s = 0; s < K; ++s) {
    auto b = sos_f64.index({s, torch::indexing::Slice(0, 3)});
    // Build 'a' tensor from CPU copy -- .item() on CPU tensors is free.
    auto a = torch::tensor({1.0, sos_cpu[s][4].item<double>(), sos_cpu[s][5].item<double>()},
                           torch::dtype(torch::kFloat64).device(x.device()));

    auto sx_s = new_sx[s];  // [C, 2]
    auto sy_s = new_sy[s];  // [C, 2]

    auto [y_s, nsx_s, nsy_s] = biquad_forward_cuda(
        section_input, b, a, sx_s, sy_s);

    new_sx[s] = nsx_s;
    new_sy[s] = nsy_s;
    section_input = y_s;
  }

  return std::make_tuple(section_input, new_sx, new_sy);
}

}  // namespace torchfx
