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
    const torch::Tensor& state_y,    // [C, 2]
    int threshold) {                 // sequential-vs-parallel-scan boundary

  TORCH_CHECK(x.is_cuda(), "biquad_forward_cuda: input must be on CUDA");
  TORCH_CHECK(x.dim() == 2, "biquad_forward_cuda: input must be [C, T]");

  // Preserve the input dtype (float32 or float64); the templated kernels run
  // natively in that precision.
  auto x_c = x.contiguous();
  auto sx = state_x.contiguous();
  auto sy = state_y.contiguous();

  auto C = x_c.size(0);
  auto T = x_c.size(1);

  // Step 1: Compute forcing function with fused state prepend — single kernel.
  auto f = compute_forcing(x_c, b0, b1, b2, sx);  // [C, T]

  // Step 2: Parallel scan to solve y[n] = f[n] - a1*y[n-1] - a2*y[n-2]
  auto [y, new_state_y] = parallel_biquad_scan(f, a1, a2, sy, threshold);

  // Step 3: Update state_x from the last 2 input samples.
  // Use narrow + flip for minimal kernel launches.
  torch::Tensor new_state_x;
  if (T >= 2) {
    // x[:, -2:] reversed to get [x[-1], x[-2]]
    new_state_x = x_c.narrow(1, T - 2, 2).flip(1).contiguous();
  } else if (T == 1) {
    new_state_x = torch::cat({
        x_c.narrow(1, 0, 1),
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
    const torch::Tensor& state_y,    // [K, C, 2]
    int threshold) {                 // per-section dispatch boundary

  TORCH_CHECK(x.is_cuda(), "sos_forward_cuda: input must be on CUDA");
  TORCH_CHECK(x.dim() == 2, "sos_forward_cuda: input must be [C, T]");
  TORCH_CHECK(sos.dim() == 2 && sos.size(1) == 6, "sos_forward_cuda: sos must be [K, 6]");

  auto sos_f64 = sos.contiguous();
  // Update state in place (no clone). The caller owns these buffers and rebinds
  // them to the return value, so mutating them is safe — and in-place update is
  // what lets a captured CUDA graph carry streaming state across replays (the
  // state buffers keep a stable address). new_sx/new_sy alias the inputs.
  auto new_sx = state_x;
  auto new_sy = state_y;

  const int64_t K = sos_f64.size(0);
  auto x_c = x.contiguous();
  const int64_t C = x_c.size(0);
  const int64_t T = x_c.size(1);

  // Use the pre-supplied CPU copy — no GPU→CPU sync needed.
  auto sos_cpu = sos_cpu_in.contiguous();

  // Persistent scratch reused across all sections (C3): one forcing buffer, two
  // ping-pong output buffers, and one block-aggregate scratch — allocated once, not
  // per section. This is what keeps the per-forward kernel sequence and its buffer
  // addresses stable so a captured CUDA graph replays correctly (per-section
  // allocations otherwise alias in the capture pool and corrupt the replay).
  auto opts = x_c.options();
  auto f = torch::empty({C, T}, opts);
  auto y_a = torch::empty({C, T}, opts);
  auto y_b = torch::empty({C, T}, opts);
  const int num_blocks = (static_cast<int>(T) + 512 - 1) / 512;  // BLOCK_SIZE = 512
  auto block_agg = torch::empty({C * num_blocks * 6}, opts);

  torch::Tensor section_input = x_c;

  // Process each SOS section sequentially, reusing the scratch buffers.
  for (int64_t s = 0; s < K; ++s) {
    // Extract all coefficients from CPU copy — no GPU sync needed.
    const double b0 = sos_cpu[s][0].item<double>();
    const double b1 = sos_cpu[s][1].item<double>();
    const double b2 = sos_cpu[s][2].item<double>();
    const double a1 = sos_cpu[s][4].item<double>();
    const double a2 = sos_cpu[s][5].item<double>();

    torch::Tensor& y_out = (s % 2 == 0) ? y_a : y_b;

    // Forcing + scan into the shared scratch (reads the current x/y state of this
    // section; both are updated in place below, after they are read).
    compute_forcing_into(section_input, b0, b1, b2, new_sx[s], f);
    auto nsy_s = parallel_biquad_scan_into(f, a1, a2, new_sy[s], threshold, y_out, block_agg);

    // New x-state = last two input samples reversed -> {x[-1], x[-2]}.
    torch::Tensor nsx_s;
    if (T >= 2) {
      nsx_s = section_input.narrow(1, T - 2, 2).flip(1).contiguous();
    } else if (T == 1) {
      nsx_s = torch::cat({section_input.narrow(1, 0, 1), new_sx[s].narrow(1, 0, 1)}, 1);
    } else {
      nsx_s = new_sx[s].clone();
    }

    new_sx[s] = nsx_s;  // in-place state update
    new_sy[s] = nsy_s;
    section_input = y_out;
  }

  return std::make_tuple(section_input, new_sx, new_sy);
}

}  // namespace torchfx
