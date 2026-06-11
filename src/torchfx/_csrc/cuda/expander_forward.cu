#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "torchfx/expander_kernel.h"

// Downward expander / noise gate with a decoupled peak detector. One thread per
// channel, each walking its own time series sequentially (the ballistics are a
// nonlinear recurrence — same topology as the sequential biquad / compressor).
// Templated on scalar_t so a float32 input runs natively in FP32. See
// expander_kernel.h for the math.

namespace torchfx {

template <typename scalar_t>
__global__ void expander_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ state,  // [C, 3] = (y1, yL, ms), in/out
    scalar_t threshold_db,
    scalar_t slope,
    scalar_t knee_db,
    scalar_t floor_db,
    scalar_t attack_coeff,
    scalar_t release_coeff,
    scalar_t rms_coeff,
    int detector,
    int C,
    int T) {

  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= C) return;

  const scalar_t* in_c = input + static_cast<int64_t>(channel) * T;
  scalar_t* out_c = output + static_cast<int64_t>(channel) * T;

  const scalar_t eps = static_cast<scalar_t>(1e-12);
  const scalar_t one = static_cast<scalar_t>(1);
  const scalar_t two = static_cast<scalar_t>(2);
  const scalar_t twenty = static_cast<scalar_t>(20);

  scalar_t y1 = state[channel * 3 + 0];
  scalar_t yL = state[channel * 3 + 1];
  scalar_t ms = state[channel * 3 + 2];
  for (int n = 0; n < T; ++n) {
    const scalar_t xn = in_c[n];

    scalar_t rect;
    if (detector == 1) {  // RMS
      ms = rms_coeff * ms + (one - rms_coeff) * xn * xn;
      rect = sqrt(ms);
    } else {  // peak
      rect = fabs(xn);
    }

    y1 = fmax(rect, release_coeff * y1);                 // release max-hold
    yL = attack_coeff * yL + (one - attack_coeff) * y1;  // attack one-pole

    const scalar_t L = twenty * log10(fmax(yL, eps));
    const scalar_t over = L - threshold_db;
    scalar_t gdb;
    if (knee_db > scalar_t(0) && two * fabs(over) <= knee_db) {
      const scalar_t t = over - knee_db / two;
      gdb = -slope * t * t / (two * knee_db);
    } else if (over < scalar_t(0)) {
      gdb = slope * over;
    } else {
      gdb = scalar_t(0);
    }
    gdb = fmax(gdb, floor_db);

    const scalar_t g = pow(static_cast<scalar_t>(10), gdb / twenty);
    out_c[n] = g * xn;
  }
  state[channel * 3 + 0] = y1;
  state[channel * 3 + 1] = yL;
  state[channel * 3 + 2] = ms;
}

std::tuple<torch::Tensor, torch::Tensor> expander_forward_cuda(
    const torch::Tensor& x,
    double threshold_db,
    double slope,
    double knee_db,
    double floor_db,
    double attack_coeff,
    double release_coeff,
    double rms_coeff,
    int detector,
    const torch::Tensor& state) {

  TORCH_CHECK(x.is_cuda(), "expander_forward_cuda: input must be on CUDA");

  auto x_cont = x.contiguous();
  int64_t C, T;
  if (x_cont.dim() == 1) {
    C = 1;
    T = x_cont.size(0);
    x_cont = x_cont.unsqueeze(0);
  } else {
    C = x_cont.size(0);
    T = x_cont.size(1);
  }

  auto output = torch::empty_like(x_cont);

  // Detector state (y1, yL, ms) per channel: fresh zeros when not supplied,
  // otherwise updated in place for chunk-to-chunk streaming continuity.
  torch::Tensor st = state;
  if (!st.defined()) {
    st = torch::zeros({C, 3}, x_cont.options());
  } else {
    TORCH_CHECK(st.size(0) == C && st.size(1) == 3 && st.dtype() == x_cont.dtype() && st.is_cuda(),
                "expander_forward_cuda: state must be CUDA [C, 3] with the input dtype");
    st = st.contiguous();
  }

  const int threads = 128;
  const int blocks = (static_cast<int>(C) + threads - 1) / threads;
  const auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x_cont.scalar_type(), "expander_forward_cuda", [&] {
    expander_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        x_cont.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), st.data_ptr<scalar_t>(),
        static_cast<scalar_t>(threshold_db), static_cast<scalar_t>(slope),
        static_cast<scalar_t>(knee_db), static_cast<scalar_t>(floor_db),
        static_cast<scalar_t>(attack_coeff), static_cast<scalar_t>(release_coeff),
        static_cast<scalar_t>(rms_coeff), detector,
        static_cast<int>(C), static_cast<int>(T));
  });

  if (x.dim() == 1) {
    return {output.squeeze(0), st};
  }
  return {output, st};
}

}  // namespace torchfx
