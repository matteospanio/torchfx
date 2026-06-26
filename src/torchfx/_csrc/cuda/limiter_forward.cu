#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "torchfx/limiter_kernel.h"

// Look-ahead brick-wall limiter (gain stage). The look-ahead windowed peak is precomputed;
// this runs the sequential gain recurrence, one thread per channel. Templated on scalar_t
// so a float32 input runs natively in FP32. See limiter_kernel.h for the math.

namespace torchfx {

template <typename scalar_t>
__global__ void limiter_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ peak_env,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ state,  // [C, 1] = (g), in/out
    scalar_t threshold_lin,
    scalar_t attack_coeff,
    scalar_t release_coeff,
    int C,
    int T) {

  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= C) return;

  const scalar_t* in_c = input + static_cast<int64_t>(channel) * T;
  const scalar_t* peak_c = peak_env + static_cast<int64_t>(channel) * T;
  scalar_t* out_c = output + static_cast<int64_t>(channel) * T;

  const scalar_t eps = static_cast<scalar_t>(1e-12);
  const scalar_t one = static_cast<scalar_t>(1);

  scalar_t g = state[channel];
  for (int n = 0; n < T; ++n) {
    const scalar_t gr = fmin(one, threshold_lin / fmax(peak_c[n], eps));
    if (gr < g) {
      g = attack_coeff * g + (one - attack_coeff) * gr;
    } else {
      g = release_coeff * g + (one - release_coeff) * gr;
    }
    const scalar_t clamp = threshold_lin / fmax(fabs(in_c[n]), eps);
    const scalar_t g_out = fmin(g, clamp);
    out_c[n] = g_out * in_c[n];
  }
  state[channel] = g;
}

std::tuple<torch::Tensor, torch::Tensor> limiter_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& peak_env,
    double threshold_lin,
    double attack_coeff,
    double release_coeff,
    const torch::Tensor& state) {

  TORCH_CHECK(x.is_cuda(), "limiter_forward_cuda: input must be on CUDA");
  TORCH_CHECK(x.sizes() == peak_env.sizes(), "limiter_forward_cuda: x and peak_env must match");

  auto x_cont = x.contiguous();
  auto peak_cont = peak_env.contiguous();
  int64_t C, T;
  if (x_cont.dim() == 1) {
    C = 1;
    T = x_cont.size(0);
    x_cont = x_cont.unsqueeze(0);
    peak_cont = peak_cont.unsqueeze(0);
  } else {
    C = x_cont.size(0);
    T = x_cont.size(1);
  }

  auto output = torch::empty_like(x_cont);

  // Gain state per channel: fresh unity gain when not supplied, otherwise
  // updated in place for chunk-to-chunk streaming continuity.
  torch::Tensor st = state;
  if (!st.defined()) {
    st = torch::ones({C, 1}, x_cont.options());
  } else {
    TORCH_CHECK(st.size(0) == C && st.size(1) == 1 && st.dtype() == x_cont.dtype() && st.is_cuda(),
                "limiter_forward_cuda: state must be CUDA [C, 1] with the input dtype");
    st = st.contiguous();
  }

  const int threads = 128;
  const int blocks = (static_cast<int>(C) + threads - 1) / threads;
  const auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x_cont.scalar_type(), "limiter_forward_cuda", [&] {
    limiter_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        x_cont.data_ptr<scalar_t>(), peak_cont.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
        st.data_ptr<scalar_t>(),
        static_cast<scalar_t>(threshold_lin), static_cast<scalar_t>(attack_coeff),
        static_cast<scalar_t>(release_coeff), static_cast<int>(C), static_cast<int>(T));
  });

  if (x.dim() == 1) {
    return std::make_tuple(output.squeeze(0), st);
  }
  return std::make_tuple(output, st);
}

}  // namespace torchfx
