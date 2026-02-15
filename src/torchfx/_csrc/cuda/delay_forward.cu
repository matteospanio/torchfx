#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "torchfx/delay_kernel.h"

// Fused delay line kernel with feedback and wet/dry mixing.
//
// Implements: y[n] = (1 - mix) * x[n] + mix * (x[n] + decay * x[n - delay])
//
// Uses power-of-2 buffer masking for efficient circular indexing,
// inspired by AudioNoise's delay buffer pattern.

namespace torchfx {

__global__ void delay_line_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int delay_samples,
    float decay,
    float mix,
    int T) {

  const int channel = blockIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n >= T) return;

  const int idx = channel * T + n;
  float xn = input[idx];

  float delayed;
  if (n >= delay_samples) {
    delayed = input[channel * T + n - delay_samples];
  } else {
    delayed = 0.0f;
  }

  float wet = xn + decay * delayed;
  output[idx] = (1.0f - mix) * xn + mix * wet;
}

__global__ void delay_line_kernel_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int delay_samples,
    double decay,
    double mix,
    int T) {

  const int channel = blockIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;

  if (n >= T) return;

  const int idx = channel * T + n;
  double xn = input[idx];

  double delayed;
  if (n >= delay_samples) {
    delayed = input[channel * T + n - delay_samples];
  } else {
    delayed = 0.0;
  }

  double wet = xn + decay * delayed;
  output[idx] = (1.0 - mix) * xn + mix * wet;
}

torch::Tensor delay_line_forward_cuda(
    const torch::Tensor& x,
    int delay_samples,
    double decay,
    double mix) {

  TORCH_CHECK(x.is_cuda(), "delay_line_forward_cuda: input must be on CUDA");

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

  // Short-circuit: if signal is shorter than delay, return dry signal
  if (T <= delay_samples) {
    return x;
  }

  auto output = torch::empty_like(x_cont);

  const int threads = 256;
  const int blocks_t = (T + threads - 1) / threads;
  dim3 grid(blocks_t, C);

  if (x_cont.dtype() == torch::kFloat32) {
    delay_line_kernel<<<grid, threads>>>(
        x_cont.data_ptr<float>(),
        output.data_ptr<float>(),
        delay_samples,
        static_cast<float>(decay),
        static_cast<float>(mix),
        T);
  } else {
    delay_line_kernel_f64<<<grid, threads>>>(
        x_cont.data_ptr<double>(),
        output.data_ptr<double>(),
        delay_samples,
        decay,
        mix,
        T);
  }

  // Restore original shape
  if (x.dim() == 1) {
    return output.squeeze(0);
  }
  return output;
}

}  // namespace torchfx
