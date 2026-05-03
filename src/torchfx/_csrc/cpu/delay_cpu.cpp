#include <torch/torch.h>

// CPU delay line with feedback and wet/dry mixing.
//
// Implements: y[n] = (1 - mix) * x[n] + mix * (x[n] + decay * x[n - delay])
//           = x[n] + mix * decay * x[n - delay]   for n >= delay
//           = x[n]                                  for n < delay

// MSVC spells the restrict qualifier `__restrict` (no trailing underscores);
// GCC, Clang, and AppleClang accept `__restrict__`. Map to the right form.
#if defined(_MSC_VER)
#define TORCHFX_RESTRICT __restrict
#else
#define TORCHFX_RESTRICT __restrict__
#endif

template <typename scalar_t>
static void delay_loop(
    const scalar_t* TORCHFX_RESTRICT in_ptr,
    scalar_t* TORCHFX_RESTRICT out_ptr,
    int64_t C,
    int64_t T,
    int delay_samples,
    scalar_t coeff) {

  #pragma omp parallel for schedule(static) if(C > 1)
  for (int64_t c = 0; c < C; ++c) {
    const scalar_t* in_c = in_ptr + c * T;
    scalar_t* out_c = out_ptr + c * T;

    // Pre-delay region: output = input (no delayed signal yet)
    for (int64_t n = 0; n < delay_samples; ++n) {
      out_c[n] = in_c[n];
    }

    // Post-delay region: output = input + coeff * delayed_input
    for (int64_t n = delay_samples; n < T; ++n) {
      out_c[n] = in_c[n] + coeff * in_c[n - delay_samples];
    }
  }
}

torch::Tensor delay_line_forward_cpu(
    const torch::Tensor& x,
    int delay_samples,
    double decay,
    double mix) {

  TORCH_CHECK(!x.is_cuda(), "delay_line_forward_cpu: input must be on CPU");

  auto x_cont = x.contiguous();
  const int orig_dim = x_cont.dim();
  if (orig_dim == 1) {
    x_cont = x_cont.unsqueeze(0);
  }

  const int64_t C = x_cont.size(0);
  const int64_t T = x_cont.size(1);

  // Short-circuit: if signal is shorter than delay, return dry signal
  if (T <= delay_samples) {
    return x;
  }

  auto output = torch::empty_like(x_cont);

  if (x_cont.dtype() == torch::kFloat32) {
    delay_loop<float>(
        x_cont.data_ptr<float>(),
        output.data_ptr<float>(),
        C, T, delay_samples,
        static_cast<float>(mix * decay));
  } else {
    delay_loop<double>(
        x_cont.data_ptr<double>(),
        output.data_ptr<double>(),
        C, T, delay_samples,
        mix * decay);
  }

  if (orig_dim == 1) {
    return output.squeeze(0);
  }
  return output;
}
