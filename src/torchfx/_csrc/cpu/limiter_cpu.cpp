#include <torch/torch.h>
#include <algorithm>
#include <cmath>

// CPU look-ahead brick-wall limiter (gain stage). The look-ahead windowed peak is
// precomputed; this runs the sequential gain recurrence, one channel per OpenMP thread.
// See include/torchfx/limiter_kernel.h for the per-sample math.

#if defined(_MSC_VER)
#define TORCHFX_RESTRICT __restrict
#else
#define TORCHFX_RESTRICT __restrict__
#endif

template <typename scalar_t>
static void limiter_loop(
    const scalar_t* TORCHFX_RESTRICT in_ptr,
    const scalar_t* TORCHFX_RESTRICT peak_ptr,
    scalar_t* TORCHFX_RESTRICT out_ptr,
    int64_t C,
    int64_t T,
    scalar_t threshold_lin,
    scalar_t attack_coeff,
    scalar_t release_coeff) {

  const scalar_t eps = static_cast<scalar_t>(1e-12);
  const scalar_t one = static_cast<scalar_t>(1);

  #pragma omp parallel for schedule(static) if(C > 1)
  for (int64_t c = 0; c < C; ++c) {
    const scalar_t* in_c = in_ptr + c * T;
    const scalar_t* peak_c = peak_ptr + c * T;
    scalar_t* out_c = out_ptr + c * T;

    scalar_t g = one;
    for (int64_t n = 0; n < T; ++n) {
      const scalar_t gr = std::min(one, threshold_lin / std::max(peak_c[n], eps));
      if (gr < g) {
        g = attack_coeff * g + (one - attack_coeff) * gr;
      } else {
        g = release_coeff * g + (one - release_coeff) * gr;
      }
      const scalar_t clamp = threshold_lin / std::max(std::abs(in_c[n]), eps);
      const scalar_t g_out = std::min(g, clamp);
      out_c[n] = g_out * in_c[n];
    }
  }
}

torch::Tensor limiter_forward_cpu(
    const torch::Tensor& x,
    const torch::Tensor& peak_env,
    double threshold_lin,
    double attack_coeff,
    double release_coeff) {

  TORCH_CHECK(!x.is_cuda(), "limiter_forward_cpu: input must be on CPU");
  TORCH_CHECK(x.sizes() == peak_env.sizes(), "limiter_forward_cpu: x and peak_env must match");

  auto x_cont = x.contiguous();
  auto peak_cont = peak_env.contiguous();
  const int orig_dim = x_cont.dim();
  if (orig_dim == 1) {
    x_cont = x_cont.unsqueeze(0);
    peak_cont = peak_cont.unsqueeze(0);
  }

  const int64_t C = x_cont.size(0);
  const int64_t T = x_cont.size(1);
  auto output = torch::empty_like(x_cont);

  if (x_cont.dtype() == torch::kFloat32) {
    limiter_loop<float>(
        x_cont.data_ptr<float>(), peak_cont.data_ptr<float>(), output.data_ptr<float>(),
        C, T, static_cast<float>(threshold_lin), static_cast<float>(attack_coeff),
        static_cast<float>(release_coeff));
  } else {
    limiter_loop<double>(
        x_cont.data_ptr<double>(), peak_cont.data_ptr<double>(), output.data_ptr<double>(),
        C, T, threshold_lin, attack_coeff, release_coeff);
  }

  if (orig_dim == 1) {
    return output.squeeze(0);
  }
  return output;
}
