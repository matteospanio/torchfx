#include <torch/torch.h>
#include <algorithm>
#include <cmath>

// CPU feed-forward compressor with a decoupled peak detector. One channel per
// OpenMP thread, sequential over time (the ballistics are a nonlinear recurrence).
// Detector state (y1, yL, ms) is threaded through a [C, 3] tensor so block-wise
// streaming is state-continuous across chunks.
// See include/torchfx/compressor_kernel.h for the per-sample math.

#if defined(_MSC_VER)
#define TORCHFX_RESTRICT __restrict
#else
#define TORCHFX_RESTRICT __restrict__
#endif

template <typename scalar_t>
static void compressor_loop(
    const scalar_t* TORCHFX_RESTRICT in_ptr,
    scalar_t* TORCHFX_RESTRICT out_ptr,
    scalar_t* TORCHFX_RESTRICT state_ptr,  // [C, 3] = (y1, yL, ms), in/out
    int64_t C,
    int64_t T,
    scalar_t threshold_db,
    scalar_t inv_ratio,
    scalar_t knee_db,
    scalar_t makeup_db,
    scalar_t attack_coeff,
    scalar_t release_coeff,
    scalar_t rms_coeff,
    int detector) {

  const scalar_t eps = static_cast<scalar_t>(1e-12);
  const scalar_t one = static_cast<scalar_t>(1);
  const scalar_t two = static_cast<scalar_t>(2);
  const scalar_t twenty = static_cast<scalar_t>(20);

  #pragma omp parallel for schedule(static) if(C > 1)
  for (int64_t c = 0; c < C; ++c) {
    const scalar_t* in_c = in_ptr + c * T;
    scalar_t* out_c = out_ptr + c * T;

    scalar_t y1 = state_ptr[c * 3 + 0];
    scalar_t yL = state_ptr[c * 3 + 1];
    scalar_t ms = state_ptr[c * 3 + 2];
    for (int64_t n = 0; n < T; ++n) {
      const scalar_t xn = in_c[n];

      scalar_t rect;
      if (detector == 1) {  // RMS
        ms = rms_coeff * ms + (one - rms_coeff) * xn * xn;
        rect = std::sqrt(ms);
      } else {  // peak
        rect = std::abs(xn);
      }

      y1 = std::max(rect, release_coeff * y1);            // release max-hold
      yL = attack_coeff * yL + (one - attack_coeff) * y1;  // attack one-pole

      const scalar_t L = twenty * std::log10(std::max(yL, eps));
      const scalar_t over = L - threshold_db;
      scalar_t Lsc;
      if (knee_db > 0 && two * std::abs(over) <= knee_db) {
        const scalar_t t = over + knee_db / two;
        Lsc = L + (inv_ratio - one) * t * t / (two * knee_db);
      } else if (over > 0) {
        Lsc = threshold_db + over * inv_ratio;
      } else {
        Lsc = L;
      }

      const scalar_t gdb = (Lsc - L) + makeup_db;
      const scalar_t g = std::pow(static_cast<scalar_t>(10), gdb / twenty);
      out_c[n] = g * xn;
    }
    state_ptr[c * 3 + 0] = y1;
    state_ptr[c * 3 + 1] = yL;
    state_ptr[c * 3 + 2] = ms;
  }
}

std::tuple<torch::Tensor, torch::Tensor> compressor_forward_cpu(
    const torch::Tensor& x,
    double threshold_db,
    double inv_ratio,
    double knee_db,
    double makeup_db,
    double attack_coeff,
    double release_coeff,
    double rms_coeff,
    int detector,
    const torch::Tensor& state) {

  TORCH_CHECK(!x.is_cuda(), "compressor_forward_cpu: input must be on CPU");

  auto x_cont = x.contiguous();
  const int orig_dim = x_cont.dim();
  if (orig_dim == 1) {
    x_cont = x_cont.unsqueeze(0);
  }

  const int64_t C = x_cont.size(0);
  const int64_t T = x_cont.size(1);
  auto output = torch::empty_like(x_cont);

  // Detector state (y1, yL, ms) per channel: fresh zeros when not supplied,
  // otherwise updated in place for chunk-to-chunk streaming continuity.
  torch::Tensor st = state;
  if (!st.defined()) {
    st = torch::zeros({C, 3}, x_cont.options());
  } else {
    TORCH_CHECK(st.size(0) == C && st.size(1) == 3 && st.dtype() == x_cont.dtype(),
                "compressor_forward_cpu: state must be [C, 3] with the input dtype");
    st = st.contiguous();
  }

  if (x_cont.dtype() == torch::kFloat32) {
    compressor_loop<float>(
        x_cont.data_ptr<float>(), output.data_ptr<float>(), st.data_ptr<float>(), C, T,
        static_cast<float>(threshold_db), static_cast<float>(inv_ratio),
        static_cast<float>(knee_db), static_cast<float>(makeup_db),
        static_cast<float>(attack_coeff), static_cast<float>(release_coeff),
        static_cast<float>(rms_coeff), detector);
  } else {
    compressor_loop<double>(
        x_cont.data_ptr<double>(), output.data_ptr<double>(), st.data_ptr<double>(), C, T,
        threshold_db, inv_ratio, knee_db, makeup_db,
        attack_coeff, release_coeff, rms_coeff, detector);
  }

  if (orig_dim == 1) {
    return std::make_tuple(output.squeeze(0), st);
  }
  return std::make_tuple(output, st);
}
