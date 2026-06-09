#include <torch/extension.h>

#ifdef WITH_CUDA
#include "torchfx/biquad_kernel.h"
#include "torchfx/delay_kernel.h"
#include "torchfx/compressor_kernel.h"
#include "torchfx/expander_kernel.h"
#include "torchfx/limiter_kernel.h"
#include "torchfx/reverb_kernel.h"
#endif

// CPU implementation declarations
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> biquad_forward_cpu(
    const torch::Tensor& x,
    const torch::Tensor& b,
    double a1,
    double a2,
    const torch::Tensor& state_x,
    const torch::Tensor& state_y);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sos_forward_cpu(
    const torch::Tensor& x,
    const torch::Tensor& sos,
    const torch::Tensor& state_x,
    const torch::Tensor& state_y);

torch::Tensor delay_line_forward_cpu(
    const torch::Tensor& x,
    int delay_samples,
    double decay,
    double mix);

torch::Tensor compressor_forward_cpu(
    const torch::Tensor& x,
    double threshold_db,
    double inv_ratio,
    double knee_db,
    double makeup_db,
    double attack_coeff,
    double release_coeff,
    double rms_coeff,
    int detector);

torch::Tensor expander_forward_cpu(
    const torch::Tensor& x,
    double threshold_db,
    double slope,
    double knee_db,
    double floor_db,
    double attack_coeff,
    double release_coeff,
    double rms_coeff,
    int detector);

torch::Tensor limiter_forward_cpu(
    const torch::Tensor& x,
    const torch::Tensor& peak_env,
    double threshold_lin,
    double attack_coeff,
    double release_coeff);

torch::Tensor reverb_forward_cpu(
    const torch::Tensor& x,
    int fs,
    double feedback,
    double damp,
    double input_gain,
    double allpass_fb,
    double wet,
    double dry);

// Dispatch: select CUDA or CPU implementation based on tensor device.
// `threshold` is the sequential-vs-parallel-scan boundary used by the CUDA path
// (ignored on CPU, which is always sequential).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> biquad_forward(
    const torch::Tensor& x,
    const torch::Tensor& b,
    double a1,
    double a2,
    const torch::Tensor& state_x,
    const torch::Tensor& state_y,
    int threshold) {
#ifdef WITH_CUDA
  if (x.is_cuda()) {
    // Extract b coefficients as scalars to avoid GPU→CPU sync in the kernel.
    auto b_cpu = b.is_cuda() ? b.detach().cpu() : b;
    const double b0 = b_cpu[0].item<double>();
    const double b1 = b_cpu[1].item<double>();
    const double b2 = b_cpu[2].item<double>();
    return torchfx::biquad_forward_cuda(x, b0, b1, b2, a1, a2, state_x, state_y, threshold);
  }
#else
  TORCH_CHECK(!x.is_cuda(), "CUDA extension not compiled; move tensors to CPU");
#endif
  (void)threshold;  // CPU kernel is always sequential.
  return biquad_forward_cpu(x, b, a1, a2, state_x, state_y);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sos_forward(
    const torch::Tensor& x,
    const torch::Tensor& sos,
    const torch::Tensor& sos_cpu,
    const torch::Tensor& state_x,
    const torch::Tensor& state_y,
    int threshold) {
#ifdef WITH_CUDA
  if (x.is_cuda()) {
    return torchfx::sos_forward_cuda(x, sos, sos_cpu, state_x, state_y, threshold);
  }
#else
  TORCH_CHECK(!x.is_cuda(), "CUDA extension not compiled; move tensors to CPU");
#endif
  (void)threshold;  // CPU kernel is always sequential.
  return sos_forward_cpu(x, sos, state_x, state_y);
}

torch::Tensor delay_line_forward(
    const torch::Tensor& x,
    int delay_samples,
    double decay,
    double mix) {
#ifdef WITH_CUDA
  if (x.is_cuda()) {
    return torchfx::delay_line_forward_cuda(x, delay_samples, decay, mix);
  }
#else
  TORCH_CHECK(!x.is_cuda(), "CUDA extension not compiled; move tensors to CPU");
#endif
  return delay_line_forward_cpu(x, delay_samples, decay, mix);
}

// Compressor dispatch. `inv_ratio` = 1/ratio (0 for an infinite-ratio limiter);
// `detector` is 0=peak, 1=rms. Coefficients are precomputed by the Python layer
// from attack/release/rms times and fs.
torch::Tensor compressor_forward(
    const torch::Tensor& x,
    double threshold_db,
    double inv_ratio,
    double knee_db,
    double makeup_db,
    double attack_coeff,
    double release_coeff,
    double rms_coeff,
    int detector) {
#ifdef WITH_CUDA
  if (x.is_cuda()) {
    return torchfx::compressor_forward_cuda(x, threshold_db, inv_ratio, knee_db,
                                            makeup_db, attack_coeff, release_coeff,
                                            rms_coeff, detector);
  }
#else
  TORCH_CHECK(!x.is_cuda(), "CUDA extension not compiled; move tensors to CPU");
#endif
  return compressor_forward_cpu(x, threshold_db, inv_ratio, knee_db, makeup_db,
                                attack_coeff, release_coeff, rms_coeff, detector);
}

// Expander / gate dispatch. `slope` = ratio-1 (a large finite value for a gate, so the
// kernel never does inf arithmetic); `floor_db` is the deepest attenuation; `detector`
// is 0=peak, 1=rms. Coefficients are precomputed by the Python layer.
torch::Tensor expander_forward(
    const torch::Tensor& x,
    double threshold_db,
    double slope,
    double knee_db,
    double floor_db,
    double attack_coeff,
    double release_coeff,
    double rms_coeff,
    int detector) {
#ifdef WITH_CUDA
  if (x.is_cuda()) {
    return torchfx::expander_forward_cuda(x, threshold_db, slope, knee_db,
                                          floor_db, attack_coeff, release_coeff,
                                          rms_coeff, detector);
  }
#else
  TORCH_CHECK(!x.is_cuda(), "CUDA extension not compiled; move tensors to CPU");
#endif
  return expander_forward_cpu(x, threshold_db, slope, knee_db, floor_db,
                              attack_coeff, release_coeff, rms_coeff, detector);
}

// Look-ahead brick-wall limiter dispatch. `peak_env` is the precomputed look-ahead
// windowed peak of |x|; `threshold_lin` is the linear ceiling.
torch::Tensor limiter_forward(
    const torch::Tensor& x,
    const torch::Tensor& peak_env,
    double threshold_lin,
    double attack_coeff,
    double release_coeff) {
#ifdef WITH_CUDA
  if (x.is_cuda()) {
    return torchfx::limiter_forward_cuda(x, peak_env, threshold_lin, attack_coeff, release_coeff);
  }
#else
  TORCH_CHECK(!x.is_cuda(), "CUDA extension not compiled; move tensors to CPU");
#endif
  return limiter_forward_cpu(x, peak_env, threshold_lin, attack_coeff, release_coeff);
}

// Freeverb-style reverb dispatch. Comb/all-pass tunings scale with `fs`; `feedback`,
// `damp`, `wet`, `dry` come from the Python layer's room-size/damping/mix parameters.
torch::Tensor reverb_forward(
    const torch::Tensor& x,
    int fs,
    double feedback,
    double damp,
    double input_gain,
    double allpass_fb,
    double wet,
    double dry) {
#ifdef WITH_CUDA
  if (x.is_cuda()) {
    return torchfx::reverb_forward_cuda(x, fs, feedback, damp, input_gain, allpass_fb, wet, dry);
  }
#else
  TORCH_CHECK(!x.is_cuda(), "CUDA extension not compiled; move tensors to CPU");
#endif
  return reverb_forward_cpu(x, fs, feedback, damp, input_gain, allpass_fb, wet, dry);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("biquad_forward", &biquad_forward,
        "Biquad filter forward (CUDA/CPU)",
        py::arg("x"), py::arg("b"), py::arg("a1"), py::arg("a2"),
        py::arg("state_x"), py::arg("state_y"), py::arg("threshold") = 2048);
  m.def("sos_forward", &sos_forward,
        "SOS cascade forward (CUDA/CPU)",
        py::arg("x"), py::arg("sos"), py::arg("sos_cpu"),
        py::arg("state_x"), py::arg("state_y"), py::arg("threshold") = 2048);
  m.def("delay_line_forward", &delay_line_forward,
        "Delay line forward (CUDA only)",
        py::arg("x"), py::arg("delay_samples"),
        py::arg("decay"), py::arg("mix"));
  m.def("compressor_forward", &compressor_forward,
        "Compressor forward (CUDA/CPU)",
        py::arg("x"), py::arg("threshold"), py::arg("inv_ratio"), py::arg("knee"),
        py::arg("makeup_db"), py::arg("attack_coeff"), py::arg("release_coeff"),
        py::arg("rms_coeff"), py::arg("detector"));
  m.def("expander_forward", &expander_forward,
        "Expander / gate forward (CUDA/CPU)",
        py::arg("x"), py::arg("threshold"), py::arg("slope"), py::arg("knee"),
        py::arg("floor_db"), py::arg("attack_coeff"), py::arg("release_coeff"),
        py::arg("rms_coeff"), py::arg("detector"));
  m.def("limiter_forward", &limiter_forward,
        "Look-ahead brick-wall limiter forward (CUDA/CPU)",
        py::arg("x"), py::arg("peak_env"), py::arg("threshold_lin"),
        py::arg("attack_coeff"), py::arg("release_coeff"));
  m.def("reverb_forward", &reverb_forward,
        "Freeverb-style reverb forward (CUDA/CPU)",
        py::arg("x"), py::arg("fs"), py::arg("feedback"), py::arg("damp"),
        py::arg("input_gain"), py::arg("allpass_fb"), py::arg("wet"), py::arg("dry"));
}
