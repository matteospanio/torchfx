#include <torch/extension.h>

#ifdef WITH_CUDA
#include "torchfx/biquad_kernel.h"
#include "torchfx/delay_kernel.h"
#endif

// CPU fallback declarations
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

// Dispatch: select CUDA or CPU implementation based on tensor device.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> biquad_forward(
    const torch::Tensor& x,
    const torch::Tensor& b,
    double a1,
    double a2,
    const torch::Tensor& state_x,
    const torch::Tensor& state_y) {
#ifdef WITH_CUDA
  if (x.is_cuda()) {
    return torchfx::biquad_forward_cuda(x, b, a1, a2, state_x, state_y);
  }
#else
  TORCH_CHECK(!x.is_cuda(), "CUDA extension not compiled; move tensors to CPU");
#endif
  return biquad_forward_cpu(x, b, a1, a2, state_x, state_y);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sos_forward(
    const torch::Tensor& x,
    const torch::Tensor& sos,
    const torch::Tensor& sos_cpu,
    const torch::Tensor& state_x,
    const torch::Tensor& state_y) {
#ifdef WITH_CUDA
  if (x.is_cuda()) {
    return torchfx::sos_forward_cuda(x, sos, sos_cpu, state_x, state_y);
  }
#else
  TORCH_CHECK(!x.is_cuda(), "CUDA extension not compiled; move tensors to CPU");
#endif
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
#endif
  TORCH_CHECK(false, "delay_line_forward CPU not implemented in C++; use Python fallback");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("biquad_forward", &biquad_forward,
        "Biquad filter forward (CUDA/CPU)",
        py::arg("x"), py::arg("b"), py::arg("a1"), py::arg("a2"),
        py::arg("state_x"), py::arg("state_y"));
  m.def("sos_forward", &sos_forward,
        "SOS cascade forward (CUDA/CPU)",
        py::arg("x"), py::arg("sos"), py::arg("sos_cpu"),
        py::arg("state_x"), py::arg("state_y"));
  m.def("delay_line_forward", &delay_line_forward,
        "Delay line forward (CUDA only)",
        py::arg("x"), py::arg("delay_samples"),
        py::arg("decay"), py::arg("mix"));
}
