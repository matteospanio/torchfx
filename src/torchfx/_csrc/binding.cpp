#include <torch/extension.h>
#include "torchfx/biquad_kernel.h"
#include "torchfx/delay_kernel.h"

// CPU fallback declarations
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> biquad_forward_cpu(
    const torch::Tensor& x,
    const torch::Tensor& b,
    const torch::Tensor& a,
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
    const torch::Tensor& a,
    const torch::Tensor& state_x,
    const torch::Tensor& state_y) {
  if (x.is_cuda()) {
    return torchfx::biquad_forward_cuda(x, b, a, state_x, state_y);
  }
  return biquad_forward_cpu(x, b, a, state_x, state_y);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sos_forward(
    const torch::Tensor& x,
    const torch::Tensor& sos,
    const torch::Tensor& state_x,
    const torch::Tensor& state_y) {
  if (x.is_cuda()) {
    return torchfx::sos_forward_cuda(x, sos, state_x, state_y);
  }
  return sos_forward_cpu(x, sos, state_x, state_y);
}

torch::Tensor delay_line_forward(
    const torch::Tensor& x,
    int delay_samples,
    double decay,
    double mix) {
  if (x.is_cuda()) {
    return torchfx::delay_line_forward_cuda(x, delay_samples, decay, mix);
  }
  // CPU fallback: use PyTorch ops (handled in Python)
  TORCH_CHECK(false, "delay_line_forward CPU not implemented in C++; use Python fallback");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("biquad_forward", &biquad_forward,
        "Biquad filter forward (CUDA/CPU)",
        py::arg("x"), py::arg("b"), py::arg("a"),
        py::arg("state_x"), py::arg("state_y"));
  m.def("sos_forward", &sos_forward,
        "SOS cascade forward (CUDA/CPU)",
        py::arg("x"), py::arg("sos"),
        py::arg("state_x"), py::arg("state_y"));
  m.def("delay_line_forward", &delay_line_forward,
        "Delay line forward (CUDA only)",
        py::arg("x"), py::arg("delay_samples"),
        py::arg("decay"), py::arg("mix"));
}
