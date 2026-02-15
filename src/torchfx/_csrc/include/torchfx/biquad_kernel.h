#pragma once

#include <torch/types.h>

namespace torchfx {

// Apply a single biquad filter section using CUDA parallel scan.
// Input:  x [C, T], b [3], a [3] (a[0]=1), state_x [C, 2], state_y [C, 2]
// Output: (y [C, T], new_state_x [C, 2], new_state_y [C, 2])
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> biquad_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& b,
    const torch::Tensor& a,
    const torch::Tensor& state_x,
    const torch::Tensor& state_y);

// Apply a cascade of biquad sections (SOS format) using CUDA parallel scan.
// Input:  x [C, T], sos [K, 6], state_x [K, C, 2], state_y [K, C, 2]
// Output: (y [C, T], new_state_x [K, C, 2], new_state_y [K, C, 2])
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sos_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& sos,
    const torch::Tensor& state_x,
    const torch::Tensor& state_y);

}  // namespace torchfx
