#pragma once

#include <torch/types.h>

namespace torchfx {

// Fused delay line with feedback and wet/dry mixing.
// Inspired by AudioNoise's circular buffer pattern with power-of-2 masking.
//
// Input:  x [C, T], delay_samples (int), decay (float), mix (float)
// Output: y [C, T]
torch::Tensor delay_line_forward_cuda(
    const torch::Tensor& x,
    int delay_samples,
    double decay,
    double mix);

}  // namespace torchfx
