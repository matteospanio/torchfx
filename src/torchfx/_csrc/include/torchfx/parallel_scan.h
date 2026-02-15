#pragma once

#include <torch/types.h>

namespace torchfx {

// Parallel prefix scan for second-order IIR (biquad) filtering.
//
// Reformulates the biquad recurrence as a linear recurrence with 3x3 matrix
// multiplication as the associative operator, then applies a Hillis-Steele
// parallel prefix scan.
//
// The biquad: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
//
// Step 1: Precompute forcing f[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2]
// Step 2: Reformulate as s[n] = M[n] * s[n-1] where:
//         s[n] = [y[n], y[n-1], 1]^T
//         M[n] = [-a1, -a2, f[n]; 1, 0, 0; 0, 0, 1]
// Step 3: Parallel prefix scan over M[0..T-1] with matrix multiply

// Compute f[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] for all n.
// Input:  x [C, T], b [3] = {b0, b1, b2}
// Output: f [C, T]
torch::Tensor compute_forcing(const torch::Tensor& x, const torch::Tensor& b);

// Parallel biquad via prefix scan.
// Input:  f [C, T] (precomputed forcing), a1, a2 (feedback coefficients),
//         state [C, 2] = {y[-1], y[-2]} per channel
// Output: y [C, T], updated state [C, 2]
std::tuple<torch::Tensor, torch::Tensor> parallel_biquad_scan(
    const torch::Tensor& f,
    double a1,
    double a2,
    const torch::Tensor& state);

}  // namespace torchfx
