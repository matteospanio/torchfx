#pragma once

#include <torch/types.h>

namespace torchfx {

// Parallel prefix scan for second-order IIR (biquad) filtering.
//
// Reformulates the biquad recurrence as a linear recurrence with 3x3 matrix
// multiplication as the associative operator, then applies a Blelloch
// work-efficient parallel prefix scan.
//
// The biquad: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
//
// Step 1: Precompute forcing f[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2]
// Step 2: Reformulate as s[n] = M[n] * s[n-1] where:
//         s[n] = [y[n], y[n-1], 1]^T
//         M[n] = [-a1, -a2, f[n]; 1, 0, 0; 0, 0, 1]
// Step 3: Parallel prefix scan over M[0..T-1] with matrix multiply

// Compute f[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] for all n, with state prepend.
// Input:  x [C, T], b0/b1/b2 (scalars), state_x [C, 2] = {x[-1], x[-2]}
// Output: f [C, T]
torch::Tensor compute_forcing(
    const torch::Tensor& x,
    double b0, double b1, double b2,
    const torch::Tensor& state_x);

// In-place variant: write the forcing into a caller-provided [C, T] buffer instead
// of allocating. Used by the cascade to reuse one scratch buffer across all
// sections (no per-section allocation), which keeps the kernel sequence and its
// buffer addresses stable for CUDA graph capture.
void compute_forcing_into(
    const torch::Tensor& x,
    double b0, double b1, double b2,
    const torch::Tensor& state_x,
    torch::Tensor& f_out);

// Parallel biquad via prefix scan.
// Input:  f [C, T] (precomputed forcing), a1, a2 (feedback coefficients),
//         state [C, 2] = {y[-1], y[-2]} per channel,
//         threshold: signals with T <= threshold use the sequential kernel;
//         longer signals use the work-efficient parallel scan.
// Output: y [C, T], updated state [C, 2]
std::tuple<torch::Tensor, torch::Tensor> parallel_biquad_scan(
    const torch::Tensor& f,
    double a1,
    double a2,
    const torch::Tensor& state,
    int threshold);

// In-place variant: write the scan output into caller-provided ``y_out`` [C, T] and
// use caller-provided ``block_agg`` scratch (sized >= C*ceil(T/512)*6, ignored on the
// sequential branch). Does NOT extract the new state — the caller reads it from
// ``y_out`` (in place for the cascade). Lets the cascade reuse one set of scratch
// buffers across sections so capture sees stable buffer addresses.
void parallel_biquad_scan_into(
    const torch::Tensor& f,
    double a1,
    double a2,
    const torch::Tensor& state,
    int threshold,
    torch::Tensor& y_out,
    torch::Tensor& block_agg);

// Fused per-section SOS path: the FIR forcing is folded into the scan so each
// section is a single kernel launch (sequential branch done; parallel branch is
// the single-pass decoupled-look-back scan). Reads ``state_x`` (forcing history)
// and ``state_y`` (recurrence init); writes ``y_out``. ``f_scratch`` / ``block_agg``
// are caller-owned reused buffers (the long-signal fallback still uses them).
void fused_sos_scan_into(
    const torch::Tensor& x,
    double b0, double b1, double b2,
    double a1, double a2,
    const torch::Tensor& state_x,
    const torch::Tensor& state_y,
    int threshold,
    torch::Tensor& y_out,
    torch::Tensor& f_scratch,
    torch::Tensor& block_agg);

}  // namespace torchfx
