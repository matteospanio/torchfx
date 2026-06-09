#pragma once

#include <torch/types.h>

namespace torchfx {

// Look-ahead brick-wall peak limiter (gain stage).
//
// The look-ahead windowed peak `peak_env[n] = max(|x[n .. n+L]|)` is precomputed by the
// Python layer (a vectorised forward max-pool), so this kernel only runs the sequential
// gain recurrence — one channel per thread, over time n:
//   gr    = min(1, threshold_lin / max(peak_env[n], eps))   (windowed target; look-ahead)
//   g     = gr < g ? a_A*g + (1-a_A)*gr                       (attack: smooth ramp down)
//                  : a_R*g + (1-a_R)*gr                        (release: smooth ramp up)
//   g_out = min(g, threshold_lin / max(|x[n]|, eps))          (per-sample brick-wall clamp)
//   y[n]  = g_out * x[n]
//
// The windowed target makes `g` start ramping down *before* a peak arrives (no transient
// overshoot, no click), while the per-sample clamp guarantees |y[n]| <= threshold_lin — a
// true brick wall — even though `g` itself is smoothed.
torch::Tensor limiter_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& peak_env,
    double threshold_lin,
    double attack_coeff,
    double release_coeff);

}  // namespace torchfx
