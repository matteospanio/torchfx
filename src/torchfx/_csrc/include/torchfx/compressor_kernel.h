#pragma once

#include <torch/types.h>

namespace torchfx {

// Feed-forward dynamic-range compressor with a decoupled peak detector.
//
// Per channel, sequentially over time n:
//   rect = |x[n]|                              (detector=0, peak)
//   ms   = rms_coeff*ms + (1-rms_coeff)*x^2;  rect = sqrt(ms)  (detector=1, rms)
//   y1   = max(rect, release_coeff * y1)       (release max-hold)
//   yL   = attack_coeff*yL + (1-attack_coeff)*y1   (attack one-pole)
//   L    = 20*log10(max(yL, eps))
//   gain curve (threshold_db, inv_ratio, knee_db) -> Lsc
//   g    = 10^((Lsc - L + makeup_db)/20);  y[n] = g * x[n]
//
// `inv_ratio` is 1/ratio (0 for an infinite-ratio limiter, avoiding inf math).
torch::Tensor compressor_forward_cuda(
    const torch::Tensor& x,
    double threshold_db,
    double inv_ratio,
    double knee_db,
    double makeup_db,
    double attack_coeff,
    double release_coeff,
    double rms_coeff,
    int detector);

}  // namespace torchfx
