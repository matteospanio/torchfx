#pragma once

#include <torch/types.h>
#include <tuple>

namespace torchfx {

// Downward expander / noise gate with a decoupled peak detector — the mirror image
// of the compressor: gain is reduced *below* the threshold instead of above it.
//
// Per channel, sequentially over time n (detector identical to the compressor):
//   rect = |x[n]|                              (detector=0, peak)
//   ms   = rms_coeff*ms + (1-rms_coeff)*x^2;  rect = sqrt(ms)  (detector=1, rms)
//   y1   = max(rect, release_coeff * y1)       (release max-hold)
//   yL   = attack_coeff*yL + (1-attack_coeff)*y1   (attack one-pole)
//   L    = 20*log10(max(yL, eps))
//   over = L - threshold_db
//   downward-expansion static curve (slope = ratio-1 >= 0, soft knee of width knee_db):
//     |2*over| <= knee_db : gdb = -slope*(over - knee_db/2)^2 / (2*knee_db)   (knee)
//     over < 0            : gdb = slope*over                                   (below thr)
//     else               : gdb = 0                                            (above thr)
//   gdb  = max(gdb, floor_db)   (floor_db <= 0 limits the maximum attenuation)
//   g    = 10^(gdb/20);  y[n] = g * x[n]
//
// `slope` is ratio-1 (a large finite value stands in for an infinite-ratio gate, so the
// kernel never does inf arithmetic); `floor_db` is the deepest attenuation in dB.
// `state` is an optional [C, 3] = (y1, yL, ms) detector-state tensor, updated in
// place for chunk-to-chunk streaming continuity (allocated fresh when undefined).
std::tuple<torch::Tensor, torch::Tensor> expander_forward_cuda(
    const torch::Tensor& x,
    double threshold_db,
    double slope,
    double knee_db,
    double floor_db,
    double attack_coeff,
    double release_coeff,
    double rms_coeff,
    int detector,
    const torch::Tensor& state = {});

}  // namespace torchfx
