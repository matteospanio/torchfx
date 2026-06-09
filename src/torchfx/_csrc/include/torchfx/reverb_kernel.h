#pragma once

#include <torch/types.h>

namespace torchfx {

// Freeverb-style algorithmic reverb: per channel, 8 parallel low-pass-feedback comb
// filters summed, then fed through 4 series all-pass diffusers, mixed wet/dry.
//
// Comb (i):  out = buf[idx]; store = out*(1-damp) + store*damp;
//            buf[idx] = in*input_gain + store*feedback;  acc += out
// Allpass(j): bo = buf[idx]; buf[idx] = acc + bo*allpass_fb;  acc = bo - acc
// y[n] = x[n]*dry + acc*wet
//
// The classic Schroeder/Moorer comb tunings (at 44.1 kHz) scaled to the signal's `fs`;
// ring buffers live in a zero-initialised scratch tensor, one contiguous block per
// channel. State is per-call (not carried across forward() calls).

// Comb / all-pass delay tunings in samples at 44.1 kHz (Freeverb defaults).
constexpr int kReverbNumCombs = 8;
constexpr int kReverbNumAllpass = 4;
constexpr int kReverbCombTuning[kReverbNumCombs] = {1116, 1188, 1277, 1356,
                                                    1422, 1491, 1557, 1617};
constexpr int kReverbAllpassTuning[kReverbNumAllpass] = {556, 441, 341, 225};
constexpr double kReverbTuningFs = 44100.0;

torch::Tensor reverb_forward_cuda(
    const torch::Tensor& x,
    int fs,
    double feedback,
    double damp,
    double input_gain,
    double allpass_fb,
    double wet,
    double dry);

}  // namespace torchfx
