#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include "torchfx/reverb_kernel.h"

// CPU Freeverb-style reverb. One channel per OpenMP thread; ring buffers live in a
// zero-initialised scratch block per channel. See include/torchfx/reverb_kernel.h.

#if defined(_MSC_VER)
#define TORCHFX_RESTRICT __restrict
#else
#define TORCHFX_RESTRICT __restrict__
#endif

namespace {

void reverb_dims(int fs, int* comb_len, int* comb_off, int* ap_len, int* ap_off,
                 int& comb_total, int& total) {
  comb_total = 0;
  for (int i = 0; i < torchfx::kReverbNumCombs; ++i) {
    const int len = std::max<int>(
        1, static_cast<int>(std::lround(torchfx::kReverbCombTuning[i] * fs / torchfx::kReverbTuningFs)));
    comb_len[i] = len;
    comb_off[i] = comb_total;
    comb_total += len;
  }
  int ap_total = 0;
  for (int j = 0; j < torchfx::kReverbNumAllpass; ++j) {
    const int len = std::max<int>(
        1, static_cast<int>(std::lround(torchfx::kReverbAllpassTuning[j] * fs / torchfx::kReverbTuningFs)));
    ap_len[j] = len;
    ap_off[j] = ap_total;
    ap_total += len;
  }
  total = comb_total + ap_total;
}

template <typename scalar_t>
void reverb_loop(
    const scalar_t* TORCHFX_RESTRICT in_ptr,
    scalar_t* TORCHFX_RESTRICT out_ptr,
    scalar_t* TORCHFX_RESTRICT scratch,
    scalar_t* TORCHFX_RESTRICT fstore_ptr,  // [C, kReverbNumCombs] damping state, in/out
    int* TORCHFX_RESTRICT idx_ptr,          // [C, combs+allpasses] ring positions, in/out
    int64_t C,
    int64_t T,
    const int* comb_len,
    const int* comb_off,
    const int* ap_len,
    const int* ap_off,
    int comb_total,
    int total,
    scalar_t feedback,
    scalar_t damp,
    scalar_t input_gain,
    scalar_t allpass_fb,
    scalar_t wet,
    scalar_t dry) {

  const scalar_t one = static_cast<scalar_t>(1);

  #pragma omp parallel for schedule(static) if (C > 1)
  for (int64_t c = 0; c < C; ++c) {
    const scalar_t* in_c = in_ptr + c * T;
    scalar_t* out_c = out_ptr + c * T;
    scalar_t* buf = scratch + c * total;

    scalar_t fstore[torchfx::kReverbNumCombs];
    int cidx[torchfx::kReverbNumCombs];
    int aidx[torchfx::kReverbNumAllpass];
    scalar_t* fs_c = fstore_ptr + c * torchfx::kReverbNumCombs;
    int* idx_c = idx_ptr + c * (torchfx::kReverbNumCombs + torchfx::kReverbNumAllpass);
    for (int i = 0; i < torchfx::kReverbNumCombs; ++i) {
      fstore[i] = fs_c[i];
      cidx[i] = idx_c[i];
    }
    for (int j = 0; j < torchfx::kReverbNumAllpass; ++j)
      aidx[j] = idx_c[torchfx::kReverbNumCombs + j];

    for (int64_t n = 0; n < T; ++n) {
      const scalar_t xn = in_c[n];
      const scalar_t inp = xn * input_gain;
      scalar_t acc = 0;

      for (int i = 0; i < torchfx::kReverbNumCombs; ++i) {
        scalar_t* cb = buf + comb_off[i];
        const scalar_t bo = cb[cidx[i]];
        fstore[i] = bo * (one - damp) + fstore[i] * damp;
        cb[cidx[i]] = inp + fstore[i] * feedback;
        if (++cidx[i] >= comb_len[i]) cidx[i] = 0;
        acc += bo;
      }
      for (int j = 0; j < torchfx::kReverbNumAllpass; ++j) {
        scalar_t* ab = buf + comb_total + ap_off[j];
        const scalar_t bo = ab[aidx[j]];
        ab[aidx[j]] = acc + bo * allpass_fb;
        acc = bo - acc;
        if (++aidx[j] >= ap_len[j]) aidx[j] = 0;
      }
      out_c[n] = xn * dry + acc * wet;
    }

    for (int i = 0; i < torchfx::kReverbNumCombs; ++i) {
      fs_c[i] = fstore[i];
      idx_c[i] = cidx[i];
    }
    for (int j = 0; j < torchfx::kReverbNumAllpass; ++j)
      idx_c[torchfx::kReverbNumCombs + j] = aidx[j];
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> reverb_forward_cpu(
    const torch::Tensor& x,
    int fs,
    double feedback,
    double damp,
    double input_gain,
    double allpass_fb,
    double wet,
    double dry,
    const torch::Tensor& scratch_in,
    const torch::Tensor& fstore_in,
    const torch::Tensor& idx_in) {

  TORCH_CHECK(!x.is_cuda(), "reverb_forward_cpu: input must be on CPU");

  auto x_cont = x.contiguous();
  const int orig_dim = x_cont.dim();
  if (orig_dim == 1) {
    x_cont = x_cont.unsqueeze(0);
  }
  const int64_t C = x_cont.size(0);
  const int64_t T = x_cont.size(1);

  int comb_len[torchfx::kReverbNumCombs], comb_off[torchfx::kReverbNumCombs];
  int ap_len[torchfx::kReverbNumAllpass], ap_off[torchfx::kReverbNumAllpass];
  int comb_total, total;
  reverb_dims(fs, comb_len, comb_off, ap_len, ap_off, comb_total, total);

  auto output = torch::empty_like(x_cont);
  const int n_idx = torchfx::kReverbNumCombs + torchfx::kReverbNumAllpass;

  // Delay-line state: fresh zeros when not supplied, otherwise updated in place
  // for chunk-to-chunk streaming continuity. All three must be passed together.
  torch::Tensor scratch = scratch_in, fstore = fstore_in, idx = idx_in;
  if (!scratch.defined()) {
    scratch = torch::zeros({C, static_cast<int64_t>(total)}, x_cont.options());
    fstore = torch::zeros({C, torchfx::kReverbNumCombs}, x_cont.options());
    idx = torch::zeros({C, n_idx}, x_cont.options().dtype(torch::kInt32));
  } else {
    TORCH_CHECK(fstore.defined() && idx.defined(),
                "reverb_forward_cpu: scratch/fstore/idx must be passed together");
    TORCH_CHECK(scratch.size(0) == C && scratch.size(1) == total &&
                    scratch.dtype() == x_cont.dtype(),
                "reverb_forward_cpu: scratch must be [C, ", total, "] with the input dtype "
                "(state from a different fs or channel count is not reusable)");
    TORCH_CHECK(fstore.size(0) == C && fstore.size(1) == torchfx::kReverbNumCombs &&
                    fstore.dtype() == x_cont.dtype(),
                "reverb_forward_cpu: fstore must be [C, ", torchfx::kReverbNumCombs, "]");
    TORCH_CHECK(idx.size(0) == C && idx.size(1) == n_idx && idx.dtype() == torch::kInt32,
                "reverb_forward_cpu: idx must be int32 [C, ", n_idx, "]");
    scratch = scratch.contiguous();
    fstore = fstore.contiguous();
    idx = idx.contiguous();
  }

  if (x_cont.dtype() == torch::kFloat32) {
    reverb_loop<float>(
        x_cont.data_ptr<float>(), output.data_ptr<float>(), scratch.data_ptr<float>(),
        fstore.data_ptr<float>(), idx.data_ptr<int>(),
        C, T, comb_len, comb_off, ap_len, ap_off, comb_total, total,
        static_cast<float>(feedback), static_cast<float>(damp), static_cast<float>(input_gain),
        static_cast<float>(allpass_fb), static_cast<float>(wet), static_cast<float>(dry));
  } else {
    reverb_loop<double>(
        x_cont.data_ptr<double>(), output.data_ptr<double>(), scratch.data_ptr<double>(),
        fstore.data_ptr<double>(), idx.data_ptr<int>(),
        C, T, comb_len, comb_off, ap_len, ap_off, comb_total, total,
        feedback, damp, input_gain, allpass_fb, wet, dry);
  }

  if (orig_dim == 1) {
    return {output.squeeze(0), scratch, fstore, idx};
  }
  return {output, scratch, fstore, idx};
}
