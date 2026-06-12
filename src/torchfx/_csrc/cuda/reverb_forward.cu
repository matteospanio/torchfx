#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
#include <cmath>
#include "torchfx/reverb_kernel.h"

// Freeverb-style reverb, one thread per channel (sequential over time, ring buffers in a
// per-channel scratch block). Templated on scalar_t so float32 runs natively in FP32. See
// reverb_kernel.h for the math.

namespace torchfx {

struct ReverbDims {
  int comb_len[kReverbNumCombs];
  int comb_off[kReverbNumCombs];
  int ap_len[kReverbNumAllpass];
  int ap_off[kReverbNumAllpass];
  int comb_total;
  int total;
};

template <typename scalar_t>
__global__ void reverb_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ scratch,
    scalar_t* __restrict__ fstore_g,  // [C, kReverbNumCombs] damping state, in/out
    int* __restrict__ idx_g,          // [C, combs+allpasses] ring positions, in/out
    int C,
    int T,
    ReverbDims dims,
    scalar_t feedback,
    scalar_t damp,
    scalar_t input_gain,
    scalar_t allpass_fb,
    scalar_t wet,
    scalar_t dry) {

  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;

  const scalar_t one = static_cast<scalar_t>(1);
  const scalar_t* in_c = input + static_cast<int64_t>(c) * T;
  scalar_t* out_c = output + static_cast<int64_t>(c) * T;
  scalar_t* buf = scratch + static_cast<int64_t>(c) * dims.total;

  scalar_t fstore[kReverbNumCombs];
  int cidx[kReverbNumCombs];
  int aidx[kReverbNumAllpass];
  scalar_t* fs_c = fstore_g + static_cast<int64_t>(c) * kReverbNumCombs;
  int* idx_c = idx_g + static_cast<int64_t>(c) * (kReverbNumCombs + kReverbNumAllpass);
#pragma unroll
  for (int i = 0; i < kReverbNumCombs; ++i) {
    fstore[i] = fs_c[i];
    cidx[i] = idx_c[i];
  }
#pragma unroll
  for (int j = 0; j < kReverbNumAllpass; ++j) aidx[j] = idx_c[kReverbNumCombs + j];

  for (int n = 0; n < T; ++n) {
    const scalar_t xn = in_c[n];
    const scalar_t inp = xn * input_gain;
    scalar_t acc = 0;

#pragma unroll
    for (int i = 0; i < kReverbNumCombs; ++i) {
      scalar_t* cb = buf + dims.comb_off[i];
      const scalar_t bo = cb[cidx[i]];
      fstore[i] = bo * (one - damp) + fstore[i] * damp;
      cb[cidx[i]] = inp + fstore[i] * feedback;
      if (++cidx[i] >= dims.comb_len[i]) cidx[i] = 0;
      acc += bo;
    }
#pragma unroll
    for (int j = 0; j < kReverbNumAllpass; ++j) {
      scalar_t* ab = buf + dims.comb_total + dims.ap_off[j];
      const scalar_t bo = ab[aidx[j]];
      ab[aidx[j]] = acc + bo * allpass_fb;
      acc = bo - acc;
      if (++aidx[j] >= dims.ap_len[j]) aidx[j] = 0;
    }
    out_c[n] = xn * dry + acc * wet;
  }

#pragma unroll
  for (int i = 0; i < kReverbNumCombs; ++i) {
    fs_c[i] = fstore[i];
    idx_c[i] = cidx[i];
  }
#pragma unroll
  for (int j = 0; j < kReverbNumAllpass; ++j) idx_c[kReverbNumCombs + j] = aidx[j];
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> reverb_forward_cuda(
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

  TORCH_CHECK(x.is_cuda(), "reverb_forward_cuda: input must be on CUDA");

  auto x_cont = x.contiguous();
  int64_t C, T;
  if (x_cont.dim() == 1) {
    C = 1;
    T = x_cont.size(0);
    x_cont = x_cont.unsqueeze(0);
  } else {
    C = x_cont.size(0);
    T = x_cont.size(1);
  }

  ReverbDims dims;
  dims.comb_total = 0;
  for (int i = 0; i < kReverbNumCombs; ++i) {
    const int len =
        std::max<int>(1, static_cast<int>(std::lround(kReverbCombTuning[i] * fs / kReverbTuningFs)));
    dims.comb_len[i] = len;
    dims.comb_off[i] = dims.comb_total;
    dims.comb_total += len;
  }
  int ap_total = 0;
  for (int j = 0; j < kReverbNumAllpass; ++j) {
    const int len = std::max<int>(
        1, static_cast<int>(std::lround(kReverbAllpassTuning[j] * fs / kReverbTuningFs)));
    dims.ap_len[j] = len;
    dims.ap_off[j] = ap_total;
    ap_total += len;
  }
  dims.total = dims.comb_total + ap_total;

  auto output = torch::empty_like(x_cont);
  const int n_idx = kReverbNumCombs + kReverbNumAllpass;

  // Delay-line state: fresh zeros when not supplied, otherwise updated in place
  // for chunk-to-chunk streaming continuity. All three must be passed together.
  torch::Tensor scratch = scratch_in, fstore = fstore_in, idx = idx_in;
  if (!scratch.defined()) {
    scratch = torch::zeros({C, static_cast<int64_t>(dims.total)}, x_cont.options());
    fstore = torch::zeros({C, kReverbNumCombs}, x_cont.options());
    idx = torch::zeros({C, n_idx}, x_cont.options().dtype(torch::kInt32));
  } else {
    TORCH_CHECK(fstore.defined() && idx.defined(),
                "reverb_forward_cuda: scratch/fstore/idx must be passed together");
    TORCH_CHECK(scratch.size(0) == C && scratch.size(1) == dims.total &&
                    scratch.dtype() == x_cont.dtype() && scratch.is_cuda(),
                "reverb_forward_cuda: scratch must be CUDA [C, ", dims.total,
                "] with the input dtype");
    TORCH_CHECK(fstore.size(0) == C && fstore.size(1) == kReverbNumCombs &&
                    fstore.dtype() == x_cont.dtype() && fstore.is_cuda(),
                "reverb_forward_cuda: fstore must be CUDA [C, ", kReverbNumCombs, "]");
    TORCH_CHECK(idx.size(0) == C && idx.size(1) == n_idx && idx.dtype() == torch::kInt32 &&
                    idx.is_cuda(),
                "reverb_forward_cuda: idx must be CUDA int32 [C, ", n_idx, "]");
    scratch = scratch.contiguous();
    fstore = fstore.contiguous();
    idx = idx.contiguous();
  }

  const int threads = 128;
  const int blocks = (static_cast<int>(C) + threads - 1) / threads;
  const auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(x_cont.scalar_type(), "reverb_forward_cuda", [&] {
    reverb_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        x_cont.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), scratch.data_ptr<scalar_t>(),
        fstore.data_ptr<scalar_t>(), idx.data_ptr<int>(),
        static_cast<int>(C), static_cast<int>(T), dims, static_cast<scalar_t>(feedback),
        static_cast<scalar_t>(damp), static_cast<scalar_t>(input_gain),
        static_cast<scalar_t>(allpass_fb), static_cast<scalar_t>(wet), static_cast<scalar_t>(dry));
  });

  if (x.dim() == 1) {
    return std::make_tuple(output.squeeze(0), scratch, fstore, idx);
  }
  return std::make_tuple(output, scratch, fstore, idx);
}

}  // namespace torchfx
