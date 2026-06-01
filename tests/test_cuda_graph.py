"""CUDA Graph capture: correctness and streaming-state continuity (Roadmap C1).

Verifies that replaying a captured graph over a sequence of fixed-shape chunks
reproduces eager chunked streaming bit-for-bit-close, including DF1 state carried
across replays (which relies on the native SOS kernel's in-place state update).
"""

from __future__ import annotations

import pytest
import torch

from torchfx.filter import HiButterworth, LoButterworth
from torchfx.filter.biquad import BiquadLPF
from torchfx.filter.fused import FusedSOSCascade

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_chain(fs: int) -> list:
    return [
        HiButterworth(80, order=2, fs=fs),
        LoButterworth(8000, order=4, fs=fs),
        BiquadLPF(2000, 0.707, fs=fs),
    ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_graph_matches_eager_streaming(dtype):
    from torchfx.realtime.cuda_graph import CudaGraphRunner

    fs, chunk = 48000, 1024
    g = torch.Generator().manual_seed(0)
    chunks = [torch.randn(2, chunk, generator=g, dtype=dtype).cuda() for _ in range(8)]

    eager = FusedSOSCascade(*_make_chain(fs))
    eager_out = [eager(c.clone()) for c in chunks]

    graphed = FusedSOSCascade(*_make_chain(fs))
    runner = CudaGraphRunner(graphed, chunks[0].clone())
    runner.reset_state()  # start from a zero state, matching eager's fresh start
    graphed_out = [runner.run(c).clone() for c in chunks]

    tol = {"rtol": 1e-4, "atol": 1e-5} if dtype == torch.float32 else {"rtol": 1e-9, "atol": 1e-11}
    for i, (e, go) in enumerate(zip(eager_out, graphed_out, strict=True)):
        torch.testing.assert_close(go, e, msg=f"chunk {i} mismatch", **tol)


def test_cuda_graph_rejects_shape_mismatch():
    from torchfx.realtime.cuda_graph import CudaGraphRunner

    fs = 48000
    runner = CudaGraphRunner(
        FusedSOSCascade(*_make_chain(fs)),
        torch.randn(2, 1024, dtype=torch.float32).cuda(),
    )
    with pytest.raises(ValueError, match="shape"):
        runner.run(torch.randn(2, 512, dtype=torch.float32).cuda())


def test_cuda_graph_requires_cuda_example():
    from torchfx.realtime.cuda_graph import CudaGraphRunner

    with pytest.raises(ValueError, match="CUDA"):
        CudaGraphRunner(FusedSOSCascade(*_make_chain(48000)), torch.randn(2, 1024))
