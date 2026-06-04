"""CUDA Graph capture: correctness and streaming-state continuity (Roadmap C1).

Verifies that replaying a captured graph over a sequence of fixed-shape chunks
reproduces eager chunked streaming bit-for-bit-close, including DF1 state carried
across replays (which relies on the native SOS kernel's in-place state update and
the once-per-forward scratch buffers, C3/C4).

Both filters are warmed identically before streaming the test chunks, so this
checks the property that matters in practice — graph replay continues streaming
state exactly like eager — rather than the reset-to-zero edge case.
"""

from __future__ import annotations

import pytest
import torch

from torchfx.filter import HiButterworth, LoButterworth
from torchfx.filter.biquad import BiquadLPF
from torchfx.filter.fused import FusedSOSCascade

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

WARMUP = 3  # CudaGraphRunner default: WARMUP warmup forwards + 1 capture forward


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
    warm = torch.randn(2, chunk, generator=g, dtype=dtype).cuda()
    test_chunks = [torch.randn(2, chunk, generator=g, dtype=dtype).cuda() for _ in range(6)]

    # Eager reference: warm up identically (WARMUP + 1 capture-equivalent), then stream.
    eager = FusedSOSCascade(*_make_chain(fs))
    for _ in range(WARMUP + 1):
        eager(warm.clone())
    eager_out = [eager(c.clone()) for c in test_chunks]

    # Graphed: runner does WARMUP warmups + 1 capture forward on `warm`, then replays.
    graphed = FusedSOSCascade(*_make_chain(fs))
    runner = CudaGraphRunner(graphed, warm.clone(), warmup=WARMUP)
    graphed_out = [runner.run(c).clone() for c in test_chunks]

    # Graph replay matches eager to float precision, not bit-exactly: the graphed
    # warmup runs on a side stream + capture while eager runs straight-line, so the
    # post-warmup state differs by float noise (~2e-11 in float64) that propagates
    # but does not grow. A real logic error would be O(0.1+).
    tol = {"rtol": 1e-4, "atol": 1e-5} if dtype == torch.float32 else {"rtol": 1e-7, "atol": 1e-9}
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
