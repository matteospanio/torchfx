"""Standalone tests for torchfx.filter._fftconv utilities."""

import pytest
import torch
from torch.nn import functional as F

from torchfx.filter._fftconv import fft_conv1d, pad_to, unfold


class TestPadTo:
    def test_pads_to_target(self):
        x = torch.ones(5)
        result = pad_to(x, 8)
        assert result.shape == (8,)
        assert torch.equal(result[:5], x)
        assert torch.equal(result[5:], torch.zeros(3))

    def test_no_op_when_already_target(self):
        x = torch.randn(10)
        result = pad_to(x, 10)
        assert torch.equal(result, x)

    def test_multidim(self):
        x = torch.randn(3, 4, 5)
        result = pad_to(x, 8)
        assert result.shape == (3, 4, 8)


class TestUnfold:
    def test_basic_shapes(self):
        x = torch.randn(100)
        frames = unfold(x, kernel_size=10, stride=5)
        n_frames = (100 - 10) // 5 + 1
        assert frames.shape == (n_frames, 10)

    def test_content_matches_naive_slicing(self):
        x = torch.arange(20, dtype=torch.float32)
        frames = unfold(x, kernel_size=5, stride=3)
        for i in range(frames.shape[0]):
            start = i * 3
            end = start + 5
            if end <= 20:
                assert torch.equal(frames[i], x[start:end])

    def test_batched(self):
        x = torch.randn(4, 2, 100)
        frames = unfold(x, kernel_size=10, stride=5)
        assert frames.shape[:-2] == (4, 2)
        assert frames.shape[-1] == 10

    def test_covers_all_positions(self):
        x = torch.randn(17)
        frames = unfold(x, kernel_size=5, stride=3)
        # Every position in x should be covered by at least one frame
        covered = set()
        for i in range(frames.shape[0]):
            for j in range(5):
                pos = i * 3 + j
                if pos < 17:
                    covered.add(pos)
        assert covered == set(range(17))


class TestFftConv1d:
    @pytest.mark.parametrize("K", [3, 5, 16, 32, 64, 128, 256])
    def test_matches_conv1d(self, K):
        """fft_conv1d output matches F.conv1d within tolerance."""
        T = 4410
        x = torch.randn(1, 1, T)
        w = torch.randn(1, 1, K)
        pad = K - 1

        expected = F.conv1d(F.pad(x, (pad, 0)), w)
        result = fft_conv1d(x, w, padding=(pad, 0))

        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_multichannel(self):
        """Same kernel is applied independently to each channel."""
        x = torch.randn(2, 4, 1000)
        w = torch.randn(1, 1, 32)
        pad = 31

        result = fft_conv1d(x, w, padding=(pad, 0))
        assert result.shape == (2, 4, 1000)

        # Compare per-channel against direct conv1d
        for b in range(2):
            for c in range(4):
                ch = x[b : b + 1, c : c + 1]
                expected = F.conv1d(F.pad(ch, (pad, 0)), w)
                assert torch.allclose(result[b, c], expected[0, 0], atol=1e-4)

    def test_raises_on_short_input(self):
        x = torch.randn(1, 1, 5)
        w = torch.randn(1, 1, 10)
        with pytest.raises(RuntimeError, match="kernel size"):
            fft_conv1d(x, w)

    def test_raises_on_bad_block_ratio(self):
        x = torch.randn(1, 1, 100)
        w = torch.randn(1, 1, 5)
        with pytest.raises(RuntimeError, match="Block ratio"):
            fft_conv1d(x, w, block_ratio=0.5)

    @pytest.mark.parametrize("T", [100, 1000, 44100])
    def test_various_lengths(self, T):
        x = torch.randn(1, 1, T)
        w = torch.randn(1, 1, 64)
        pad = 63
        expected = F.conv1d(F.pad(x, (pad, 0)), w)
        result = fft_conv1d(x, w, padding=(pad, 0))
        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-4, rtol=1e-4)

    def test_symmetric_padding(self):
        x = torch.randn(1, 1, 200)
        w = torch.randn(1, 1, 16)
        result = fft_conv1d(x, w, padding=(8, 7))
        expected = F.conv1d(F.pad(x, (8, 7)), w)
        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-4, rtol=1e-4)
