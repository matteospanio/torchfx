"""Tests for :mod:`torchfx._ops` — the native-extension dispatch layer.

Baseline coverage was 64% with only indirect coverage via higher-level filter
tests. The CUDA branches are genuinely unreachable on this workstation, but
the CPU-side dispatch contract — state shape handling, fallback-to-``None``
paths, and the ``TORCHFX_NO_CUDA`` env switch — is fully testable here.

"""

from __future__ import annotations

import importlib

import pytest
import torch

from torchfx import _ops

# ---------- module constants & availability ----------


class TestAvailability:
    def test_parallel_scan_threshold(self):
        """The threshold is a public knob — lock the default so tuning is deliberate."""
        assert _ops.PARALLEL_SCAN_THRESHOLD == 2048

    def test_is_native_available_is_bool(self):
        assert isinstance(_ops.is_native_available(), bool)

    def test_is_native_available_caches_load_attempt(self):
        """After the first call, ``_ext_load_attempted`` must be sticky."""
        _ = _ops.is_native_available()
        assert _ops._ext_load_attempted is True

    def test_reload_with_torchfx_no_cuda(self, monkeypatch):
        """Setting TORCHFX_NO_CUDA before first load disables the CUDA branch.

        The CPU-only extension still compiles, so ``is_native_available`` may
        still return True on systems with a working toolchain. The point of
        the env switch is that *no CUDA sources* are compiled into the
        resulting module. We verify it by re-importing the module with the
        env var set and confirming the load attempt runs cleanly.

        """
        monkeypatch.setenv("TORCHFX_NO_CUDA", "1")
        mod = importlib.reload(_ops)
        try:
            # Just run the loader — we don't care whether it succeeds on this
            # box, only that the branch evaluates ``use_cuda = False`` and
            # doesn't raise.
            _ = mod.is_native_available()
            assert mod._ext_load_attempted is True
        finally:
            importlib.reload(_ops)  # restore module state for other tests


# ---------- biquad dispatch ----------


class TestBiquadDispatch:
    def _coeffs(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Trivial pass-through biquad: b=[1,0,0], a=[1,0,0].
        b = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        return b, a

    def test_returns_none_when_extension_missing(self, monkeypatch):
        monkeypatch.setattr(_ops, "_ext", None)
        monkeypatch.setattr(_ops, "_ext_load_attempted", True)
        b, a = self._coeffs()
        x = torch.randn(2, 128, dtype=torch.float64)
        assert _ops.biquad_forward(x, b, a, None, None) is None

    @pytest.mark.skipif(not _ops.is_native_available(), reason="native ext not built")
    def test_state_defaults_allocated(self):
        b, a = self._coeffs()
        x = torch.randn(2, 256, dtype=torch.float64)
        out = _ops.biquad_forward(x, b, a, None, None)
        assert out is not None
        y, sx, sy = out
        assert y.shape == x.shape
        assert sx.shape == (2, 2)
        assert sy.shape == (2, 2)

    @pytest.mark.skipif(not _ops.is_native_available(), reason="native ext not built")
    def test_state_roundtrip(self):
        """Feeding back the returned state must not raise and must preserve shape."""
        b, a = self._coeffs()
        x1 = torch.randn(2, 128, dtype=torch.float64)
        x2 = torch.randn(2, 128, dtype=torch.float64)

        out1 = _ops.biquad_forward(x1, b, a, None, None)
        assert out1 is not None
        _, sx, sy = out1

        out2 = _ops.biquad_forward(x2, b, a, sx, sy)
        assert out2 is not None
        y2, sx2, sy2 = out2
        assert y2.shape == x2.shape
        assert sx2.shape == sx.shape
        assert sy2.shape == sy.shape

    @pytest.mark.skipif(not _ops.is_native_available(), reason="native ext not built")
    def test_falls_back_to_none_on_kernel_exception(self, monkeypatch):
        """If the kernel raises, dispatch must swallow and return None."""
        b, a = self._coeffs()
        x = torch.randn(2, 64, dtype=torch.float64)

        class _BadExt:
            def biquad_forward(self, *args, **kwargs):  # noqa: ARG002
                raise RuntimeError("boom")

        monkeypatch.setattr(_ops, "_ext", _BadExt())
        monkeypatch.setattr(_ops, "_ext_load_attempted", True)
        assert _ops.biquad_forward(x, b, a, None, None) is None


# ---------- SOS cascade dispatch ----------


class TestSosDispatch:
    def _sos(self) -> torch.Tensor:
        # Two pass-through biquad sections.
        return torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )

    def test_returns_none_when_extension_missing(self, monkeypatch):
        monkeypatch.setattr(_ops, "_ext", None)
        monkeypatch.setattr(_ops, "_ext_load_attempted", True)
        sos = self._sos()
        x = torch.randn(2, 128, dtype=torch.float64)
        assert _ops.parallel_iir_forward(x, sos, None, None) is None

    @pytest.mark.skipif(not _ops.is_native_available(), reason="native ext not built")
    def test_state_defaults_allocated(self):
        sos = self._sos()
        x = torch.randn(2, 256, dtype=torch.float64)
        out = _ops.parallel_iir_forward(x, sos, None, None)
        assert out is not None
        y, sx, sy = out
        assert y.shape == x.shape
        # state is (K, C, 2)
        assert sx.shape == (sos.shape[0], x.shape[0], 2)
        assert sy.shape == (sos.shape[0], x.shape[0], 2)

    @pytest.mark.skipif(not _ops.is_native_available(), reason="native ext not built")
    def test_falls_back_to_none_on_kernel_exception(self, monkeypatch):
        sos = self._sos()
        x = torch.randn(2, 64, dtype=torch.float64)

        class _BadExt:
            def sos_forward(self, *args, **kwargs):  # noqa: ARG002
                raise RuntimeError("boom")

        monkeypatch.setattr(_ops, "_ext", _BadExt())
        monkeypatch.setattr(_ops, "_ext_load_attempted", True)
        assert _ops.parallel_iir_forward(x, sos, None, None) is None


# ---------- delay-line dispatch ----------


class TestDelayDispatch:
    def test_cpu_tensor_returns_none(self):
        """Delay dispatch is CUDA-only — CPU input must short-circuit to None."""
        x = torch.randn(2, 512)
        assert _ops.delay_line_forward(x, delay_samples=100, decay=0.5, mix=0.5) is None

    def test_returns_none_when_extension_missing(self, monkeypatch):
        """Even if a CUDA tensor could be constructed, missing ext → None.

        We can't easily fake a CUDA tensor on a CPU-only box, so this test
        locks the second guard by patching ``is_cuda`` on a CPU tensor via a
        stub object. The dispatch code only reads ``x.is_cuda``, so a minimal
        duck type is enough.

        """

        class _FakeCudaTensor:
            is_cuda = True

        monkeypatch.setattr(_ops, "_ext", None)
        monkeypatch.setattr(_ops, "_ext_load_attempted", True)
        assert (
            _ops.delay_line_forward(
                _FakeCudaTensor(),  # type: ignore[arg-type]
                delay_samples=100,
                decay=0.5,
                mix=0.5,
            )
            is None
        )
