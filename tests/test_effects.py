import pytest
import torch

from torchfx.effects import (
    Gain,
    NormalizationStrategy,
    Normalize,
    PeakNormalizationStrategy,
    PercentileNormalizationStrategy,
    PerChannelNormalizationStrategy,
    Reverb,
    RMSNormalizationStrategy,
)


class DummyStrategy(NormalizationStrategy):
    def __call__(self, waveform, peak):
        return waveform * 0 + peak


def test_gain_amplitude():
    waveform = torch.tensor([0.1, -0.2, 0.3])
    gain = Gain(gain=2.0, gain_type="amplitude")
    out = gain(waveform)
    torch.testing.assert_close(out, waveform * 2.0)


def test_gain_db(monkeypatch):
    waveform = torch.tensor([0.1, -0.2, 0.3])
    called = {}

    def fake_gain(waveform, gain):
        called["args"] = (waveform, gain)
        return waveform + gain

    monkeypatch.setattr("torchaudio.functional.gain", fake_gain)
    gain = Gain(gain=6.0, gain_type="db")
    out = gain(waveform)
    assert torch.allclose(out, waveform + 6.0)
    assert called["args"][1] == 6.0


def test_gain_power(monkeypatch):
    waveform = torch.tensor([0.1, -0.2, 0.3])
    called = {}

    def fake_gain(waveform, gain):
        called["args"] = (waveform, gain)
        return waveform + gain

    monkeypatch.setattr("torchaudio.functional.gain", fake_gain)
    gain = Gain(gain=10.0, gain_type="power")
    out = gain(waveform)
    expected_gain = 10 * torch.log10(torch.tensor(10.0)).item()
    assert torch.allclose(out, waveform + expected_gain)
    assert called["args"][1] == expected_gain


def test_gain_clamp():
    waveform = torch.tensor([2.0, -2.0, 0.5])
    gain = Gain(gain=1.0, clamp=True)
    out = gain(waveform)
    torch.testing.assert_close(out, torch.tensor([1.0, -1.0, 0.5]))


def test_gain_invalid_gain_type():
    with pytest.raises(ValueError):
        Gain(gain=-1.0, gain_type="amplitude")
    with pytest.raises(ValueError):
        Gain(gain=-1.0, gain_type="power")


def test_normalize_peak_strategy():
    waveform = torch.tensor([0.2, -0.5, 0.4])
    norm = Normalize(peak=1.0)
    out = norm(waveform)
    torch.testing.assert_close(out, waveform / 0.5 * 1.0)


def test_normalize_custom_strategy():
    waveform = torch.tensor([0.2, -0.5, 0.4])
    norm = Normalize(peak=2.0, strategy=DummyStrategy())
    out = norm(waveform)
    torch.testing.assert_close(out, torch.full_like(waveform, 2.0))


def test_normalize_invalid_peak():
    with pytest.raises(ValueError):
        Normalize(peak=0)


def test_normalize_invalid_strategy():
    with pytest.raises(TypeError):
        Normalize(peak=1.0, strategy="not_a_strategy")


def test_peak_normalization_strategy():
    waveform = torch.tensor([0.2, -0.5, 0.4])
    strat = PeakNormalizationStrategy()
    out = strat(waveform, 2.0)
    torch.testing.assert_close(out, waveform / 0.5 * 2.0)


def test_peak_normalization_strategy_zero():
    waveform = torch.zeros(3)
    strat = PeakNormalizationStrategy()
    out = strat(waveform, 1.0)
    torch.testing.assert_close(out, waveform)


def test_rms_normalization_strategy():
    waveform = torch.tensor([3.0, 4.0])
    strat = RMSNormalizationStrategy()
    rms = torch.sqrt(torch.mean(waveform**2))
    out = strat(waveform, 2.0)
    torch.testing.assert_close(out, waveform / rms * 2.0)


def test_rms_normalization_strategy_zero():
    waveform = torch.zeros(3)
    strat = RMSNormalizationStrategy()
    out = strat(waveform, 1.0)
    torch.testing.assert_close(out, waveform)


def test_percentile_normalization_strategy():
    waveform = torch.tensor([1.0, 2.0, 3.0, 4.0])
    strat = PercentileNormalizationStrategy(percentile=50.0)
    out = strat(waveform, 2.0)
    threshold = torch.quantile(torch.abs(waveform), 0.5, interpolation="linear")
    torch.testing.assert_close(out, waveform / threshold * 2.0)


def test_percentile_normalization_strategy_invalid():
    with pytest.raises(ValueError):
        PercentileNormalizationStrategy(percentile=0)
    with pytest.raises(ValueError):
        PercentileNormalizationStrategy(percentile=101)


def test_percentile_normalization_strategy_zero():
    waveform = torch.zeros(4)
    strat = PercentileNormalizationStrategy(percentile=50.0)
    out = strat(waveform, 1.0)
    torch.testing.assert_close(out, waveform)


def test_per_channel_normalization_strategy_2d():
    waveform = torch.tensor([[1.0, -2.0], [0.5, -0.5]])
    strat = PerChannelNormalizationStrategy()
    out = strat(waveform, 1.0)
    expected = torch.tensor([[1.0 / 2.0, -2.0 / 2.0], [0.5 / 0.5, -0.5 / 0.5]])
    torch.testing.assert_close(out, expected)


def test_per_channel_normalization_strategy_3d():
    waveform = torch.tensor([[[1.0, -2.0], [0.5, -0.5]], [[2.0, -4.0], [1.0, -1.0]]])
    strat = PerChannelNormalizationStrategy()
    out = strat(waveform, 1.0)
    expected = torch.empty_like(waveform)
    expected[0, 0] = waveform[0, 0] / 2.0
    expected[0, 1] = waveform[0, 1] / 0.5
    expected[1, 0] = waveform[1, 0] / 4.0
    expected[1, 1] = waveform[1, 1] / 1.0
    torch.testing.assert_close(out, expected)


def test_per_channel_normalization_strategy_invalid_shape():
    waveform = torch.tensor([1.0, -2.0])
    strat = PerChannelNormalizationStrategy()
    with pytest.raises(ValueError):
        strat(waveform, 1.0)


def test_per_channel_normalization_strategy_zero():
    waveform = torch.zeros((2, 3))
    strat = PerChannelNormalizationStrategy()
    out = strat(waveform, 1.0)
    torch.testing.assert_close(out, waveform)


def test_reverb_basic():
    # Simple waveform, delay=2, decay=0.5, mix=1.0 (fully wet)
    waveform = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    reverb = Reverb(delay=2, decay=0.5, mix=1.0)
    # Expected: y[n] = x[n] + 0.5 * x[n-2] for n >= 2, else x[n]
    expected = torch.tensor(
        [
            1.0,  # n=0: no delay
            2.0,  # n=1: no delay
            3.0 + 0.5 * 1.0,  # n=2
            4.0 + 0.5 * 2.0,  # n=3
            5.0 + 0.5 * 3.0,  # n=4
        ]
    )
    out = reverb(waveform)
    torch.testing.assert_close(out, expected)


def test_reverb_mix_zero():
    # mix=0 should return the original waveform
    waveform = torch.randn(10)
    reverb = Reverb(delay=3, decay=0.7, mix=0.0)
    out = reverb(waveform)
    torch.testing.assert_close(out, waveform)


def test_reverb_short_waveform():
    # If waveform shorter than delay, should return unchanged
    waveform = torch.tensor([1.0, 2.0])
    reverb = Reverb(delay=3, decay=0.5, mix=1.0)
    out = reverb(waveform)
    torch.testing.assert_close(out, waveform)


def test_reverb_multichannel():
    # Test with 2D waveform (channels, time)
    waveform = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]])
    reverb = Reverb(delay=2, decay=0.5, mix=1.0)
    expected = torch.empty_like(waveform)
    # Channel 0
    expected[0, 0] = 1.0
    expected[0, 1] = 2.0
    expected[0, 2] = 3.0 + 0.5 * 1.0
    expected[0, 3] = 4.0 + 0.5 * 2.0
    # Channel 1
    expected[1, 0] = 0.5
    expected[1, 1] = 1.5
    expected[1, 2] = 2.5 + 0.5 * 0.5
    expected[1, 3] = 3.5 + 0.5 * 1.5
    out = reverb(waveform)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("delay", [0, -1])
def test_reverb_invalid_delay(delay):
    with pytest.raises(ValueError):
        Reverb(delay=delay, decay=0.5, mix=0.5)


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.1, 1.1])
def test_reverb_invalid_decay(decay):
    with pytest.raises(ValueError):
        Reverb(delay=2, decay=decay, mix=0.5)


@pytest.mark.parametrize("mix", [-0.1, 1.1])
def test_reverb_invalid_mix(mix):
    with pytest.raises(ValueError):
        Reverb(delay=2, decay=0.5, mix=mix)
