import pytest
import torch

from torchfx.effect import (
    CustomNormalizationStrategy,
    Delay,
    DelayStrategy,
    Gain,
    MonoDelayStrategy,
    NormalizationStrategy,
    Normalize,
    PeakNormalizationStrategy,
    PercentileNormalizationStrategy,
    PerChannelNormalizationStrategy,
    PingPongDelayStrategy,
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


def test_normalize_callable_strategy():
    waveform = torch.tensor([1.0, 2.0, 3.0])

    norm = Normalize(peak=5.0, strategy=lambda w, p: w + p)
    out = norm(waveform)
    torch.testing.assert_close(out, waveform + 5.0)


def test_normalize_invalid_peak():
    with pytest.raises(AssertionError):
        Normalize(peak=0)


def test_normalize_invalid_strategy():
    with pytest.raises(TypeError):
        Normalize(peak=1.0, strategy="not_a_strategy")  # type: ignore


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


def test_custom_normalization_strategy():
    waveform = torch.tensor([1.0, 2.0, 3.0])

    def custom_func(waveform, peak):
        return waveform + peak

    strat = CustomNormalizationStrategy(custom_func)
    out = strat(waveform, 5.0)
    torch.testing.assert_close(out, waveform + 5.0)


def test_percentile_normalization_strategy():
    waveform = torch.tensor([1.0, 2.0, 3.0, 4.0])
    strat = PercentileNormalizationStrategy(percentile=50.0)
    out = strat(waveform, 2.0)
    threshold = torch.quantile(torch.abs(waveform), 0.5, interpolation="linear")
    torch.testing.assert_close(out, waveform / threshold * 2.0)


def test_percentile_normalization_strategy_invalid():
    with pytest.raises(AssertionError):
        PercentileNormalizationStrategy(percentile=0)
    with pytest.raises(AssertionError):
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
    with pytest.raises(AssertionError):
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
    with pytest.raises(AssertionError):
        Reverb(delay=delay, decay=0.5, mix=0.5)


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.1, 1.1])
def test_reverb_invalid_decay(decay):
    with pytest.raises(AssertionError):
        Reverb(delay=2, decay=decay, mix=0.5)


@pytest.mark.parametrize("mix", [-0.1, 1.1])
def test_reverb_invalid_mix(mix):
    with pytest.raises(AssertionError):
        Reverb(delay=2, decay=0.5, mix=mix)


# Delay tests
def test_delay_basic():
    """Test basic delay functionality with known input/output."""
    # Simple waveform, delay=2, feedback=0.0, mix=1.0 (fully wet)
    # With feedback=0.0, only first tap should appear (amplitude 1.0)
    waveform = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    delay = Delay(delay_samples=2, feedback=0.0, mix=1.0, taps=1)
    out = delay(waveform)
    # Output should be extended: original (5) + delay (2) = 7 samples
    assert out.size(0) >= 7
    # With mix=1.0, original positions should be 0.0 (only delayed signal)
    assert out[0].item() == pytest.approx(0.0, abs=1e-5)
    assert out[1].item() == pytest.approx(0.0, abs=1e-5)
    # First tap at position 2 should have amplitude 1.0
    assert out[2].item() == pytest.approx(1.0, abs=1e-5)


def test_delay_feedback_zero():
    """Test delay with feedback=0.0 (first tap should still work)."""
    # Verify our fix: feedback=0.0 should still create first tap
    waveform = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    delay = Delay(delay_samples=2, feedback=0.0, mix=1.0, taps=2)
    out = delay(waveform)
    # First tap at position 2 should have amplitude 1.0
    # Second tap at position 4 should have amplitude 0.0 (feedback=0.0)
    assert out[2].item() == pytest.approx(1.0, abs=1e-5)
    assert out[4].item() == pytest.approx(0.0, abs=1e-5)


def test_delay_feedback():
    """Test delay with feedback creates multiple taps."""
    waveform = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    delay = Delay(delay_samples=2, feedback=0.5, mix=1.0, taps=3)
    out = delay(waveform)
    # First tap at position 2: amplitude 1.0
    # Second tap at position 4: amplitude 0.5 (feedback^1)
    # Third tap at position 6: amplitude 0.25 (feedback^2)
    assert out[2].item() == pytest.approx(1.0, abs=1e-5)
    assert out[4].item() == pytest.approx(0.5, abs=1e-5)
    assert out[6].item() == pytest.approx(0.25, abs=1e-5)


def test_delay_mix_zero():
    """Test delay with mix=0 (should return original signal, possibly extended)."""
    waveform = torch.randn(10)
    delay = Delay(delay_samples=3, feedback=0.5, mix=0.0)
    out = delay(waveform)
    # With mix=0.0, output should match original in the original region
    torch.testing.assert_close(out[:10], waveform)
    # Output may be extended, but extended region should be zeros
    if out.size(0) > 10:
        assert torch.allclose(out[10:], torch.zeros(out.size(0) - 10), atol=1e-5)


def test_delay_mix_one():
    """Test delay with mix=1.0 (should return only delayed echoes)."""
    waveform = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    delay = Delay(delay_samples=2, feedback=0.0, mix=1.0, taps=1)
    out = delay(waveform)
    # With mix=1.0, output should be only delayed (no original)
    # Original at position 0 should be 0.0
    assert out[0].item() == pytest.approx(0.0, abs=1e-5)
    # Delayed at position 2 should be 1.0
    assert out[2].item() == pytest.approx(1.0, abs=1e-5)


def test_delay_mix_half():
    """Test delay with mix=0.5 (50% original, 50% delayed)."""
    waveform = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    delay = Delay(delay_samples=2, feedback=0.0, mix=0.5, taps=1)
    out = delay(waveform)
    # output = 0.5 * original + 0.5 * delayed
    # Position 0: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    # Position 2: 0.5 * 0.0 + 0.5 * 1.0 = 0.5
    assert out[0].item() == pytest.approx(0.5, abs=1e-5)
    assert out[2].item() == pytest.approx(0.5, abs=1e-5)


def test_delay_multichannel():
    """Test delay with multi-channel waveform."""
    waveform = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]])
    delay = Delay(delay_samples=2, feedback=0.0, mix=1.0, taps=1)
    out = delay(waveform)
    # Channel 0: delayed at position 2 should be 1.0
    # Channel 1: delayed at position 2 should be 0.5
    assert out[0, 2].item() == pytest.approx(1.0, abs=1e-5)
    assert out[1, 2].item() == pytest.approx(0.5, abs=1e-5)


def test_delay_short_waveform():
    """Test delay with waveform shorter than delay time."""
    waveform = torch.tensor([1.0, 2.0])
    delay = Delay(delay_samples=3, feedback=0.5, mix=1.0, taps=1)
    out = delay(waveform)
    # Output should be extended to accommodate delay
    # Original (2) + delay (3) = 5 samples minimum
    assert out.size(0) >= 5
    # With mix=1.0, original region should be zeros, delayed region should have signal
    assert out[0].item() == pytest.approx(0.0, abs=1e-5)
    assert out[1].item() == pytest.approx(0.0, abs=1e-5)
    # Delayed signal should appear at position 3
    assert out[3].item() == pytest.approx(1.0, abs=1e-5)


def test_delay_bpm_synced():
    """Test BPM-synced delay calculation."""
    waveform = torch.randn(2, 44100)  # 1 second at 44.1kHz
    delay = Delay(bpm=120, delay_time="1/8", fs=44100, feedback=0.3, mix=0.2)
    # 120 BPM = 0.5 seconds per beat
    # 1/8 note = 0.25 seconds = 11025 samples at 44.1kHz
    assert delay.delay_samples == 11025
    output = delay(waveform)
    # Output should be extended (original length + delay * taps)
    assert output.shape[0] == waveform.shape[0]  # Same number of channels
    assert output.shape[1] >= waveform.shape[1]  # Extended time dimension


def test_delay_bpm_calculation():
    """Test BPM delay calculation for different time divisions."""
    bpm = 120
    fs = 44100
    # 120 BPM = 0.5 seconds per beat

    # 1/4 note = 0.5 seconds = 22050 samples
    delay = Delay(bpm=bpm, delay_time="1/4", fs=fs)
    assert delay.delay_samples == 22050

    # 1/8 note = 0.25 seconds = 11025 samples
    delay = Delay(bpm=bpm, delay_time="1/8", fs=fs)
    assert delay.delay_samples == 11025

    # 1/16 note = 0.125 seconds = 5512 samples
    delay = Delay(bpm=bpm, delay_time="1/16", fs=fs)
    assert delay.delay_samples == 5512


def test_delay_musical_divisions():
    """Test different musical time divisions."""
    waveform = torch.randn(1, 44100)
    fs = 44100
    bpm = 120

    # Test various divisions
    for div in ["1/4", "1/8", "1/16", "1/8d", "1/4d", "1/8t"]:
        delay = Delay(bpm=bpm, delay_time=div, fs=fs, feedback=0.3, mix=0.2)
        output = delay(waveform)
        assert output.shape[0] == waveform.shape[0]  # Same number of channels
        assert output.shape[1] >= waveform.shape[1]  # Extended time dimension
        assert delay.delay_samples > 0


def test_delay_pingpong():
    """Test ping-pong delay mode alternates between channels."""
    # Create impulse on left channel only
    waveform = torch.zeros(2, 10)
    waveform[0, 0] = 1.0  # Impulse on left channel

    delay = Delay(delay_samples=2, feedback=0.0, mix=1.0, taps=2, strategy=PingPongDelayStrategy())
    out = delay(waveform)

    # Tap 1 (odd): Left -> Right, should appear on right channel at position 2
    # Tap 2 (even): Right -> Left, but right has no signal, so nothing
    assert out[1, 2].item() == pytest.approx(1.0, abs=1e-5)  # Left -> Right
    assert out[0, 4].item() == pytest.approx(0.0, abs=1e-5)  # Right -> Left (no signal)


def test_delay_multiple_taps():
    """Test delay with multiple taps."""
    waveform = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    delay = Delay(delay_samples=2, feedback=0.5, mix=1.0, taps=3)
    out = delay(waveform)
    # Should have taps at positions 2, 4, 6
    assert out[2].item() > 0
    assert out[4].item() > 0
    assert out[6].item() > 0


def test_delay_samples():
    """Test delay with direct sample specification."""
    waveform = torch.randn(1, 10000)
    delay = Delay(delay_samples=2205, feedback=0.4, mix=0.3)
    output = delay(waveform)
    assert output.shape[0] == waveform.shape[0]  # Same number of channels
    assert output.shape[1] >= waveform.shape[1]  # Extended time dimension
    assert delay.delay_samples == 2205


@pytest.mark.parametrize("delay_samples", [0, -1])
def test_delay_invalid_delay_samples(delay_samples):
    """Test delay with invalid delay_samples."""
    with pytest.raises(AssertionError):
        Delay(delay_samples=delay_samples, feedback=0.3, mix=0.2)


def test_delay_missing_bpm():
    """Test delay without BPM or delay_samples."""
    with pytest.raises(AssertionError):
        Delay(feedback=0.3, mix=0.2)


def test_delay_invalid_bpm():
    """Test delay with invalid BPM."""
    with pytest.raises(AssertionError):
        Delay(bpm=0, delay_time="1/8", fs=44100)
    with pytest.raises(AssertionError):
        Delay(bpm=-1, delay_time="1/8", fs=44100)


def test_delay_invalid_sample_rate():
    """Test delay with invalid sample rate."""
    with pytest.raises(AssertionError):
        Delay(bpm=120, delay_time="1/8", fs=0)
    with pytest.raises(AssertionError):
        Delay(bpm=120, delay_time="1/8", fs=-1)


@pytest.mark.parametrize("feedback", [-0.1, 1.0, 1.1])
def test_delay_invalid_feedback(feedback):
    """Test delay with invalid feedback values."""
    with pytest.raises(AssertionError):
        Delay(delay_samples=1000, feedback=feedback, mix=0.2)


@pytest.mark.parametrize("mix", [-0.1, 1.1])
def test_delay_invalid_mix(mix):
    """Test delay with invalid mix values."""
    with pytest.raises(AssertionError):
        Delay(delay_samples=1000, feedback=0.3, mix=mix)


def test_delay_invalid_taps():
    """Test delay with invalid taps."""
    with pytest.raises(AssertionError):
        Delay(delay_samples=1000, feedback=0.3, mix=0.2, taps=0)


def test_delay_lazy_fs_inference_with_wave():
    """Test fs auto-inference from Wave pipeline."""
    from torchfx import Wave

    delay = Delay(bpm=120, delay_time="1/8", feedback=0.3, mix=0.2)
    assert delay.fs is None

    wave = Wave(torch.randn(2, 44100), fs=44100)
    _ = wave | delay

    assert delay.fs == 44100
    assert delay.delay_samples == 11025


def test_delay_lazy_fs_inference_error():
    """Test error when using delay without fs."""
    delay = Delay(bpm=120, delay_time="1/8", feedback=0.3, mix=0.2)
    waveform = torch.randn(2, 44100)

    with pytest.raises(AssertionError, match="Sample rate \\(fs\\) is required"):
        delay(waveform)


@pytest.mark.parametrize(
    "time_str,numerator,denominator,expected_fraction",
    [
        ("1/4", 1, 4, 0.25),
        ("1/8", 1, 8, 0.125),
        ("3/16", 3, 16, 0.1875),
    ],
)
def test_musical_time_parser_basic(time_str, numerator, denominator, expected_fraction):
    """Test MusicalTime string parsing."""
    from torchfx.typing import MusicalTime

    mt = MusicalTime.from_string(time_str)
    assert mt.numerator == numerator
    assert mt.denominator == denominator
    assert mt.modifier == ""
    assert mt.fraction() == expected_fraction


@pytest.mark.parametrize(
    "time_str,modifier,expected_fraction",
    [
        ("1/8d", "d", 0.125 * 1.5),
        ("1/4d", "d", 0.25 * 1.5),
        ("1/8t", "t", 0.125 * (1 / 3)),
    ],
)
def test_musical_time_parser_modifiers(time_str, modifier, expected_fraction):
    """Test MusicalTime modifiers."""
    from torchfx.typing import MusicalTime

    mt = MusicalTime.from_string(time_str)
    assert mt.modifier == modifier
    assert mt.fraction() == pytest.approx(expected_fraction)


@pytest.mark.parametrize(
    "time_str,bpm,expected_duration",
    [
        ("1/4", 120, 0.5),
        ("1/8", 120, 0.25),
        ("1/8d", 120, 0.375),
    ],
)
def test_musical_time_duration_calculation(time_str, bpm, expected_duration):
    """Test MusicalTime duration calculation."""
    from torchfx.typing import MusicalTime

    mt = MusicalTime.from_string(time_str)
    assert mt.duration_seconds(bpm) == pytest.approx(expected_duration)


@pytest.mark.parametrize(
    "invalid_string",
    [
        "invalid",
        "1/x",
        "1-4",
        "1/8x",
    ],
)
def test_musical_time_invalid_strings(invalid_string):
    """Test MusicalTime rejects invalid strings."""
    from torchfx.typing import MusicalTime

    with pytest.raises(ValueError, match="Invalid musical time string"):
        MusicalTime.from_string(invalid_string)


def test_musical_time_invalid_modifier():
    """Test MusicalTime rejects invalid modifiers."""
    from torchfx.typing import MusicalTime

    mt = MusicalTime(numerator=1, denominator=4, modifier="x")
    with pytest.raises(ValueError, match="Invalid time duration modifier"):
        mt.fraction()


def test_delay_invalid_time_string():
    """Test delay rejects invalid time strings."""
    with pytest.raises(ValueError, match="Invalid musical time string"):
        Delay(bpm=120, delay_time="invalid", fs=44100)


def test_delay_mono_strategy_explicit():
    """Test MonoDelayStrategy processes channels independently."""
    waveform = torch.zeros(2, 10)
    waveform[0, 0] = 1.0

    delay = Delay(delay_samples=2, feedback=0.0, mix=1.0, taps=1, strategy=MonoDelayStrategy())
    out = delay(waveform)

    assert out[0, 2].item() == pytest.approx(1.0, abs=1e-5)
    assert out[1, 2].item() == pytest.approx(0.0, abs=1e-5)


def test_delay_pingpong_strategy_explicit():
    """Test PingPongDelayStrategy alternates channels."""
    waveform = torch.zeros(2, 10)
    waveform[0, 0] = 1.0

    delay = Delay(delay_samples=2, feedback=0.0, mix=1.0, taps=2, strategy=PingPongDelayStrategy())
    out = delay(waveform)

    assert out[1, 2].item() == pytest.approx(1.0, abs=1e-5)
    assert out[0, 4].item() == pytest.approx(0.0, abs=1e-5)


def test_delay_strategy_extensibility():
    """Test custom DelayStrategy injection."""

    class DoubleStrategy(DelayStrategy):
        def apply_delay(self, waveform, delay_samples, taps, feedback):
            return waveform * 2

    waveform = torch.ones(5)
    delay = Delay(delay_samples=10, feedback=0.5, mix=0.5, strategy=DoubleStrategy())
    out = delay(waveform)

    assert torch.allclose(out[:5], torch.ones(5) * 1.5, atol=1e-5)


def test_delay_pingpong_fallback_mono():
    """Test PingPongDelayStrategy falls back to mono for non-stereo."""
    waveform = torch.zeros(10)
    waveform[0] = 1.0

    delay = Delay(delay_samples=2, feedback=0.0, mix=1.0, taps=1, strategy=PingPongDelayStrategy())
    out = delay(waveform)

    assert out[2].item() == pytest.approx(1.0, abs=1e-5)


def test_delay_batched_audio():
    """Test delay with batched 3D input."""
    waveform = torch.randn(4, 2, 1000)
    delay = Delay(delay_samples=100, feedback=0.5, mix=0.3, taps=2)
    out = delay(waveform)

    assert out.shape == (4, 2, 1200)


def test_delay_batched_audio_with_strategy():
    """Test batched audio with PingPongDelayStrategy."""
    waveform = torch.zeros(2, 2, 20)
    waveform[0, 0, 0] = 1.0

    delay = Delay(delay_samples=3, feedback=0.0, mix=1.0, taps=1, strategy=PingPongDelayStrategy())
    out = delay(waveform)

    assert out[0, 1, 3].item() == pytest.approx(1.0, abs=1e-5)
