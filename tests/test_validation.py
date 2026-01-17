"""Tests for TorchFX validation module."""

import pytest
import torch

from torchfx.validation import (
    COMMON_SAMPLE_RATES,
    AudioProcessingError,
    CoefficientComputationError,
    FilterInstabilityError,
    InvalidParameterError,
    InvalidRangeError,
    InvalidSampleRateError,
    InvalidShapeError,
    InvalidTypeError,
    TorchFXError,
    validate_audio_tensor,
    validate_cutoff_frequency,
    validate_filter_order,
    validate_in_set,
    validate_positive,
    validate_q_factor,
    validate_range,
    validate_sample_rate,
    validate_tensor_ndim,
    validate_type,
)


# =============================================================================
# Exception Hierarchy Tests
# =============================================================================


class TestExceptionHierarchy:
    """Test the custom exception hierarchy."""

    def test_all_exceptions_inherit_from_torchfx_error(self):
        """All custom exceptions should inherit from TorchFXError."""
        assert issubclass(InvalidParameterError, TorchFXError)
        assert issubclass(InvalidSampleRateError, TorchFXError)
        assert issubclass(InvalidRangeError, TorchFXError)
        assert issubclass(InvalidShapeError, TorchFXError)
        assert issubclass(InvalidTypeError, TorchFXError)
        assert issubclass(AudioProcessingError, TorchFXError)
        assert issubclass(CoefficientComputationError, TorchFXError)
        assert issubclass(FilterInstabilityError, TorchFXError)

    def test_invalid_parameter_errors_inherit_from_invalid_parameter(self):
        """Specific parameter errors should inherit from InvalidParameterError."""
        assert issubclass(InvalidSampleRateError, InvalidParameterError)
        assert issubclass(InvalidRangeError, InvalidParameterError)
        assert issubclass(InvalidShapeError, InvalidParameterError)
        assert issubclass(InvalidTypeError, InvalidParameterError)

    def test_audio_processing_errors_inherit_from_audio_processing(self):
        """Specific processing errors should inherit from AudioProcessingError."""
        assert issubclass(CoefficientComputationError, AudioProcessingError)
        assert issubclass(FilterInstabilityError, AudioProcessingError)

    def test_can_catch_all_torchfx_errors(self):
        """Should be able to catch all errors with TorchFXError."""
        with pytest.raises(TorchFXError):
            raise InvalidSampleRateError(actual_value=-1)

        with pytest.raises(TorchFXError):
            raise InvalidRangeError("cutoff", -1, min_value=0)

        with pytest.raises(TorchFXError):
            raise AudioProcessingError("Test error")

    def test_torchfx_error_message_formatting(self):
        """TorchFXError should format messages with context."""
        err = TorchFXError(
            "Test error",
            parameter_name="test_param",
            actual_value=42,
            suggestion="Try something else",
        )
        msg = str(err)
        assert "Test error" in msg
        assert "test_param" in msg
        assert "42" in msg
        assert "Try something else" in msg

    def test_invalid_parameter_error_message_formatting(self):
        """InvalidParameterError should include expected value."""
        err = InvalidParameterError(
            "Invalid value",
            parameter_name="cutoff",
            actual_value=-100,
            expected="positive float",
        )
        msg = str(err)
        assert "cutoff" in msg
        assert "-100" in msg
        assert "positive float" in msg

    def test_invalid_range_error_shows_bounds(self):
        """InvalidRangeError should show expected bounds."""
        err = InvalidRangeError("decay", 1.5, min_value=0, max_value=1)
        msg = str(err)
        assert "decay" in msg
        assert "1.5" in msg
        assert "[0, 1]" in msg

    def test_invalid_range_error_exclusive_bounds(self):
        """InvalidRangeError should show exclusive bounds correctly."""
        err = InvalidRangeError("decay", 0, min_value=0, max_value=1, min_inclusive=False)
        msg = str(err)
        assert "(0, 1]" in msg

    def test_invalid_shape_error_message(self):
        """InvalidShapeError should show shape information."""
        err = InvalidShapeError(
            "waveform",
            actual_shape=(4, 2, 1000),
            expected_ndim=2,
        )
        msg = str(err)
        assert "waveform" in msg
        assert "(4, 2, 1000)" in msg
        assert "2D tensor" in msg

    def test_invalid_type_error_message(self):
        """InvalidTypeError should show type information."""
        err = InvalidTypeError(
            "gain",
            actual_type=str,
            expected_types=(int, float),
        )
        msg = str(err)
        assert "gain" in msg
        assert "str" in msg
        assert "int" in msg
        assert "float" in msg

    def test_coefficient_computation_error_message(self):
        """CoefficientComputationError should include filter type and reason."""
        err = CoefficientComputationError(
            filter_type="LoButterworth",
            reason="Cutoff exceeds Nyquist",
        )
        msg = str(err)
        assert "LoButterworth" in msg
        assert "Cutoff exceeds Nyquist" in msg

    def test_filter_instability_error_message(self):
        """FilterInstabilityError should include filter type."""
        err = FilterInstabilityError(filter_type="HiChebyshev1")
        msg = str(err)
        assert "HiChebyshev1" in msg
        assert "unstable" in msg


# =============================================================================
# Sample Rate Validation Tests
# =============================================================================


class TestValidateSampleRate:
    """Tests for validate_sample_rate function."""

    def test_valid_sample_rates(self):
        """Common sample rates should pass validation."""
        for fs in COMMON_SAMPLE_RATES:
            validate_sample_rate(fs)  # Should not raise

    def test_none_allowed_when_specified(self):
        """None should be allowed when allow_none=True."""
        validate_sample_rate(None, allow_none=True)  # Should not raise

    def test_none_not_allowed_by_default(self):
        """None should raise by default."""
        with pytest.raises(InvalidSampleRateError):
            validate_sample_rate(None)

    def test_negative_sample_rate(self):
        """Negative sample rates should raise."""
        with pytest.raises(InvalidSampleRateError):
            validate_sample_rate(-44100)

    def test_zero_sample_rate(self):
        """Zero sample rate should raise."""
        with pytest.raises(InvalidSampleRateError):
            validate_sample_rate(0)

    def test_non_integer_sample_rate(self):
        """Non-integer sample rates should raise."""
        with pytest.raises(InvalidSampleRateError):
            validate_sample_rate(44100.5)  # type: ignore

    def test_sample_rate_above_max(self):
        """Sample rates above max should raise."""
        with pytest.raises(InvalidSampleRateError):
            validate_sample_rate(500000)

    def test_custom_bounds(self):
        """Custom bounds should be respected."""
        validate_sample_rate(100, min_rate=100, max_rate=200)
        with pytest.raises(InvalidSampleRateError):
            validate_sample_rate(99, min_rate=100, max_rate=200)


# =============================================================================
# Range Validation Tests
# =============================================================================


class TestValidatePositive:
    """Tests for validate_positive function."""

    def test_positive_values_pass(self):
        """Positive values should pass."""
        validate_positive(1, "param")
        validate_positive(0.001, "param")
        validate_positive(1000000, "param")

    def test_zero_fails_by_default(self):
        """Zero should fail by default."""
        with pytest.raises(InvalidRangeError):
            validate_positive(0, "param")

    def test_zero_passes_when_allowed(self):
        """Zero should pass when allow_zero=True."""
        validate_positive(0, "param", allow_zero=True)

    def test_negative_always_fails(self):
        """Negative values should always fail."""
        with pytest.raises(InvalidRangeError):
            validate_positive(-1, "param")

        with pytest.raises(InvalidRangeError):
            validate_positive(-1, "param", allow_zero=True)

    def test_float_values(self):
        """Float values should work correctly."""
        validate_positive(0.0001, "param")
        with pytest.raises(InvalidRangeError):
            validate_positive(-0.0001, "param")


class TestValidateRange:
    """Tests for validate_range function."""

    def test_value_in_range(self):
        """Values in range should pass."""
        validate_range(0.5, "param", min_value=0, max_value=1)

    def test_value_at_inclusive_bounds(self):
        """Values at inclusive bounds should pass."""
        validate_range(0, "param", min_value=0, max_value=1)
        validate_range(1, "param", min_value=0, max_value=1)

    def test_value_at_exclusive_min_bound_fails(self):
        """Values at exclusive min bound should fail."""
        with pytest.raises(InvalidRangeError):
            validate_range(0, "param", min_value=0, max_value=1, min_inclusive=False)

    def test_value_at_exclusive_max_bound_fails(self):
        """Values at exclusive max bound should fail."""
        with pytest.raises(InvalidRangeError):
            validate_range(1, "param", min_value=0, max_value=1, max_inclusive=False)

    def test_value_below_min(self):
        """Values below minimum should fail."""
        with pytest.raises(InvalidRangeError):
            validate_range(-0.1, "param", min_value=0, max_value=1)

    def test_value_above_max(self):
        """Values above maximum should fail."""
        with pytest.raises(InvalidRangeError):
            validate_range(1.1, "param", min_value=0, max_value=1)

    def test_no_min_bound(self):
        """Should work with only max bound."""
        validate_range(-1000, "param", max_value=0)
        with pytest.raises(InvalidRangeError):
            validate_range(1, "param", max_value=0)

    def test_no_max_bound(self):
        """Should work with only min bound."""
        validate_range(1000, "param", min_value=0)
        with pytest.raises(InvalidRangeError):
            validate_range(-1, "param", min_value=0)

    def test_no_bounds(self):
        """Should pass when no bounds specified."""
        validate_range(1000, "param")
        validate_range(-1000, "param")


class TestValidateInSet:
    """Tests for validate_in_set function."""

    def test_valid_value(self):
        """Values in set should pass."""
        validate_in_set("amplitude", "gain_type", ["amplitude", "db", "power"])

    def test_invalid_value(self):
        """Values not in set should fail."""
        with pytest.raises(InvalidParameterError):
            validate_in_set("invalid", "gain_type", ["amplitude", "db", "power"])

    def test_empty_set(self):
        """Empty set should always fail."""
        with pytest.raises(InvalidParameterError):
            validate_in_set("anything", "param", [])

    def test_tuple_as_valid_values(self):
        """Tuple should work as valid_values."""
        validate_in_set("a", "param", ("a", "b", "c"))


# =============================================================================
# Tensor Shape Validation Tests
# =============================================================================


class TestValidateTensorNdim:
    """Tests for validate_tensor_ndim function."""

    def test_correct_ndim(self):
        """Tensors with correct ndim should pass."""
        t = torch.randn(2, 1000)
        validate_tensor_ndim(t, "waveform", expected_ndim=2)

    def test_incorrect_ndim(self):
        """Tensors with incorrect ndim should fail."""
        t = torch.randn(2, 1000)
        with pytest.raises(InvalidShapeError):
            validate_tensor_ndim(t, "waveform", expected_ndim=3)

    def test_multiple_allowed_ndims(self):
        """Should accept multiple allowed ndims."""
        t1 = torch.randn(1000)
        t2 = torch.randn(2, 1000)
        t3 = torch.randn(4, 2, 1000)

        validate_tensor_ndim(t1, "waveform", expected_ndim=[1, 2, 3])
        validate_tensor_ndim(t2, "waveform", expected_ndim=[1, 2, 3])
        validate_tensor_ndim(t3, "waveform", expected_ndim=[1, 2, 3])

    def test_multiple_ndims_fails_when_not_in_list(self):
        """Should fail when ndim not in allowed list."""
        t = torch.randn(4, 2, 3, 1000)
        with pytest.raises(InvalidShapeError):
            validate_tensor_ndim(t, "waveform", expected_ndim=[1, 2, 3])


class TestValidateAudioTensor:
    """Tests for validate_audio_tensor function."""

    def test_mono_1d(self):
        """1D mono tensors should pass by default."""
        t = torch.randn(1000)
        validate_audio_tensor(t)

    def test_stereo_2d(self):
        """2D stereo tensors should pass."""
        t = torch.randn(2, 1000)
        validate_audio_tensor(t)

    def test_batched_3d(self):
        """3D batched tensors should pass."""
        t = torch.randn(4, 2, 1000)
        validate_audio_tensor(t)

    def test_mono_not_allowed(self):
        """1D tensors should fail when allow_mono=False."""
        t = torch.randn(1000)
        with pytest.raises(InvalidShapeError):
            validate_audio_tensor(t, allow_mono=False)

    def test_4d_tensor_fails(self):
        """4D tensors should fail."""
        t = torch.randn(2, 4, 2, 1000)
        with pytest.raises(InvalidShapeError):
            validate_audio_tensor(t)

    def test_min_channels(self):
        """Should enforce minimum channels."""
        t_mono = torch.randn(1, 1000)
        t_stereo = torch.randn(2, 1000)

        validate_audio_tensor(t_stereo, min_channels=2)

        with pytest.raises(InvalidShapeError):
            validate_audio_tensor(t_mono, min_channels=2)

    def test_max_channels(self):
        """Should enforce maximum channels."""
        t_stereo = torch.randn(2, 1000)
        t_surround = torch.randn(8, 1000)

        validate_audio_tensor(t_stereo, max_channels=2)

        with pytest.raises(InvalidShapeError):
            validate_audio_tensor(t_surround, max_channels=2)

    def test_min_samples(self):
        """Should enforce minimum samples."""
        t_short = torch.randn(2, 10)
        t_long = torch.randn(2, 1000)

        validate_audio_tensor(t_long, min_samples=100)

        with pytest.raises(InvalidShapeError):
            validate_audio_tensor(t_short, min_samples=100)

    def test_1d_mono_channel_validation(self):
        """1D mono tensor should have 1 channel for channel validation."""
        t = torch.randn(1000)
        validate_audio_tensor(t, max_channels=2)  # Should pass (1 channel)
        with pytest.raises(InvalidShapeError):
            validate_audio_tensor(t, min_channels=2)  # Should fail (only 1 channel)

    def test_batched_channel_validation(self):
        """3D batched tensor should validate channels correctly."""
        t = torch.randn(4, 2, 1000)  # batch=4, channels=2, samples=1000
        validate_audio_tensor(t, min_channels=2)
        with pytest.raises(InvalidShapeError):
            validate_audio_tensor(t, min_channels=4)


# =============================================================================
# Type Validation Tests
# =============================================================================


class TestValidateType:
    """Tests for validate_type function."""

    def test_correct_single_type(self):
        """Values with correct type should pass."""
        validate_type(1, "param", int)
        validate_type(1.0, "param", float)
        validate_type("hello", "param", str)

    def test_correct_multiple_types(self):
        """Values matching any of multiple types should pass."""
        validate_type(1, "param", (int, float))
        validate_type(1.0, "param", (int, float))

    def test_incorrect_type(self):
        """Values with incorrect type should fail."""
        with pytest.raises(InvalidTypeError):
            validate_type("hello", "param", (int, float))

    def test_none_type(self):
        """None type validation should work."""
        validate_type(None, "param", type(None))

    def test_bool_is_int_in_python(self):
        """Bool is a subclass of int in Python."""
        # This is expected Python behavior
        validate_type(True, "param", int)


# =============================================================================
# Audio-Specific Validation Tests
# =============================================================================


class TestValidateCutoffFrequency:
    """Tests for validate_cutoff_frequency function."""

    def test_valid_cutoff(self):
        """Valid cutoff frequencies should pass."""
        validate_cutoff_frequency(1000, 44100)
        validate_cutoff_frequency(100, 44100)
        validate_cutoff_frequency(20000, 44100)

    def test_cutoff_at_nyquist(self):
        """Cutoff at Nyquist should fail."""
        with pytest.raises(InvalidRangeError):
            validate_cutoff_frequency(22050, 44100)

    def test_cutoff_above_nyquist(self):
        """Cutoff above Nyquist should fail."""
        with pytest.raises(InvalidRangeError):
            validate_cutoff_frequency(30000, 44100)

    def test_negative_cutoff(self):
        """Negative cutoff should fail."""
        with pytest.raises(InvalidRangeError):
            validate_cutoff_frequency(-1000, 44100)

    def test_zero_cutoff(self):
        """Zero cutoff should fail."""
        with pytest.raises(InvalidRangeError):
            validate_cutoff_frequency(0, 44100)

    def test_cutoff_without_fs(self):
        """Cutoff validation without fs should only check positive."""
        validate_cutoff_frequency(1000, None)
        validate_cutoff_frequency(100000, None)  # No Nyquist check

        with pytest.raises(InvalidRangeError):
            validate_cutoff_frequency(-1000, None)


class TestValidateFilterOrder:
    """Tests for validate_filter_order function."""

    def test_valid_orders(self):
        """Valid orders should pass."""
        validate_filter_order(1)
        validate_filter_order(4)
        validate_filter_order(16)

    def test_zero_order(self):
        """Zero order should fail."""
        with pytest.raises(InvalidRangeError):
            validate_filter_order(0)

    def test_negative_order(self):
        """Negative order should fail."""
        with pytest.raises(InvalidRangeError):
            validate_filter_order(-1)

    def test_non_integer_order(self):
        """Non-integer order should fail."""
        with pytest.raises(InvalidTypeError):
            validate_filter_order(4.5)  # type: ignore

    def test_custom_min_order(self):
        """Custom min_order should be respected."""
        validate_filter_order(2, min_order=2)
        with pytest.raises(InvalidRangeError):
            validate_filter_order(1, min_order=2)

    def test_custom_max_order(self):
        """Custom max_order should be respected."""
        validate_filter_order(8, max_order=10)
        with pytest.raises(InvalidRangeError):
            validate_filter_order(12, max_order=10)


class TestValidateQFactor:
    """Tests for validate_q_factor function."""

    def test_valid_q(self):
        """Valid Q factors should pass."""
        validate_q_factor(0.707)
        validate_q_factor(1.0)
        validate_q_factor(10.0)

    def test_zero_q(self):
        """Zero Q should fail."""
        with pytest.raises(InvalidRangeError):
            validate_q_factor(0)

    def test_negative_q(self):
        """Negative Q should fail."""
        with pytest.raises(InvalidRangeError):
            validate_q_factor(-1)

    def test_very_small_q(self):
        """Very small Q should pass if above min_q."""
        validate_q_factor(0.01)
        with pytest.raises(InvalidRangeError):
            validate_q_factor(0.0001)  # Below default min_q of 0.001

    def test_custom_bounds(self):
        """Custom bounds should be respected."""
        validate_q_factor(0.5, min_q=0.1, max_q=2.0)
        with pytest.raises(InvalidRangeError):
            validate_q_factor(3.0, max_q=2.0)


# =============================================================================
# Integration Tests
# =============================================================================


class TestImports:
    """Test that all exports are accessible."""

    def test_import_from_validation(self):
        """All exports should be importable from torchfx.validation."""
        from torchfx.validation import (
            COMMON_SAMPLE_RATES,
            AudioProcessingError,
            CoefficientComputationError,
            FilterInstabilityError,
            InvalidParameterError,
            InvalidRangeError,
            InvalidSampleRateError,
            InvalidShapeError,
            InvalidTypeError,
            TorchFXError,
            validate_audio_tensor,
            validate_cutoff_frequency,
            validate_filter_order,
            validate_in_set,
            validate_positive,
            validate_q_factor,
            validate_range,
            validate_sample_rate,
            validate_tensor_ndim,
            validate_type,
        )

        # Just verify they're not None
        assert TorchFXError is not None
        assert COMMON_SAMPLE_RATES is not None
        assert validate_sample_rate is not None

    def test_import_via_torchfx(self):
        """Validation should be accessible via torchfx.validation."""
        import torchfx

        assert hasattr(torchfx, "validation")
        assert hasattr(torchfx.validation, "TorchFXError")
        assert hasattr(torchfx.validation, "validate_sample_rate")
