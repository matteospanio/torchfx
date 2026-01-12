"""Tests for deprecation utilities."""

import warnings

import pytest

from torchfx._deprecation import DeprecatedAlias, deprecated, deprecated_parameter


class TestDeprecatedDecorator:
    """Test suite for @deprecated decorator."""

    def test_deprecated_function_warns(self):
        """Test that deprecated function issues a warning."""

        @deprecated(
            version="0.3.0",
            reason="This function is obsolete",
            alternative="new_function()",
            removal_version="1.0.0",
        )
        def old_function():
            return 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_function()

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "old_function is deprecated since version 0.3.0" in str(w[0].message)
            assert "This function is obsolete" in str(w[0].message)
            assert "Use new_function() instead" in str(w[0].message)
            assert "It will be removed in version 1.0.0" in str(w[0].message)
            assert result == 42

    def test_deprecated_function_without_alternative(self):
        """Test deprecated function without alternative suggestion."""

        @deprecated(
            version="0.3.0",
            reason="No longer supported",
        )
        def old_function():
            return "test"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_function()

            assert len(w) == 1
            assert "old_function is deprecated since version 0.3.0" in str(w[0].message)
            assert "No longer supported" in str(w[0].message)
            assert "Use" not in str(w[0].message)
            assert result == "test"

    def test_deprecated_method(self):
        """Test that deprecated method works correctly."""

        class TestClass:
            @deprecated(
                version="0.3.0",
                reason="Method renamed",
                alternative="new_method()",
            )
            def old_method(self, x):
                return x * 2

        obj = TestClass()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = obj.old_method(5)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "old_method is deprecated" in str(w[0].message)
            assert result == 10

    def test_deprecated_metadata(self):
        """Test that deprecated decorator adds metadata."""

        @deprecated(
            version="0.3.0",
            reason="Test",
            alternative="new_func()",
            removal_version="1.0.0",
        )
        def test_func():
            pass

        assert hasattr(test_func, "__deprecated__")
        assert test_func.__deprecated__ is True
        assert test_func.__deprecated_info__["version"] == "0.3.0"
        assert test_func.__deprecated_info__["reason"] == "Test"
        assert test_func.__deprecated_info__["alternative"] == "new_func()"
        assert test_func.__deprecated_info__["removal_version"] == "1.0.0"


class TestDeprecatedParameter:
    """Test suite for @deprecated_parameter decorator."""

    def test_deprecated_parameter_warns_when_used(self):
        """Test that using deprecated parameter issues warning."""

        @deprecated_parameter(
            param_name="old_param",
            version="0.3.0",
            reason="Parameter renamed",
            alternative="new_param",
            removal_version="1.0.0",
        )
        def test_function(old_param=None, new_param=None):
            return old_param or new_param

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_function(old_param=42)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Parameter 'old_param' of test_function is deprecated" in str(w[0].message)
            assert "since version 0.3.0" in str(w[0].message)
            assert "Use 'new_param' instead" in str(w[0].message)
            assert result == 42

    def test_deprecated_parameter_no_warn_when_not_used(self):
        """Test that not using deprecated parameter doesn't warn."""

        @deprecated_parameter(
            param_name="old_param",
            version="0.3.0",
            reason="Parameter renamed",
        )
        def test_function(old_param=None, new_param=None):
            return new_param or "default"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_function(new_param="value")

            assert len(w) == 0
            assert result == "value"


class TestDeprecatedAlias:
    """Test suite for DeprecatedAlias."""

    def test_deprecated_alias_warns(self):
        """Test that using deprecated alias issues warning."""

        class NewClass:
            def __init__(self, value):
                self.value = value

        OldClass = DeprecatedAlias(
            NewClass,
            version="0.3.0",
            removal_version="1.0.0",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = OldClass(42)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated alias since version 0.3.0" in str(w[0].message)
            assert "Use NewClass instead" in str(w[0].message)
            assert obj.value == 42

    def test_deprecated_alias_warns_once(self):
        """Test that deprecated alias only warns once per session."""

        class NewClass:
            pass

        OldClass = DeprecatedAlias(NewClass, version="0.3.0")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OldClass()
            OldClass()
            OldClass()

            # Should only warn once
            assert len(w) == 1

    def test_deprecated_alias_forwards_attributes(self):
        """Test that deprecated alias forwards attributes correctly."""

        class NewClass:
            class_attr = "test"

            def __init__(self):
                self.instance_attr = "value"

        OldClass = DeprecatedAlias(NewClass, version="0.3.0")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # Test class attribute access
            assert OldClass.class_attr == "test"
