"""Tests for TorchFX logging module."""

import io
import logging
import time

import pytest

from torchfx.logging import (
    DEFAULT_DATE_FORMAT,
    DEFAULT_FORMAT,
    LogPerformance,
    disable_logging,
    enable_debug_logging,
    enable_logging,
    get_logger,
    log_performance,
)


class TestLoggerConfiguration:
    """Tests for logger configuration functions."""

    def test_get_logger_root(self):
        """get_logger() should return torchfx root logger."""
        logger = get_logger()
        assert logger.name == "torchfx"

    def test_get_logger_child(self):
        """get_logger(name) should return child logger."""
        logger = get_logger("filter.iir")
        assert logger.name == "torchfx.filter.iir"

    def test_enable_logging_default_level(self):
        """enable_logging() should set INFO level by default."""
        enable_logging()
        logger = get_logger()
        assert logger.level == logging.INFO
        disable_logging()

    def test_enable_logging_custom_level(self):
        """enable_logging() should respect custom level."""
        enable_logging(level="WARNING")
        logger = get_logger()
        assert logger.level == logging.WARNING
        disable_logging()

    def test_enable_debug_logging(self):
        """enable_debug_logging() should set DEBUG level."""
        enable_debug_logging()
        logger = get_logger()
        assert logger.level == logging.DEBUG
        disable_logging()

    def test_disable_logging(self):
        """disable_logging() should remove handlers and reset level."""
        enable_logging()
        disable_logging()
        logger = get_logger()
        # Should only have NullHandler
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.NullHandler)

    def test_enable_logging_custom_stream(self):
        """enable_logging() should write to custom stream."""
        stream = io.StringIO()
        enable_logging(level="INFO", stream=stream)
        logger = get_logger()
        logger.info("Test message")
        output = stream.getvalue()
        assert "Test message" in output
        disable_logging()

    def test_enable_logging_custom_format(self):
        """enable_logging() should use custom format."""
        stream = io.StringIO()
        enable_logging(
            level="INFO",
            format_string="%(levelname)s: %(message)s",
            stream=stream,
        )
        logger = get_logger()
        logger.info("Test")
        output = stream.getvalue()
        assert "INFO: Test" in output
        disable_logging()

    def test_null_handler_by_default(self):
        """TorchFX logger should have NullHandler by default."""
        disable_logging()  # Reset to default state
        logger = logging.getLogger("torchfx")
        has_null = any(isinstance(h, logging.NullHandler) for h in logger.handlers)
        assert has_null

    def test_multiple_enable_calls_no_duplicate_handlers(self):
        """Multiple enable_logging() calls should not add duplicate handlers."""
        enable_logging()
        enable_logging()
        enable_logging()
        logger = get_logger()
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        ]
        assert len(stream_handlers) == 1
        disable_logging()

    def test_default_format_constants(self):
        """Default format constants should be defined."""
        assert DEFAULT_FORMAT is not None
        assert DEFAULT_DATE_FORMAT is not None
        assert "%(message)s" in DEFAULT_FORMAT


class TestPerformanceLogging:
    """Tests for performance logging utilities."""

    def test_log_performance_context_manager(self):
        """log_performance should log execution time."""
        stream = io.StringIO()
        enable_logging(level="INFO", stream=stream)

        with log_performance("test_operation"):
            pass  # Simulate work

        output = stream.getvalue()
        assert "test_operation completed in" in output
        disable_logging()

    def test_log_performance_returns_timing_info(self):
        """log_performance should populate timing dict."""
        enable_logging(level="INFO")

        with log_performance("test") as timing:
            time.sleep(0.01)  # 10ms

        assert "elapsed_seconds" in timing
        assert timing["elapsed_seconds"] >= 0.01
        assert timing["operation_name"] == "test"
        disable_logging()

    def test_log_performance_timing_accuracy(self):
        """log_performance should measure time accurately."""
        enable_logging(level="INFO")

        with log_performance("timing_test") as timing:
            time.sleep(0.05)  # 50ms

        # Should be at least 50ms but less than 100ms
        assert 0.05 <= timing["elapsed_seconds"] < 0.1
        disable_logging()

    def test_log_performance_decorator(self):
        """LogPerformance decorator should log function execution."""
        stream = io.StringIO()
        enable_logging(level="INFO", stream=stream)

        @LogPerformance("decorated_func")
        def test_func():
            return 42

        result = test_func()
        assert result == 42
        output = stream.getvalue()
        assert "decorated_func completed in" in output
        disable_logging()

    def test_log_performance_decorator_auto_name(self):
        """LogPerformance should use function name if not specified."""
        stream = io.StringIO()
        enable_logging(level="INFO", stream=stream)

        @LogPerformance()
        def my_function():
            pass

        my_function()
        output = stream.getvalue()
        assert "my_function completed in" in output
        disable_logging()

    def test_log_performance_decorator_preserves_return_value(self):
        """LogPerformance should preserve function return value."""
        enable_logging(level="INFO")

        @LogPerformance()
        def return_value():
            return {"key": "value", "number": 42}

        result = return_value()
        assert result == {"key": "value", "number": 42}
        disable_logging()

    def test_log_performance_decorator_preserves_args(self):
        """LogPerformance should preserve function arguments."""
        enable_logging(level="INFO")

        @LogPerformance()
        def add(a, b, c=0):
            return a + b + c

        assert add(1, 2) == 3
        assert add(1, 2, c=3) == 6
        disable_logging()

    def test_log_performance_custom_level(self):
        """log_performance should respect custom log level."""
        stream = io.StringIO()
        enable_logging(level="WARNING", stream=stream)

        # INFO level message should not appear
        with log_performance("test", level=logging.INFO):
            pass

        output = stream.getvalue()
        assert output == ""  # Nothing logged at INFO when level is WARNING

        # WARNING level message should appear
        with log_performance("test", level=logging.WARNING):
            pass

        output = stream.getvalue()
        assert "test completed in" in output
        disable_logging()

    def test_log_performance_custom_logger(self):
        """log_performance should use custom logger when provided."""
        stream = io.StringIO()
        custom_logger = logging.getLogger("custom_test")
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        custom_logger.addHandler(handler)
        custom_logger.setLevel(logging.INFO)

        with log_performance("operation", logger=custom_logger):
            pass

        output = stream.getvalue()
        assert "custom_test:" in output
        assert "operation completed in" in output

        # Cleanup
        custom_logger.removeHandler(handler)


class TestLoggerHierarchy:
    """Tests for logger hierarchy behavior."""

    def test_child_loggers_inherit_level(self):
        """Child loggers should inherit parent level."""
        enable_logging(level="DEBUG")
        parent = get_logger()
        child = get_logger("wave")

        # Child should be effective at parent's level
        assert child.getEffectiveLevel() == logging.DEBUG
        disable_logging()

    def test_child_logger_messages_propagate(self):
        """Child logger messages should propagate to parent handler."""
        stream = io.StringIO()
        enable_logging(level="INFO", stream=stream)

        child = get_logger("wave")
        child.info("Child message")

        output = stream.getvalue()
        assert "Child message" in output
        assert "torchfx.wave" in output
        disable_logging()

    def test_performance_logger_hierarchy(self):
        """Performance logger should be a child of torchfx."""
        perf_logger = logging.getLogger("torchfx.performance")
        assert perf_logger.parent.name == "torchfx"  # type: ignore[union-attr]


class TestImports:
    """Test that all exports are accessible."""

    def test_import_from_logging(self):
        """All exports should be importable from torchfx.logging."""
        from torchfx.logging import (
            DEFAULT_DATE_FORMAT,
            DEFAULT_FORMAT,
            LogPerformance,
            disable_logging,
            enable_debug_logging,
            enable_logging,
            get_logger,
            log_performance,
        )

        assert enable_debug_logging is not None
        assert enable_logging is not None
        assert disable_logging is not None
        assert get_logger is not None
        assert log_performance is not None
        assert LogPerformance is not None
        assert DEFAULT_FORMAT is not None
        assert DEFAULT_DATE_FORMAT is not None

    def test_import_via_torchfx(self):
        """Logging should be accessible via torchfx.logging."""
        import torchfx

        assert hasattr(torchfx, "logging")
        assert hasattr(torchfx.logging, "enable_debug_logging")
        assert hasattr(torchfx.logging, "log_performance")
        assert hasattr(torchfx.logging, "LogPerformance")
        assert hasattr(torchfx.logging, "get_logger")
