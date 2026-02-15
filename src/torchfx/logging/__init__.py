"""Logging infrastructure for TorchFX.

This module provides structured logging capabilities for TorchFX with support
for debug logging, performance profiling, and custom handlers.

The library follows Python logging best practices:

- NullHandler is attached by default (opt-in logging)
- Users can enable logging via convenience functions
- Standard logging levels: DEBUG, INFO, WARNING, ERROR

Configuration Functions
-----------------------
enable_debug_logging
    Enable DEBUG level logging to stderr.
enable_logging
    Enable logging at a specified level.
disable_logging
    Disable all TorchFX logging.
get_logger
    Get a logger for a specific TorchFX module.

Performance Logging
-------------------
log_performance
    Context manager for timing code blocks.
LogPerformance
    Decorator for timing function execution.

Examples
--------
Enable debug logging for the entire library:

>>> import torchfx
>>> torchfx.logging.enable_debug_logging()

Enable INFO level logging:

>>> import torchfx
>>> torchfx.logging.enable_logging(level="INFO")

Profile a filter chain:

>>> from torchfx.logging import log_performance
>>> with log_performance("filter_chain"):
...     result = wave | filter1 | filter2
# Logs: "filter_chain completed in 0.045s"

Use the performance decorator:

>>> from torchfx.logging import LogPerformance
>>> @LogPerformance("my_function")
... def process_audio(wave):
...     return wave | some_filter

Standard Python logging configuration also works:

>>> import logging
>>> logging.getLogger("torchfx").setLevel(logging.DEBUG)
>>> logging.getLogger("torchfx").addHandler(logging.StreamHandler())

"""

import logging

from torchfx.logging.config import (
    DEFAULT_DATE_FORMAT,
    DEFAULT_FORMAT,
    disable_logging,
    enable_debug_logging,
    enable_logging,
    get_logger,
)
from torchfx.logging.performance import (
    LogPerformance,
    log_performance,
)

# Library root logger - NullHandler by default per Python guidelines
# https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
_root_logger = logging.getLogger("torchfx")
_root_logger.addHandler(logging.NullHandler())

__all__ = [
    # Configuration
    "enable_debug_logging",
    "enable_logging",
    "disable_logging",
    "get_logger",
    # Performance
    "log_performance",
    "LogPerformance",
    # Constants
    "DEFAULT_FORMAT",
    "DEFAULT_DATE_FORMAT",
]
