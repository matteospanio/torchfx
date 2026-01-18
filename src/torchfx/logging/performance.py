"""Performance logging utilities for TorchFX.

This module provides tools for measuring and logging the execution time of audio
processing operations, enabling performance profiling of filter chains and effects
pipelines.

"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# Performance logger is a child of the main torchfx logger
_perf_logger = logging.getLogger("torchfx.performance")


@contextmanager
def log_performance(
    operation_name: str,
    level: int = logging.INFO,
    logger: logging.Logger | None = None,
) -> Any:
    """Context manager for logging the execution time of a code block.

    Parameters
    ----------
    operation_name : str
        A descriptive name for the operation being timed.
    level : int, optional
        The logging level for the timing message. Default is INFO.
    logger : logging.Logger | None, optional
        The logger to use. If None, uses the torchfx.performance logger.

    Yields
    ------
    dict
        A dictionary that will contain timing information after the block
        completes. Keys: "elapsed_seconds", "operation_name".

    Examples
    --------
    Basic usage:

    >>> from torchfx.logging import log_performance
    >>> with log_performance("filter_chain"):
    ...     result = wave | filter1 | filter2
    # Logs: "filter_chain completed in 0.045s"

    Capture timing information:

    >>> with log_performance("processing") as timing:
    ...     result = wave | effect
    >>> print(f"Took {timing['elapsed_seconds']:.3f}s")

    With custom logger:

    >>> import logging
    >>> my_logger = logging.getLogger("myapp")
    >>> with log_performance("operation", logger=my_logger):
    ...     pass

    """
    log = logger or _perf_logger
    timing_info: dict[str, Any] = {"operation_name": operation_name}

    start_time = time.perf_counter()
    try:
        yield timing_info
    finally:
        elapsed = time.perf_counter() - start_time
        timing_info["elapsed_seconds"] = elapsed
        log.log(level, "%s completed in %.3fs", operation_name, elapsed)


class LogPerformance:
    """Decorator for logging function execution time.

    This decorator wraps a function to automatically log its execution time
    each time it is called.

    Parameters
    ----------
    operation_name : str | None, optional
        A descriptive name for the operation. If None, uses the function name.
    level : int, optional
        The logging level for timing messages. Default is INFO.
    logger : logging.Logger | None, optional
        The logger to use. If None, uses the torchfx.performance logger.

    Examples
    --------
    Basic usage with automatic naming:

    >>> from torchfx.logging import LogPerformance
    >>> @LogPerformance()
    ... def process_audio(wave):
    ...     return wave | filter1 | filter2
    # Each call logs: "process_audio completed in X.XXXs"

    Custom operation name:

    >>> @LogPerformance("audio_processing_pipeline")
    ... def process(wave):
    ...     return wave | filter

    With custom logger:

    >>> import logging
    >>> my_logger = logging.getLogger("myapp")
    >>> @LogPerformance("processing", logger=my_logger)
    ... def process(wave):
    ...     return wave | filter

    """

    def __init__(
        self,
        operation_name: str | None = None,
        level: int = logging.INFO,
        logger: logging.Logger | None = None,
    ) -> None:
        self.operation_name = operation_name
        self.level = level
        self.logger = logger or _perf_logger

    def __call__(self, func: F) -> F:
        """Wrap the function with performance logging."""
        name = self.operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start_time
                self.logger.log(self.level, "%s completed in %.3fs", name, elapsed)

        return wrapper  # type: ignore[return-value]
