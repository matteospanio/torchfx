"""Logger configuration utilities for TorchFX.

This module provides functions to configure TorchFX logging behavior. By default,
TorchFX attaches a NullHandler to the root logger, meaning no log output is produced
unless explicitly enabled by the user.

"""

from __future__ import annotations

import logging
import sys
from typing import Literal

#: Default format for TorchFX log messages
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

#: Default date format for timestamps
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger for a TorchFX module.

    Parameters
    ----------
    name : str | None, optional
        The module name to get a logger for. If None, returns the root
        TorchFX logger. If provided, returns a child logger under
        "torchfx.<name>".

    Returns
    -------
    logging.Logger
        A logger instance for the specified module.

    Examples
    --------
    Get the root TorchFX logger:

    >>> logger = get_logger()
    >>> logger.name
    'torchfx'

    Get a logger for a specific module:

    >>> logger = get_logger("filter.iir")
    >>> logger.name
    'torchfx.filter.iir'

    """
    if name is None:
        return logging.getLogger("torchfx")
    return logging.getLogger(f"torchfx.{name}")


def enable_logging(
    level: LogLevel = "INFO",
    format_string: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    stream: object | None = None,
) -> None:
    """Enable TorchFX logging at the specified level.

    This function configures the TorchFX root logger with a StreamHandler
    that outputs to the specified stream (stderr by default).

    Parameters
    ----------
    level : {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}, optional
        The logging level to set. Default is "INFO".
    format_string : str, optional
        The format string for log messages. Default includes timestamp,
        logger name, level, and message.
    date_format : str, optional
        The date format for timestamps. Default is "%Y-%m-%d %H:%M:%S".
    stream : file-like object | None, optional
        The stream to write log messages to. Default is sys.stderr.

    Examples
    --------
    Enable INFO level logging:

    >>> import torchfx
    >>> torchfx.logging.enable_logging()

    Enable DEBUG level logging:

    >>> torchfx.logging.enable_logging(level="DEBUG")

    Custom format:

    >>> torchfx.logging.enable_logging(
    ...     level="DEBUG",
    ...     format_string="%(levelname)s: %(message)s"
    ... )

    """
    logger = logging.getLogger("torchfx")
    logger.setLevel(getattr(logging, level))

    # Remove NullHandler if present
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)

    # Check if we already have a StreamHandler
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        for h in logger.handlers
    )

    if not has_stream_handler:
        handler = logging.StreamHandler(stream or sys.stderr)  # type: ignore[arg-type]
        handler.setLevel(getattr(logging, level))
        formatter = logging.Formatter(format_string, datefmt=date_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def enable_debug_logging(
    format_string: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> None:
    """Enable DEBUG level logging for TorchFX.

    This is a convenience function equivalent to calling
    ``enable_logging(level="DEBUG")``.

    Parameters
    ----------
    format_string : str, optional
        The format string for log messages.
    date_format : str, optional
        The date format for timestamps.

    Examples
    --------
    >>> import torchfx
    >>> torchfx.logging.enable_debug_logging()

    """
    enable_logging(level="DEBUG", format_string=format_string, date_format=date_format)


def disable_logging() -> None:
    """Disable TorchFX logging.

    This function removes all handlers from the TorchFX logger and
    re-attaches a NullHandler to suppress output.

    Examples
    --------
    >>> import torchfx
    >>> torchfx.logging.enable_debug_logging()
    >>> # ... do some work ...
    >>> torchfx.logging.disable_logging()  # Suppress further output

    """
    logger = logging.getLogger("torchfx")

    # Remove all handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # Re-attach NullHandler
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.NOTSET)
