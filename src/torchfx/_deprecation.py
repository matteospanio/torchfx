"""Deprecation utilities for TorchFX.

This module provides decorators and utilities for marking APIs as deprecated and
providing migration guidance to users.

"""

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(
    version: str,
    reason: str,
    alternative: str | None = None,
    removal_version: str | None = None,
) -> Callable[[F], F]:
    """Mark a function, method, or class as deprecated.

    Parameters
    ----------
    version : str
        The version in which the API was deprecated (e.g., "0.3.0").
    reason : str
        A brief explanation of why the API is deprecated.
    alternative : str, optional
        The recommended alternative to use instead.
    removal_version : str, optional
        The version in which the API will be removed (e.g., "1.0.0").

    Returns
    -------
    Callable
        A decorator that marks the function as deprecated.

    Examples
    --------
    >>> @deprecated(
    ...     version="0.3.0",
    ...     reason="Use the new API instead",
    ...     alternative="new_function()",
    ...     removal_version="1.0.0"
    ... )
    ... def old_function():
    ...     pass

    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            message = f"{func.__name__} is deprecated since version {version}. {reason}"

            if alternative:
                message += f" Use {alternative} instead."

            if removal_version:
                message += f" It will be removed in version {removal_version}."

            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        # Mark the function as deprecated for documentation purposes
        wrapper.__deprecated__ = True  # type: ignore
        wrapper.__deprecated_info__ = {  # type: ignore
            "version": version,
            "reason": reason,
            "alternative": alternative,
            "removal_version": removal_version,
        }

        return wrapper  # type: ignore

    return decorator


def deprecated_parameter(
    param_name: str,
    version: str,
    reason: str,
    alternative: str | None = None,
    removal_version: str | None = None,
) -> Callable[[F], F]:
    """Mark a function parameter as deprecated.

    Parameters
    ----------
    param_name : str
        The name of the deprecated parameter.
    version : str
        The version in which the parameter was deprecated.
    reason : str
        A brief explanation of why the parameter is deprecated.
    alternative : str, optional
        The recommended alternative parameter to use instead.
    removal_version : str, optional
        The version in which the parameter will be removed.

    Returns
    -------
    Callable
        A decorator that issues a warning when the deprecated parameter is used.

    Examples
    --------
    >>> @deprecated_parameter(
    ...     param_name="old_param",
    ...     version="0.3.0",
    ...     reason="Parameter renamed for clarity",
    ...     alternative="new_param",
    ...     removal_version="1.0.0"
    ... )
    ... def some_function(old_param=None, new_param=None):
    ...     pass

    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if param_name in kwargs:
                message = (
                    f"Parameter '{param_name}' of {func.__name__} is deprecated "
                    f"since version {version}. {reason}"
                )

                if alternative:
                    message += f" Use '{alternative}' instead."

                if removal_version:
                    message += f" It will be removed in version {removal_version}."

                warnings.warn(message, DeprecationWarning, stacklevel=2)

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


class DeprecatedAlias:
    """Create a deprecated alias for a class or function.

    This is useful when renaming classes or functions while maintaining
    backward compatibility.

    Parameters
    ----------
    target : type or callable
        The new class or function to alias.
    version : str
        The version in which the alias was deprecated.
    removal_version : str, optional
        The version in which the alias will be removed.

    Examples
    --------
    >>> class NewClassName:
    ...     pass
    >>> OldClassName = DeprecatedAlias(
    ...     NewClassName,
    ...     version="0.3.0",
    ...     removal_version="1.0.0"
    ... )

    """

    def __init__(
        self,
        target: Any,
        version: str,
        removal_version: str | None = None,
    ) -> None:
        self.target = target
        self.version = version
        self.removal_version = removal_version
        self._warned = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self._warned:
            message = (
                f"{self.__class__.__name__} is a deprecated alias since version "
                f"{self.version}. Use {self.target.__name__} instead."
            )
            if self.removal_version:
                message += f" It will be removed in version {self.removal_version}."

            warnings.warn(message, DeprecationWarning, stacklevel=2)
            self._warned = True

        return self.target(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)
