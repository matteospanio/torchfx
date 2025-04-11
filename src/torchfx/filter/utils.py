"""Utility functions for filter design."""

from torchfx.typing import FilterOrderScale


def compute_order(o: int, scale: FilterOrderScale) -> int:
    """Compute the correct filter's order given a specified scale.

    The filter order is usually specified as an abstract number,
    but it occurs also as db quantity, therefore this function
    is used to prepare the given order to be inputted to a filter
    class.

    Parameters
    ----------
    o : int
        The input order referred to the input scale
    scale : FilterOrderScale
        The input scale used to quantify the order

    Returns
    -------
    int
        The correct order value in linear scale

    Example
    -------
    >>> o = 24 # db
    >>> compute_order(o, "db")
    4
    """
    match scale:
        case "db":
            return o // 6
        case "linear":
            return o
