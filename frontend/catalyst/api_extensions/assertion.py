"""
This module provides the catalyst.assert function for runtime assertions
in Catalyst JIT-compiled functions.
"""

from catalyst.jax_primitives import assert_p


def catalyst_assert(condition: bool, error: str):
    """
    Asserts at runtime that the given condition holds, raising an error with the
    provided message if it does not.

    Args:
        condition (bool): The condition to assert.
        message (str): The error message to raise if the condition is False.

    Raises:
        RuntimeError: $message.
    """

    return assert_p.bind(condition, error=error)
