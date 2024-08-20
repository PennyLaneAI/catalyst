"""
This module provides the catalyst.assert function for runtime assertions
in Catalyst JIT-compiled functions.
"""

from catalyst.jax_primitives import assert_p


def debug_assert(condition: bool, error: str):
    """
    Asserts at runtime that the given condition holds, raising an error with the
    provided message if it does not.

    Note that Python-based exceptions (via ``raise``) and assertions (via ``assert``)
    will always be evaluated at program capture time, before certain runtime information
    may be available.

    Use ``debug_assert`` to instead raise assertions at runtime, including
    assertions that depend on values of dynamic variables.

    Args:
        condition (bool): The condition to assert.
        message (str): The error message to raise if the condition is False.

    Raises:
        RuntimeError: raised if the condition evaluates to ``False``

    **Example**

    .. code-block:: python

        @qjit
        def f(x):
            debug_assert(x < 5, "x was greater than 5")
            return x * 8

    >>> f(4)
    Array(32, dtype=int64)
    >>> f(6)
    RuntimeError: x was greater than 5

    Assertions can be disabled globally for a qjit-compiled function
    via the ``disable_assertions`` keyword argument:

    .. code-block:: python

        @qjit(disable_assertions=True)
        def g(x):
            debug_assert(x < 5, "x was greater than 5")
            return x * 8

    >>> g(6)
    Array(48, dtype=int64)
    """

    return assert_p.bind(condition, error=error)
