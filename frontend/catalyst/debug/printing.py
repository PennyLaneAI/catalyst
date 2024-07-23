# Copyright 2023-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module enables runtime printing of program values."""

import builtins
from functools import partial

import jax

from catalyst.debug.callback import callback as debug_callback
from catalyst.jax_primitives import print_p
from catalyst.tracing.contexts import EvaluationContext


# pylint: disable=redefined-builtin
def print(fmt, *args, **kwargs):
    """A :func:`~.qjit` compatible print function for printing values at runtime.

    This print function allows printing of values at runtime, unlike usage of standard Python
    ``print`` which will print values at program capture/compile time.

    ``debug.print`` is a minimal wrapper around :func:`~.debug.callback` which calls Python's
    ``builtins.print`` function, and thus will use the same formatting styles as Python's built-in
    print function.

    Args:
        fmt (str): The string to be printed. Note that this may also be a
            format string used to format input arguments (for example
            ``cost={x}``), similar to those permitted by ``str.format``. See
            the Python docs on
            `string formatting <https://docs.python.org/3/library/stdtypes.html#str.format>`__ and
            `format string syntax <https://docs.python.org/3/library/string.html#formatstrings>`__.
        **args: Arguments to be passed to the format string.
        **kwargs: Keyword arguments to be passed to the format string.

    .. seealso:: :func:`~.print_memref`, :func:`~.debug.callback`.

    **Example**

    >>> @qjit
    ... def f(a, b, c):
    ...     debug.print("c={c} b={b} a={a}", a=a, b=b, c=c)
    >>> f(1, 2, 3)
    c=3 b=2 a=1

    In addition to passing keyword arguments to the format string, we can also pass arguments
    positionally:

    >>> @qjit
    ... @grad
    ... def f(x, y):
    ...     debug.print("Value of x = {0:.2f}", x)
    ...     return x * jnp.sin(y)
    >>> f(0.543, 0.23)
    Value of x = 0.54
    Array(0.22797752, dtype=float64)

    Note that during differentiation, printing will only be executed
    during the forward pass.

    .. note::

        Using Python f-strings as the ``fmt`` string will not work as expected since they will be
        treated as Python objects.

        This means that array values embedded in them will have their compile-time representation
        printed, instead of actual data.
    """

    if isinstance(fmt, str):
        debug_callback(partial(_format_print_callback, fmt))(*args, **kwargs)
        return

    debug_callback(_print_callback)(fmt, *args, **kwargs)


def _format_print_callback(fmt: str, *args, **kwargs):
    """
    This function has been modified from its original form in the JAX project at
    https://github.com/google/jax/blob/5eeebf2c1829bf3c66c947b6e12464a779434e29/jax/_src/debugging.py#L269
    version released under the Apache License, Version 2.0, with the following copyright notice:
    Copyright 2022 The Jax Authors.
    """
    builtins.print(fmt.format(*args, **kwargs))


def _print_callback(*args, **kwargs):
    """Print without formatting"""
    builtins.print(*args, **kwargs)


def print_memref(x):
    """A :func:`qjit` compatible print function for printing numeric values at runtime with memref
    information.

    Enables printing of numeric values at runtime and the value's metadata.

    Tensors in the Catalyst runtime are represented as memref descriptor structs.
    For more information about memref descriptors see the
    `MLIR documentation <https://mlir.llvm.org/docs/Dialects/MemRef/>`__.
    This function will print the base memory address of the data buffer, as well as the rank of
    the array, the size of each dimension, and the strides between elements.

    Args:
        x (jax.Array, Any): A single jax array whose numeric values are printed at runtime.

    .. seealso:: :func:`~.debug.print`

    **Example**

    .. code-block:: python

        @qjit
        def func(x: float):
            debug.print_memref(x)

    >>> func(jnp.array(0.43))
    Unranked Memref base@ = 0x5629ff2b6680 rank = 0 offset = 0 sizes = [] strides = [] data =
    [0.43]

    Outside a :func:`qjit` compiled function the operation falls back to the Python print statement.
    """
    if EvaluationContext.is_tracing():
        if not isinstance(x, jax.core.Tracer):
            raise TypeError("Arguments to print_memref must be of type jax.core.Tracer")
        print_p.bind(x, memref=True)
    else:
        # Dispatch to Python print outside a qjit context.
        builtins.print(x)
