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
import sys
from functools import partial

import jax

from catalyst.api_extensions.callbacks import debug_callback
from catalyst.jax_primitives import print_p
from catalyst.tracing.contexts import EvaluationContext


# pylint: disable=redefined-builtin
def print(fmt, *args, **kwargs):
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
    sys.stdout.write(fmt.format(*args, **kwargs))
    sys.stdout.write("\n")


def _print_callback(*args, **kwargs):
    builtins.print(*args, **kwargs)


# pylint: disable=redefined-builtin
def print_memref(x):
    """A :func:`qjit` compatible print function for printing values at runtime.

    Enables printing of numeric values at runtime.

    Args:
        x (jax.Array, Any): A single jax array whose numeric values are printed at runtime

    **Example**

    .. code-block:: python

        @qjit
        def func(x: float):
            debug.print_memref(x)

    >>> func(jnp.array(0.43))
    Unranked Memref base@ = 0x5629ff2b6680 rank = 0 offset = 0 sizes = [] strides = [] data =
    [0.43]

    Outside a :func:`qjit` compiled function the operation falls back to the Python print statement.

    .. note::

        Python f-strings will not work as expected since they will be treated as Python objects.
        This means that array values embeded in them will have their compile-time representation
        printed, instead of actual data.
    """
    if EvaluationContext.is_tracing():
        if not isinstance(x, jax.core.Tracer):
            raise TypeError("Arguments to print_memref must be of type jax.core.Tracer")
        print_p.bind(x, memref=True)
    else:
        # Dispatch to Python print outside a qjit context.
        builtins.print(x)
