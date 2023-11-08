# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Catalyst's debug module contains functions useful for user program debugging, such as
runtime printing.
"""

import builtins

import jax

from catalyst.jax_primitives import print_p
from catalyst.utils.contexts import EvaluationContext


# pylint: disable=redefined-builtin
def print(x, memref=False):
    """A :func:`qjit` compatible print function for printing values at runtime.

    Enables printing of numeric values at runtime. Can also print objects or strings as constants.

    Args:
        x (jax.Array, Any): A single jax array whose numeric values are printed at runtime, or any
            object whose string representation will be treated as a constant and printed at runtime.
        memref (Optional[bool]): When set to ``True``, additional information about how the array is
            stored in memory is printed, via the so-called "memref" descriptor. This includes the
            base memory address of the data buffer, as well as the rank of the array, the size of
            each dimension, and the strides between elements.

    **Example**

    .. code-block:: python

        @qjit
        def func(x: float):
            debug.print(x, memref=True)
            debug.print("exit")

    >>> func(jnp.array(0.43))
    Unranked Memref base@ = 0x5629ff2b6680 rank = 0 offset = 0 sizes = [] strides = [] data =
    [0.43]
    exit

    Outside a :func:`qjit` compiled function the operation falls back to the Python print statement.

    .. note::

        Python f-strings will not work as expected since they will be treated as Python objects.
        This means that array values embeded in them will have their compile-time representation
        printed, instead of actual data.
    """
    if EvaluationContext.is_tracing():
        if isinstance(x, jax.core.Tracer):
            print_p.bind(x, memref=memref)
        else:
            print_p.bind(string=str(x))
    else:
        # Dispatch to Python print outside a qjit context.
        builtins.print(x)
