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

"""This module enables runtime Python callbacks with no return values."""
from catalyst.api_extensions.callbacks import base_callback


def callback(callback_fn):
    """Execute a Python
    function with no return value and potential side effects from within a qjit-compiled function.

    This makes it an easy entry point for debugging, for example via printing or logging at
    runtime.

    The callback function will be quantum just-in-time compiled alongside the rest of the
    workflow, however it will be executed at runtime by the Python virtual machine.
    This is in contrast to functions which get directly qjit-compiled by Catalyst, which will
    be executed at runtime by machine-native code.

    .. note::

        Callbacks do not currently support differentiation, and cannot be used inside
        functions that :func:`.catalyst.grad` is applied to.

    Args:
        callback_fn (callable): The function to be used as a callback.
            Any Python-based function is supported, as long as it does not
            return anything (or returns None).

    .. seealso:: :func:`.debug.print`, :func:`.pure_callback`.

    **Example**

    ``debug.callback`` can be used as a decorator:

    .. code-block:: python

        @catalyst.debug.callback
        def callback_fn(y):
            print("Value of y =", y)

        @qjit
        def fn(x):
            y = jnp.sin(x)
            callback_fn(y)
            return y ** 2

    >>> fn(0.54)
    Value of y = 0.5141359916531132
    array(0.26433582)
    >>> fn(1.52)
    Value of y = 0.998710143975583
    array(0.99742195)

    It can also be used functionally:

    .. code-block:: python

        @qjit
        def fn(x):
            y = jnp.sin(x)
            catalyst.debug.callback(lambda z: print("Value of y = ", z))(y)
            return y ** 2

    >>> fn(0.543)
    Value of y =  0.5167068002272901
    array(0.26698592)
    """

    @base_callback
    def closure(*args, **kwargs) -> None:
        retval = callback_fn(*args, **kwargs)
        if retval is not None:
            raise ValueError("debug.callback is expected to return None")

    return closure
