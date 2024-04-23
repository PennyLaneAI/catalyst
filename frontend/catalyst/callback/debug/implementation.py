# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..implementation import callback

"""debug.callback module"""

def io_callback(callback_fn):
    """IO callback

    An IO callback is a python function that can write to stdout or to a file.
    It is expected to return no values.

    Using `io_callback` allows a user to run a python function with side effects inside an `@qjit`
    context. To mark a function as an `io_callback`, one can use a decorator:

    ```python
    @io_callback
    def my_custom_print(x):
        print(x)

    @qjit
    def foo(x):
       my_custom_print(x)

    # Can also be used outside of a JIT compiled context.
    my_custom_print(x)
    ```

    or through a more functional syntax:

    ```python
    @io_callback
    def my_custom_print(x):
        print(x)

    @qjit
    def foo(x):
        io_callback(my_custom_print)(x)
    ```

    `io_callback`s are expected to not return anything.
    May be useful for custom printing and logging into files.

    At the moment, `pure_callback`s should not be used inside gradients.
    """

    @callback
    def closure(*args, **kwargs) -> None:
        retval = callback_fn(*args, **kwargs)
        if retval is not None:
            raise ValueError("io_callback is expected to return None")

    return closure
