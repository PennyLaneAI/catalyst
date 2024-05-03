# Copyright 2022-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains public API functions that enable host callbacks from
a compiled program. Host callbacks are able to run non-jittable code at runtime
but require a Python interpreter instance.
"""

import ctypes
import inspect
from collections.abc import Sequence
from functools import wraps
from typing import Any, Callable

import jax.numpy as jnp
from jax._src.api_util import shaped_abstractify
from jax._src.tree_util import tree_flatten, tree_leaves, tree_map, tree_unflatten

from catalyst.jax_primitives import python_callback_p
from catalyst.tracing.contexts import EvaluationContext
from catalyst.utils.jnp_to_memref import (
    get_ranked_memref_descriptor,
    get_unranked_memref_descriptor,
    ranked_memref_to_numpy,
)
from catalyst.utils.types import convert_pytype_to_shaped_array


## API ##
def pure_callback(callback_fn, result_type=None):
    """Pure callback

    A pure function is a function that:

      1. Given the same arguments *args, the results will be the same each time the function is
         called.
      2. The function has no side effect.
      3. Examples of side effects include modifying a non-local variable, printing, etc.

    A pure callback is a pure python function that can be executed by the python virtual machine.
    This is in direct contrast to functions which get JIT compiled by Catalyst.

    Using `pure_callback` allows a user to run python.
    `pure_callback`s can be used via a decorator:

    ```python
    @pure_callback
    def add_1(x) -> int:
        return x + 1

    @qjit
    def context(x):
        return add_1(x)

    # Can also be used outside a JIT compiled context
    two = add_1(1)
    ```

    It can also be used through a more functional syntax:


    ```python
    def add_1(x):
        return x + 1

    @qjit
    def context(x):
        return pure_callback(add_1, int)(x)
    ```

    `pure_callback`s are expected to have a return type which matches
    the return type of the function being called. This can be specified
    as type hints in the decorator syntax or as the second parameter in the functional
    syntax.

    At the moment, `pure_callback`s should not be used inside gradients.
    """

    if result_type is None:
        signature = inspect.signature(callback_fn)
        result_type = signature.return_annotation

    result_type = tree_map(convert_pytype_to_shaped_array, result_type)
    if result_type is None:
        msg = "A function using pure_callback requires return types "
        msg += "to be passed in as a parameter or type annotation."
        raise TypeError(msg)

    @base_callback
    def closure(*args, **kwargs) -> result_type:
        return callback_fn(*args, **kwargs)

    return closure


def debug_callback(callback_fn):
    """A function that is useful for printing and logging and allows users to execute a Python
    function (with no return values) at runtime within their qjitted workflows.

    A debug callback is a python function that can write to stdout or to a file.
    It is expected to return no values.

    Using `debug.callback` allows a user to run a python function with side effects inside an
    `@qjit` context. To mark a function as a `debug.callback`, one can use a decorator:

    ```python
    @debug.callback
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
    def my_custom_print(x):
        print(x)

    @qjit
    def foo(x):
        debug.callback(my_custom_print)(x)
    ```

    `debug.callback`s are expected to not return anything.
    May be useful for custom printing and logging into files.

    At the moment, `debug.callback`s should not be used inside gradients.
    """

    @base_callback
    def closure(*args, **kwargs) -> None:
        retval = callback_fn(*args, **kwargs)
        if retval is not None:
            raise ValueError("debug.callback is expected to return None")

    return closure


## IMPL ##
def base_callback(func):
    """Decorator that will correctly pass the signature as arguments to the callback
    implementation.
    """
    signature = inspect.signature(func)
    retty = signature.return_annotation

    # We just disable inconsistent return statements
    # Since we are building this feature step by step.
    @wraps(func)
    def bind_callback(*args, **kwargs):
        if not EvaluationContext.is_tracing():
            # If we are not in the tracing context, just evaluate the function.
            return func(*args, **kwargs)

        return callback_implementation(func, retty, *args, **kwargs)

    return bind_callback


def callback_implementation(
    cb: Callable[..., Any], result_shape_dtypes: Any, *args: Any, **kwargs: Any
):
    """
    This function has been modified from its original form in the JAX project at
    github.com/google/jax/blob/ce0d0c17c39cb78debc78b5eaf9cc3199264a438/jax/_src/callback.py#L231
    version released under the Apache License, Version 2.0, with the following copyright notice:

    Copyright 2022 The JAX Authors.
    """

    flat_args, in_tree = tree_flatten((args, kwargs))
    metadata = CallbackClosure(args, kwargs)

    results_aval = tree_map(convert_pytype_to_shaped_array, result_shape_dtypes)

    flat_results_aval, out_tree = tree_flatten(results_aval)

    def _flat_callback(flat_args):
        """Each flat_arg is a pointer.

        It is a pointer to a memref object.
        To find out which element type it has, we use the signature obtained previously.
        """
        jnpargs = metadata.getArgsAsJAXArrays(flat_args)
        args, kwargs = tree_unflatten(in_tree, jnpargs)
        retvals = tree_leaves(cb(*args, **kwargs))
        return_values = []
        results_aval_sequence = (
            results_aval if isinstance(results_aval, Sequence) else [results_aval]
        )
        for retval, exp_aval in zip(retvals, results_aval_sequence):
            obs_aval = shaped_abstractify(retval)
            if obs_aval != exp_aval:
                raise TypeError(
                    # pylint: disable-next=line-too-long
                    f"Callback {cb.__name__} expected type {exp_aval} but observed {obs_aval} in its return value"
                )
            ranked_memref = get_ranked_memref_descriptor(retval)
            element_size = ctypes.sizeof(ranked_memref.aligned.contents)
            unranked_memref = get_unranked_memref_descriptor(retval)
            unranked_memref_ptr = ctypes.cast(ctypes.pointer(unranked_memref), ctypes.c_void_p)
            # We need to keep a value of retval around
            # Otherwise, Python's garbage collection will collect the memory
            # before we run the memory copy in the runtime.
            # We need to copy the unranked_memref_ptr and we need to know the element size.
            return_values.append((unranked_memref_ptr, element_size, retval))
        return return_values

    out_flat = python_callback_p.bind(
        *flat_args, callback=_flat_callback, results_aval=tuple(flat_results_aval)
    )
    return tree_unflatten(out_tree, out_flat)


class CallbackClosure:
    """This is just a class containing data that is important for the callback."""

    def __init__(self, *absargs, **abskwargs):
        self.absargs = absargs
        self.abskwargs = abskwargs

    @property
    def tree_flatten(self):
        """Flatten args and kwargs."""
        return tree_flatten((self.absargs, self.abskwargs))

    @property
    def low_level_sig(self):
        """Get the memref descriptor types"""
        flat_params, _ = self.tree_flatten
        low_level_flat_params = []
        for param in flat_params:
            empty_memref_descriptor = get_ranked_memref_descriptor(param)
            memref_type = type(empty_memref_descriptor)
            ptr_ty = ctypes.POINTER(memref_type)
            low_level_flat_params.append(ptr_ty)
        return low_level_flat_params

    def getArgsAsJAXArrays(self, flat_args):
        """Get arguments as JAX arrays. Since our integration is mostly compatible with JAX,
        it is best for the user if we continue with that idea and forward JAX arrays."""
        jnpargs = []
        for void_ptr, ty in zip(flat_args, self.low_level_sig):
            memref_ty = ctypes.cast(void_ptr, ty)
            nparray = ranked_memref_to_numpy(memref_ty)
            jnparray = jnp.asarray(nparray)
            jnpargs.append(jnparray)
        return jnpargs
