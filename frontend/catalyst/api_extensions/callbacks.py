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
import jax

from catalyst.jax_primitives import python_callback_p
from catalyst.tracing.contexts import EvaluationContext
from catalyst.utils.jnp_to_memref import (
    get_ranked_memref_descriptor,
    get_unranked_memref_descriptor,
    ranked_memref_to_numpy,
)
from catalyst.utils.types import convert_pytype_to_shaped_array


class ActiveCallback:

    def __init__(self, func, restype):
        self.func = func
        self.restype = restype
        self._fwd = None
        self._fwd_restype = None
        self._fwd_jaxpr = None
        self._bwd = None
        self._bwd_restype = None
        self._bwd_jaxpr = None
        self.callback = None

    @staticmethod
    def _get_return_signature(func, result_type):
        if result_type is None:
            signature = inspect.signature(func)
            result_type = signature.return_annotation
        result_type = tree_map(convert_pytype_to_shaped_array, result_type)
        return result_type

    def fwd(self, func, restype=None):
        self._fwd = func
        self._fwd_restype = ActiveCallback._get_return_signature(func, restype)

    def bwd(self, func, restype=None):
        self._bwd = func
        self._bwd_restype = ActiveCallback._get_return_signature(func, restype)

    def __call__(self, *args, **kwargs):
        if self.callback:
            return self.callback(*args, **kwargs)

        def closure(*args, **kwargs) -> self.restype:
            return self.func(*args, **kwargs)

        # We need this here to avoid infinite recursion
        self.callback = base_callback(closure)

        # The arguments here are tracers.
        # And we want to just get the abstraction of the tracers (i.e., the types)
        absargs, abskwargs = tree_map(shaped_abstractify, (args, kwargs))
        # Once we have the types, we can call this self.func with the absargs and abskwargs.
        # We don't need the jaxpr representation but the output is the shape of the cotangents.
        _, cotangents = jax.make_jaxpr(self.func, return_shape=True)(*absargs, **abskwargs)

        # The forward pass must have the same input types as the original function
        self._fwd_jaxpr, shape = jax.make_jaxpr(self._fwd, return_shape=True)(*absargs, **abskwargs)

        # But its output is always going to be two pairs.
        primal, residuals = shape

        # The input for the bwd pass is the residuals and the cotangents.
        self._bwd_jaxpr = jax.make_jaxpr(self._bwd)(residuals, cotangents)

        self.callback = base_callback(
            closure,
            fwd=self._fwd_jaxpr,
            fwd_func=self._fwd,
            bwd=self._bwd_jaxpr,
            bwd_func=self._bwd,
        )

        return self.callback(*args, **kwargs)


## API ##
def pure_callback(callback_fn, result_type=None):
    """Execute and return the results of a functionally pure Python
    function from within a qjit-compiled function.

    The callback function will be quantum just-in-time compiled alongside the rest of the
    workflow, however it will be executed at runtime by the Python virtual machine.
    This is in contrast to functions which get directly qjit-compiled by Catalyst, which will
    be executed at runtime as machine-native code.

    .. note::

        Callbacks do not currently support differentiation, and cannot be used inside
        functions that :func:`.catalyst.grad` is applied to.

    Args:
        callback_fn (callable): The pure function to be used as a callback.
            Any Python-based function is supported, as long as it:

            * is a pure function
              (meaning it is deterministic --- for the same function arguments, the same result
              is always returned --- and has no side effects, such as modifying a non-local
              variable),

            * has a signature that can be inspected (that is, it is not a NumPy ufunc or Python
              builtin),

            * the return type and shape is deterministic and known ahead of time.
        result_type (type): The type returned by the function.

    .. seealso:: :func:`.debug.print`, :func:`.debug.callback`.

    **Example**

    ``pure_callback`` can be used as a decorator. In this case, we must specify the result type
    via a type hint:

    .. code-block:: python

        @catalyst.pure_callback
        def callback_fn(x) -> float:
            # here we call non-JAX compatible code, such
            # as standard NumPy
            return np.sin(x)

        @qjit
        def fn(x):
            return jnp.cos(callback_fn(x ** 2))

    >>> fn(0.654)
    array(0.9151995)

    It can also be used functionally:

    >>> @qjit
    >>> def add_one(x):
    ...     return catalyst.pure_callback(lambda x: x + 1, int)(x)
    >>> add_one(2)
    array(3)

    For callback functions that return arrays, a ``jax.ShapeDtypeStruct``
    object can be created to specify the expected return shape and data type:

    .. code-block:: python

        @qjit
        def fn(x):
            x = jnp.cos(x)

            result_shape = jax.ShapeDtypeStruct(x.shape, jnp.complex128)

            @catalyst.pure_callback
            def callback_fn(y) -> result_shape:
                return jax.jit(jnp.fft.fft)(y)

            x = callback_fn(x)
            return x

    >>> fn(jnp.array([0.1, 0.2]))
    array([1.97507074+0.j, 0.01493759+0.j])
    """

    if result_type is None:
        signature = inspect.signature(callback_fn)
        result_type = signature.return_annotation

    result_type = tree_map(convert_pytype_to_shaped_array, result_type)
    if result_type is None:
        msg = "A function using pure_callback requires return types "
        msg += "to be passed in as a parameter or type annotation."
        raise TypeError(msg)

    return ActiveCallback(callback_fn, result_type)


## IMPL ##
def base_callback(func, fwd=None, fwd_func=None, bwd=None, bwd_func=None):
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

        return callback_implementation(
            func, retty, *args, fwd=fwd, fwd_func=fwd_func, bwd=bwd, bwd_func=bwd_func, **kwargs
        )

    return bind_callback


class FlatCallable:
    """This is a simple class that wraps around a function and calls it with
    a flat list."""

    def __init__(self, func, *params, **kwparams):
        self.func = func
        self.flat_params, self.shape = tree_flatten((params, kwparams))

    def __call__(self, flat_args):
        """args: flat list of arguments
        returns flat list of return values"""
        args, kwargs = tree_unflatten(self.shape, flat_args)
        return tree_leaves(self.func(*args, **kwargs))

    def getOperand(self, i):
        """Get operand at position i"""
        return self.flat_params[i]

    def getOperands(self):
        """Get all operands"""
        return self.flat_params

    def getOperandTypes(self):
        """Get operand types"""
        return map(type, self.getOperands())


class MemrefCallable(FlatCallable):
    """Callable that receives void ptrs."""

    def __init__(self, func, results_aval, *args, **kwargs):
        super().__init__(func, *args, **kwargs)
        self.results_aval = results_aval

    def __call__(self, args):
        jnpargs = self.asarrays(args)
        retvals = super().__call__(jnpargs)
        return_values = []
        results_aval_sequence = (
            self.results_aval if isinstance(self.results_aval, Sequence) else [self.results_aval]
        )
        for retval, exp_aval in zip(retvals, results_aval_sequence):
            self._check_types(retval, exp_aval)
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

    def _check_types(self, obs, exp_aval):
        """Raise error if observed value is different than expected abstract value"""
        obs_aval = shaped_abstractify(obs)
        if obs_aval != exp_aval:
            # pylint: disable-next=line-too-long
            msg = f"Callback {self.func.__name__} expected type {exp_aval} but observed {obs_aval} in its return value"
            raise TypeError(msg)

    def asarrays(self, void_ptrs):
        """cast void_ptrs to jax arrays"""
        expected_types = self.getOperandTypes()
        return MemrefCallable._asarrays(void_ptrs, expected_types)

    @staticmethod
    def _asarrays(void_ptrs, ptr_tys):
        """cast void_ptrs to jax arrays"""
        asarray = MemrefCallable.asarray
        return [asarray(mem, ty) for mem, ty in zip(void_ptrs, ptr_tys)]

    @staticmethod
    def asarray(void_ptr, ptr_ty):
        """cast a single void pointer to a jax array"""
        # The type is guaranteed by JAX, so we don't need
        # to check here.
        ptr_to_memref_descriptor = ctypes.cast(void_ptr, ptr_ty)
        array = ranked_memref_to_numpy(ptr_to_memref_descriptor)
        return jnp.asarray(array)

    def getOperand(self, i):
        """Get operand at position i"""
        array = super().getOperand(i)
        return get_ranked_memref_descriptor(array)

    def getOperands(self):
        """Get operands"""
        operands = super().getOperands()
        return [get_ranked_memref_descriptor(operand) for operand in operands]

    def getOperandTypes(self):
        """Get operand types"""
        operandTys = map(type, self.getOperands())
        return list(map(ctypes.POINTER, operandTys))


def callback_implementation(
    cb: Callable[..., Any],
    result_shape_dtypes: Any,
    *args: Any,
    fwd: None,
    bwd: None,
    fwd_func: None,
    bwd_func: None,
    **kwargs: Any,
):
    """
    This function has been modified from its original form in the JAX project at
    github.com/google/jax/blob/ce0d0c17c39cb78debc78b5eaf9cc3199264a438/jax/_src/callback.py#L231
    version released under the Apache License, Version 2.0, with the following copyright notice:

    Copyright 2022 The JAX Authors.
    """

    flat_args = tree_leaves((args, kwargs))

    results_aval = tree_map(convert_pytype_to_shaped_array, result_shape_dtypes)
    flat_results_aval, out_tree = tree_flatten(results_aval)
    memref_callable = MemrefCallable(cb, results_aval, *args, **kwargs)

    out_flat = python_callback_p.bind(
        *flat_args,
        callback=memref_callable,
        fwd=fwd,
        fwd_func=fwd_func,
        bwd=bwd,
        bwd_func=bwd_func,
        results_aval=tuple(flat_results_aval),
    )
    return tree_unflatten(out_tree, out_flat)
