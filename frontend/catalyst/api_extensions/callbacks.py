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

import copy
import ctypes
import functools
import inspect
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax._src.api_util import shaped_abstractify
from jax._src.tree_util import tree_flatten, tree_leaves, tree_map, tree_unflatten, Partial

from catalyst.jax_extras import transient_jax_config
from catalyst.jax_primitives import python_callback_p
from catalyst.tracing.contexts import EvaluationContext
from catalyst.utils.exceptions import DifferentiableCompileError
from catalyst.utils.jnp_to_memref import (
    get_ranked_memref_descriptor,
    get_unranked_memref_descriptor,
    ranked_memref_to_numpy,
)
from catalyst.utils.types import convert_pytype_to_shaped_array


## API ##
def accelerate(func=None, *, dev=None):
    """Execute a ``jax.jit`` accelerated function on classical
    accelerators such as GPUs from within a qjit-compiled function.

    .. note::

        ``catalyst.accelerate`` doses not currently support
        differentiation, and cannot be used inside functions that
        :func:`catalyst.grad` is applied to.

    Args:
        func (Callable or PjitFunction): The function to be classically
            accelerated from within the qjit-compiled workflow. This
            function can be already just-in-time compiled with JAX via
            the ``jax.jit`` decorator and a specified device. If not,
            it will be implicitly JIT-compiled, and so must be JIT
            compatible.
        dev (jax.Device): the classical accelerator device the JIT-compiled
            function will run on. Available devices can be retrieved via
            ``jax.devices()``. If not provided, the default value of
            ``jax.devices()[0]`` as determined by JAX will be used.

    .. seealso:: :func:`~.pure_callback`, :func:`.debug.callback`.

    **Example**

    .. code-block:: python

        @accelerate(dev=jax.devices("gpu")[0])
        def classical_fn(x):
            return jnp.sin(x) ** 2

        @qjit
        def hybrid_fn(x):
            y = classical_fn(jnp.sqrt(x)) # will be executed on a GPU
            return jnp.cos(y)

    In addition, you can accelerate function that have already been
    ``jax.jit`` decorated:

    .. code-block:: python

        @jax.jit
        def classical_fn(x):
            x = jax.device_put(x, jax.local_devices("gpu")[0])
            return jnp.sin(x) ** 2

        @qjit
        def hybrid_fn(x):
            y = accelerate(classical_fn)(x) # will be executed on a GPU
            return jnp.cos(y)
    """

    if dev is None:
        dev = jax.devices()[0]

    if not isinstance(func, Callable):
        kwargs = copy.copy(locals())
        kwargs.pop("func")
        return functools.partial(accelerate, **kwargs)

    # If this is a partial, we need to make the tracers part of the input
    is_partial = isinstance(func, Partial)
    context = []
    if is_partial:
        context = tree_leaves(func)

    def total(context, *args, **kwargs):
        nonlocal func
        if is_partial:
            _, shape = tree_flatten(func)
            func = tree_unflatten(shape, context)
            return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    with transient_jax_config({"jax_dynamic_shapes": False}):
        jitted_fn = jax.jit(total)

    @functools.wraps(func)
    def defer(*args, **kwargs):
        absextra, absargs, abskwargs = tree_map(shaped_abstractify, (context, args, kwargs))
        try:
            # Find the shape of the return value
            with transient_jax_config({"jax_dynamic_shapes": False}):
                _, returnshape = jax.make_jaxpr(total, return_shape=True)(
                    absextra, *absargs, **abskwargs
                )
        except Exception as e:
            name = func.__name__
            msg = f"Function {name} must be jax.jit-able."
            raise ValueError(msg) from e
        return jax_jit_callback(jitted_fn, returnshape, device=dev)(context, *args, **kwargs)

    return defer


def pure_callback(callback_fn, result_type=None):
    """Execute and return the results of a functionally pure Python
    function from within a qjit-compiled function.

    The callback function will be quantum just-in-time compiled alongside the rest of the
    workflow, however it will be executed at runtime by the Python virtual machine.
    This is in contrast to functions which get directly qjit-compiled by Catalyst, which will
    be executed at runtime as machine-native code.

    .. note::

        Callbacks do not automatically support differentiation. To use them
        within functions that are being differentiated, please define their
        vector-Jacobian product (see below for more details).

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

    .. seealso:: :func:`accelerate`, :func:`.debug.print`, :func:`.debug.callback`.

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

    .. details::
        :title: Differentiating callbacks with custom VJP rules

        Pure callbacks must have custom gradients manually
        registered with the Catalyst compiler in order to support differentiation.

        This can be done via the ``pure_callback.fwd`` and ``pure_callback.bwd`` methods,
        to specify how the forwards and backwards pass (the vector-Jacobian product)
        of the callback should be computed:

        .. code-block:: python

            @catalyst.pure_callback
            def callback_fn(x) -> float:
                return np.sin(x[0]) * x[1]

            @callback_fn.fwd
            def callback_fn_fwd(x):
                # returns the evaluated function as well as residual
                # values that may be useful for the backwards pass
                return callback_fn(x), x

            @callback_fn.bwd
            def callback_fn_vjp(res, dy):
                # Accepts residuals from the forward pass, as well
                # as (one or more) cotangent vectors dy, and returns
                # a tuple of VJPs corresponding to each input parameter.

                def vjp(x, dy) -> (jax.ShapeDtypeStruct((2,), jnp.float64),):
                    return (np.array([np.cos(x[0]) * dy * x[1], np.sin(x[0]) * dy]),)

                # The VJP function can also be a pure callback
                return catalyst.pure_callback(vjp)(res, dy)

        >>> @qml.qjit
        ... @catalyst.grad
        ... def f(x):
        ...     y = jnp.array([jnp.cos(x[0]), x[1]])
        ...     return jnp.sin(callback_fn(y))
        >>> f(jnp.array([0.1, 0.2]))
        array([-0.01071923,  0.82698717])
    """

    if result_type is None:
        signature = inspect.signature(callback_fn)
        result_type = signature.return_annotation

    result_type = tree_map(convert_pytype_to_shaped_array, result_type)
    if result_type is None:
        msg = "A function using pure_callback requires return types "
        msg += "to be passed in as a parameter or type annotation."
        raise TypeError(msg)

    return CallbackWithPotentialCustomGrad(callback_fn, result_type)


## IMPL ##
class CallbackWithCustomGrad:
    """A callback with a custom grad"""

    def __init__(self, func, restype, forward, reverse):
        assert func and forward and reverse
        self.func = func
        self.restype = restype
        self._fwd = forward
        self._fwd_jaxpr = None
        self._bwd = reverse
        self._bwd_jaxpr = None
        self.callback = None

    def __call__(self, *args, **kwargs):
        if self.callback:
            return self.callback(*args, **kwargs)

        def closure(*args, **kwargs) -> self.restype:
            return self.func(*args, **kwargs)

        # We need this here to avoid infinite recursion
        # Where does the infinite recursion happen?
        # It happens if the fwd or bwd passes have a call to
        # the pure_callback implementation.
        self.callback = base_callback(closure, custom_grad=self)

        # The arguments here are tracers.
        # And we want to just get the abstraction of the tracers (i.e., the types)
        absargs, abskwargs = tree_map(shaped_abstractify, (args, kwargs))
        cotangents = tree_map(shaped_abstractify, self.restype)

        # The forward pass must have the same input types as the original function
        with transient_jax_config({"jax_dynamic_shapes": False}):
            self._fwd_jaxpr, shape = jax.make_jaxpr(self._fwd, return_shape=True)(
                *absargs, **abskwargs
            )

        # But its output is always going to be two pairs.
        _primal, residuals = shape

        # The input for the bwd pass is the residuals and the cotangents.
        with transient_jax_config({"jax_dynamic_shapes": False}):
            self._bwd_jaxpr = jax.make_jaxpr(self._bwd)(residuals, cotangents)

        return self.callback(*args, **kwargs)


class CallbackWithPotentialCustomGrad:
    """A callback which is not guaranteed to have a custom grad,
    but the user may define one. E.g., a pure_callback is not required
    to have a custom grad if it is never differentiated, but a user
    may register one. A debug.callback will never have a custom grad."""

    def __init__(self, func, restype):
        self.func = func
        self.restype = restype
        self._fwd = None
        self._bwd = None
        self.callback = None

    def fwd(self, func):
        """Save forward pass as implemented by the user"""
        self._fwd = func

    def bwd(self, func):
        """Save reverse pass as implemented by the user"""
        self._bwd = func

    def __call__(self, *args, **kwargs):
        incomplete_grad = bool(self._fwd) != bool(self._bwd)
        if incomplete_grad:
            # If we are here, then we have either _fwd and _bwd but not both
            msg = f"Function {self.func} differentiated but missing "
            msg += "forward" if not self._fwd else "reverse"
            msg += " pass"
            raise DifferentiableCompileError(msg)

        if self.callback:
            return self.callback(*args, **kwargs)

        if self._fwd and self._bwd:
            self.callback = CallbackWithCustomGrad(self.func, self.restype, self._fwd, self._bwd)
            return self.callback(*args, **kwargs)

        def closure(*args, **kwargs) -> self.restype:
            return self.func(*args, **kwargs)

        self.callback = base_callback(closure)
        return self.callback(*args, **kwargs)


def jax_jit_callback(callback_fn, result_type, device=None):
    """Wrapper around base callback that can accept a device as a parameter"""

    result_type = tree_map(convert_pytype_to_shaped_array, result_type)

    def closure(*args, **kwargs) -> result_type:
        return callback_fn(*args, **kwargs)

    return base_callback(closure, device=device)


## IMPL ##
def base_callback(func, device=None, custom_grad=None):
    """Decorator that will correctly pass the signature as arguments to the callback
    implementation.
    """
    signature = inspect.signature(func)
    retty = signature.return_annotation

    # We just disable inconsistent return statements
    # Since we are building this feature step by step.
    @functools.wraps(func)
    def bind_callback(*args, **kwargs):
        if not EvaluationContext.is_tracing():
            # If we are not in the tracing context, just evaluate the function.
            return func(*args, **kwargs)

        return callback_implementation(
            func, retty, *args, device=device, custom_grad=custom_grad, **kwargs
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

    CACHE = {}

    def __new__(cls, func, results_aval, *args, **kwargs):
        # Hash-cons: https://en.wikipedia.org/wiki/Hash_consing
        absargs, abskwargs = tree_map(shaped_abstractify, (args, kwargs))
        flat_params, _ = tree_flatten((absargs, abskwargs))
        flat_results_aval, _ = tree_flatten(results_aval)
        cache_key = (func, *flat_params, *flat_results_aval)
        if cls.CACHE.get(cache_key):
            return cls.CACHE.get(cache_key)

        instance = super().__new__(cls)
        cls.CACHE[cache_key] = instance
        return instance

    @classmethod
    def clearcache(cls):
        """Clear the memref callable cache"""
        cls.CACHE.clear()

    def __init__(self, func, results_aval, *args, **kwargs):
        super().__init__(func, *args, **kwargs)
        self.results_aval = results_aval

    def __call__(self, args):
        jnpargs = self.asarrays(args)
        retvals = super().__call__(jnpargs)
        return_values = []
        flat_results_aval, _ = tree_flatten(self.results_aval)
        for retval, exp_aval in zip(retvals, flat_results_aval):
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


class JaxJitCallable(MemrefCallable):
    """Callable that places the arguments in device before execution"""

    def __init__(self, func, device, results_aval, *args, **kwargs):
        assert device is not None, "Cannot have none device"
        self.device = device
        super().__init__(func, results_aval, *args, **kwargs)

    def asarrays(self, void_ptrs):
        """cast void_ptrs to jax arrays and move them to a device"""
        expected_types = self.getOperandTypes()
        jnparrays = MemrefCallable._asarrays(void_ptrs, expected_types)
        movedarrays = [jax.device_put(array, self.device) for array in jnparrays]
        return movedarrays


def callback_implementation(
    cb: Callable[..., Any],
    result_shape_dtypes: Any,
    *args: Any,
    device=None,
    custom_grad=None,
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
    if device is None:
        memref_callable = MemrefCallable(cb, results_aval, *args, **kwargs)
    else:
        memref_callable = JaxJitCallable(cb, device, results_aval, *args, **kwargs)

    out_flat = python_callback_p.bind(
        *flat_args,
        callback=memref_callable,
        custom_grad=custom_grad,
        results_aval=tuple(flat_results_aval),
    )
    return tree_unflatten(out_tree, out_flat)
