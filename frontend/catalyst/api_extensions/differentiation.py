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
This module contains public API functions that provide differentiation
capabilities for hybrid quantum programs. This includes the computation
of gradients, jacobians, jacobian-vector products, and more.
"""

import copy
import functools
import numbers
from typing import Callable, Iterable, List, Optional, Union

import jax
from jax._src.tree_util import PyTreeDef, tree_flatten, tree_unflatten
from pennylane import QNode

import catalyst
from catalyst.jax_extras import Jaxpr
from catalyst.jax_primitives import (
    GradParams,
    expval_p,
    func_p,
    grad_p,
    jvp_p,
    probs_p,
    vjp_p,
)
from catalyst.jax_tracer import Function
from catalyst.tracing.contexts import EvaluationContext
from catalyst.utils.exceptions import DifferentiableCompileError

Differentiable = Union[Function, QNode]
DifferentiableLike = Union[Differentiable, Callable, "catalyst.QJIT"]


## API ##
def grad(fn=None, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible gradient transformation for PennyLane/Catalyst.

    This function allows the gradient of a hybrid quantum-classical function to be computed within
    the compiled program. Outside of a compiled function, this function will simply dispatch to its
    JAX counterpart ``jax.grad``. The function ``f`` can return any pytree-like shape.

    .. warning::

        Currently, higher-order differentiation is only supported by the finite-difference
        method.

    Args:
        fn (Callable): a function or a function object to differentiate
        method (str): The method used for differentiation, which can be any of ``["auto", "fd"]``,
                      where:

                      - ``"auto"`` represents deferring the quantum differentiation to the method
                        specified by the QNode, while the classical computation is differentiated
                        using traditional auto-diff. Catalyst supports ``"parameter-shift"`` and
                        ``"adjoint"`` on internal QNodes. Notably, QNodes with
                        ``diff_method="finite-diff"`` is not supported with ``"auto"``.

                      - ``"fd"`` represents first-order finite-differences for the entire hybrid
                        function.

        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnum (Tuple[int, List[int]]): the argument indices to differentiate

    Returns:
        Callable: A callable object that computes the gradient of the wrapped function for the given
                  arguments.

    Raises:
        ValueError: Invalid method or step size parameters.
        DifferentiableCompilerError: Called on a function that doesn't return a single scalar.

    .. note::

        Any JAX-compatible optimization library, such as `JAXopt
        <https://jaxopt.github.io/stable/index.html>`_, can be used
        alongside ``grad`` for JIT-compatible variational workflows.
        See the :doc:`/dev/quick_start` for examples.

    .. seealso:: :func:`~.jacobian`

    **Example 1 (Classical preprocessing)**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        def workflow(x):
            @qml.qnode(dev)
            def circuit(x):
                qml.RX(jnp.pi * x, wires=0)
                return qml.expval(qml.PauliY(0))

            g = grad(circuit)
            return g(x)

    >>> workflow(2.0)
    array(-3.14159265)

    **Example 2 (Classical preprocessing and postprocessing)**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        def grad_loss(theta):
            @qml.qnode(dev, diff_method="adjoint")
            def circuit(theta):
                qml.RX(jnp.exp(theta ** 2) / jnp.cos(theta / 4), wires=0)
                return qml.expval(qml.PauliZ(wires=0))

            def loss(theta):
                return jnp.pi / jnp.tanh(circuit(theta))

            return catalyst.grad(loss, method="auto")(theta)

    >>> grad_loss(1.0)
    array(-1.90958669)

    **Example 3 (Multiple QNodes with their own differentiation methods)**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        def grad_loss(theta):
            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit_A(params):
                qml.RX(jnp.exp(params[0] ** 2) / jnp.cos(params[1] / 4), wires=0)
                return qml.probs()

            @qml.qnode(dev, diff_method="adjoint")
            def circuit_B(params):
                qml.RX(jnp.exp(params[1] ** 2) / jnp.cos(params[0] / 4), wires=0)
                return qml.expval(qml.PauliZ(wires=0))

            def loss(params):
                return jnp.prod(circuit_A(params)) + circuit_B(params)

            return catalyst.grad(loss)(theta)

    >>> grad_loss(jnp.array([1.0, 2.0]))
    array([ 0.57367285, 44.4911605 ])

    **Example 4 (Purely classical functions)**

    .. code-block:: python

        def square(x: float):
            return x ** 2

        @qjit
        def dsquare(x: float):
            return catalyst.grad(square)(x)

    >>> dsquare(2.3)
    array(4.6)
    """
    kwargs = copy.copy(locals())
    kwargs.pop("fn")

    if fn is None:
        return functools.partial(grad, **kwargs)

    scalar_out = True

    return Grad(fn, GradParams(method, scalar_out, h, argnum))


def value_and_grad(fn=None, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible gradient transformation for PennyLane/Catalyst.

    This function allows the value and the gradient of a hybrid quantum-classical function to be
    computed within the compiled program. Outside of a compiled function, this function will
    simply dispatch to its JAX counterpart ``jax.value_and_grad``. The function ``f`` can return
    any pytree-like shape.

    .. warning::

        Currently, higher-order differentiation is only supported by the finite-difference
        method.

    Args:
        fn (Callable): a function or a function object to differentiate
        method (str): The method used for differentiation, which can be any of ``["auto", "fd"]``,
                      where:

                      - ``"auto"`` represents deferring the quantum differentiation to the method
                        specified by the QNode, while the classical computation is differentiated
                        using traditional auto-diff. Catalyst supports ``"parameter-shift"`` and
                        ``"adjoint"`` on internal QNodes. Notably, QNodes with
                        ``diff_method="finite-diff"`` is not supported with ``"auto"``.

                      - ``"fd"`` represents first-order finite-differences for the entire hybrid
                        function.

        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnum (Tuple[int, List[int]]): the argument indices to differentiate

    Returns:
        Callable: A callable object that computes the value and gradient of the wrapped function
        for the given arguments.

    Raises:
        ValueError: Invalid method or step size parameters.
        DifferentiableCompilerError: Called on a function that doesn't return a single scalar.

    .. note::

        Any JAX-compatible optimization library, such as `JAXopt
        <https://jaxopt.github.io/stable/index.html>`_, can be used
        alongside ``value_and_grad`` for JIT-compatible variational workflows.
        See the :doc:`/dev/quick_start` for examples.

    .. seealso:: :func:`~.jacobian`

    **Example 1 (Classical preprocessing)**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        def workflow(x):
            @qml.qnode(dev)
            def circuit(x):
                qml.RX(jnp.pi * x, wires=0)
                return qml.expval(qml.PauliY(0))

            g = value_and_grad(circuit)
            return g(x)

    >>> workflow(2.0)
    (array(0.2), array(-3.14159265))

    **Example 2 (Classical preprocessing and postprocessing)**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        def value_and_grad_loss(theta):
            @qml.qnode(dev, diff_method="adjoint")
            def circuit(theta):
                qml.RX(jnp.exp(theta ** 2) / jnp.cos(theta / 4), wires=0)
                return qml.expval(qml.PauliZ(wires=0))

            def loss(theta):
                return jnp.pi / jnp.tanh(circuit(theta))

            return catalyst.value_and_grad(loss, method="auto")(theta)

    >>> value_and_grad_loss(1.0)
    (array(-4.12502201), array(4.34374983))

    **Example 3 (Purely classical functions)**

    .. code-block:: python

        def square(x: float):
            return x ** 2

        @qjit
        def dsquare(x: float):
            return catalyst.value_and_grad(square)(x)

    >>> dsquare(2.3)
    (array(5.29), array(4.6))
    """
    kwargs = copy.copy(locals())
    kwargs.pop("fn")

    if fn is None:
        return functools.partial(value_and_grad, **kwargs)

    scalar_out = True

    return Grad(fn, GradParams(method, scalar_out, h, argnum, with_value=True))


def jacobian(fn=None, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible Jacobian transformation for PennyLane/Catalyst.

    This function allows the Jacobian of a hybrid quantum-classical function to be computed within
    the compiled program. Outside of a compiled function, this function will simply dispatch to its
    JAX counterpart ``jax.jacobian``. The function ``f`` can return any pytree-like shape.

    Args:
        fn (Callable): a function or a function object to differentiate
        method (str): The method used for differentiation, which can be any of ``["auto", "fd"]``,
                      where:

                      - ``"auto"`` represents deferring the quantum differentiation to the method
                        specified by the QNode, while the classical computation is differentiated
                        using traditional auto-diff. Catalyst supports ``"parameter-shift"`` and
                        ``"adjoint"`` on internal QNodes. Notably, QNodes with
                        ``diff_method="finite-diff"`` is not supported with ``"auto"``.

                      - ``"fd"`` represents first-order finite-differences for the entire hybrid
                        function.

        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnum (Tuple[int, List[int]]): the argument indices to differentiate

    Returns:
        Callable: A callable object that computes the Jacobian of the wrapped function for the given
                  arguments.

    Raises:
        ValueError: Invalid method or step size parameters.

    .. note::

        Any JAX-compatible optimization library, such as `JAXopt
        <https://jaxopt.github.io/stable/index.html>`_, can be used
        alongside ``jacobian`` for JIT-compatible variational workflows.
        See the :doc:`/dev/quick_start` for examples.

    .. seealso:: :func:`~.grad`

    **Example**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qjit
        def workflow(x):
            @qml.qnode(dev)
            def circuit(x):
                qml.RX(jnp.pi * x[0], wires=0)
                qml.RY(x[1], wires=0)
                return qml.probs()

            g = jacobian(circuit)
            return g(x)

    >>> workflow(jnp.array([2.0, 1.0]))
    array([[ 3.48786850e-16 -4.20735492e-01]
           [-8.71967125e-17  4.20735492e-01]])
    """
    kwargs = copy.copy(locals())
    kwargs.pop("fn")

    if fn is None:
        return functools.partial(grad, **kwargs)

    scalar_out = False

    return Grad(fn, GradParams(method, scalar_out, h, argnum))


# pylint: disable=too-many-arguments
def jvp(f: DifferentiableLike, params, tangents, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible Jacobian-vector product for PennyLane/Catalyst.

    This function allows the Jacobian-vector Product of a hybrid quantum-classical function to be
    computed within the compiled program. Outside of a compiled function, this function will simply
    dispatch to its JAX counterpart ``jax.jvp``. The function ``f`` can return any pytree-like
    shape.

    Args:
        f (Callable): Function-like object to calculate JVP for
        params (List[Array]): List (or a tuple) of the function arguments specifying the point
                              to calculate JVP at. A subset of these parameters are declared as
                              differentiable by listing their indices in the ``argnum`` parameter.
        tangents(List[Array]): List (or a tuple) of tangent values to use in JVP. The list size and
                               shapes must match the ones of differentiable params.
        method(str): Differentiation method to use, same as in :func:`~.grad`.
        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnum (Union[int, List[int]]): the params' indices to differentiate.

    Returns:
        Tuple[Any]: Return values of ``f`` paired with the JVP values.

    Raises:
        TypeError: invalid parameter types
        ValueError: invalid parameter values

    **Example 1 (basic usage)**

    .. code-block:: python

        @qjit
        def jvp(params, tangent):
          def f(x):
              y = [jnp.sin(x[0]), x[1] ** 2, x[0] * x[1]]
              return jnp.stack(y)

          return catalyst.jvp(f, [params], [tangent])

    >>> x = jnp.array([0.1, 0.2])
    >>> tangent = jnp.array([0.3, 0.6])
    >>> jvp(x, tangent)
    (array([0.09983342, 0.04      , 0.02      ]), array([0.29850125, 0.24      , 0.12      ]))

    **Example 2 (argnum usage)**

    Here we show how to use ``argnum`` to ignore the non-differentiable parameter ``n`` of the
    target function. Note that the length and shapes of tangents must match the length and shape of
    primal parameters which we mark as differentiable by passing their indices to ``argnum``.

    .. code-block:: python

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(n, params):
            qml.RX(params[n, 0], wires=n)
            qml.RY(params[n, 1], wires=n)
            return qml.expval(qml.PauliZ(1))

        @qjit
        def workflow(primals, tangents):
            return catalyst.jvp(circuit, [1, primals], [tangents], argnum=[1])

    >>> params = jnp.array([[0.54, 0.3154], [0.654, 0.123]])
    >>> dy = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    >>> workflow(params, dy)
    (array(0.78766064), array(-0.7011436))
    """

    def check_is_iterable(x, hint):
        if not isinstance(x, Iterable):
            raise ValueError(f"vjp '{hint}' argument must be an iterable, not {type(x)}")

    check_is_iterable(params, "params")
    check_is_iterable(tangents, "tangents")

    if EvaluationContext.is_tracing():
        scalar_out = False
        fn = _ensure_differentiable(f)
        args_flatten, in_tree = tree_flatten(params)
        tangents_flatten, _ = tree_flatten(tangents)
        grad_params = _check_grad_params(method, scalar_out, h, argnum, len(args_flatten), in_tree)

        jaxpr, out_tree = _make_jaxpr_check_differentiable(fn, grad_params, *params)

        results = jvp_p.bind(
            *args_flatten, *tangents_flatten, jaxpr=jaxpr, fn=fn, grad_params=grad_params
        )

        midpoint = len(results) // 2
        func_res = results[:midpoint]
        jvps = results[midpoint:]

        func_res = tree_unflatten(out_tree, func_res)
        jvps = tree_unflatten(out_tree, jvps)
        results = (func_res, jvps)

    else:
        results = jax.jvp(f, params, tangents)

    return results


# pylint: disable=too-many-arguments
def vjp(f: DifferentiableLike, params, cotangents, *, method=None, h=None, argnum=None):
    """A :func:`~.qjit` compatible Vector-Jacobian product for PennyLane/Catalyst.

    This function allows the Vector-Jacobian Product of a hybrid quantum-classical function to be
    computed within the compiled program. Outside of a compiled function, this function will simply
    dispatch to its JAX counterpart ``jax.vjp``. The function ``f`` can return any pytree-like
    shape.

    Args:
        f(Callable): Function-like object to calculate JVP for
        params(List[Array]): List (or a tuple) of f's arguments specifying the point to calculate
                             VJP at. A subset of these parameters are declared as
                             differentiable by listing their indices in the ``argnum`` parameter.
        cotangents(List[Array]): List (or a tuple) of tangent values to use in JVP. The list size
                                 and shapes must match the size and shape of ``f`` outputs.
        method(str): Differentiation method to use, same as in ``grad``.
        h (float): the step-size value for the finite-difference (``"fd"``) method
        argnum (Union[int, List[int]]): the params' indices to differentiate.

    Returns:
        Tuple[Any]): Return values of ``f`` paired with the VJP values.

    Raises:
        TypeError: invalid parameter types
        ValueError: invalid parameter values

    **Example**

    .. code-block:: python

        @qjit
        def vjp(params, cotangent):
          def f(x):
              y = [jnp.sin(x[0]), x[1] ** 2, x[0] * x[1]]
              return jnp.stack(y)

          return catalyst.vjp(f, [params], [cotangent])

    >>> x = jnp.array([0.1, 0.2])
    >>> dy = jnp.array([-0.5, 0.1, 0.3])
    >>> vjp(x, dy)
    (array([0.09983342, 0.04      , 0.02      ]), (array([-0.43750208,  0.07      ]),))
    """

    def check_is_iterable(x, hint):
        if not isinstance(x, Iterable):
            raise ValueError(f"vjp '{hint}' argument must be an iterable, not {type(x)}")

    check_is_iterable(params, "params")
    check_is_iterable(cotangents, "cotangents")

    if EvaluationContext.is_tracing():
        scalar_out = False
        fn = _ensure_differentiable(f)

        args_flatten, in_tree = tree_flatten(params)
        cotangents_flatten, _ = tree_flatten(cotangents)

        grad_params = _check_grad_params(method, scalar_out, h, argnum, len(args_flatten), in_tree)

        args_argnum = tuple(params[i] for i in grad_params.argnum)
        _, in_tree = tree_flatten(args_argnum)

        jaxpr, out_tree = _make_jaxpr_check_differentiable(fn, grad_params, *params)

        cotangents, _ = tree_flatten(cotangents)

        results = vjp_p.bind(
            *args_flatten, *cotangents_flatten, jaxpr=jaxpr, fn=fn, grad_params=grad_params
        )

        func_res = results[: len(jaxpr.out_avals)]
        vjps = results[len(jaxpr.out_avals) :]
        func_res = tree_unflatten(out_tree, func_res)
        vjps = tree_unflatten(in_tree, vjps)

        results = (func_res, vjps)

    else:
        if isinstance(params, jax.numpy.ndarray) and params.ndim == 0:
            primal_outputs, vjp_fn = jax.vjp(f, params)
        else:
            primal_outputs, vjp_fn = jax.vjp(f, *params)

        results = (primal_outputs, vjp_fn(cotangents))

    return results


## IMPL ##
class Grad:
    """An object that specifies that a function will be differentiated.

    Args:
        fn (Differentiable): the function to differentiate
        method (str): the method used for differentiation
        h (float): the step-size value for the finite difference method
        argnum (list[int]): the argument indices which define over which arguments to differentiate

    Raises:
        ValueError: Higher-order derivatives and derivatives of non-QNode functions can only be
                    computed with the finite difference method.
        TypeError: Non-differentiable object was passed as `fn` argument.
    """

    def __init__(self, fn: Differentiable, grad_params: GradParams):
        self.fn = fn
        self.__name__ = f"grad.{getattr(fn, '__name__', 'unknown')}"
        self.grad_params = grad_params

    def __call__(self, *args, **kwargs):
        """Specifies that an actual call to the differentiated function.
        Args:
            args: the arguments to the differentiated function
        """

        if EvaluationContext.is_tracing():
            assert (
                not self.grad_params.with_value
            ), "Tracing of value_and_grad is not implemented yet"

            fn = _ensure_differentiable(self.fn)

            args_data, in_tree = tree_flatten(args)
            grad_params = _check_grad_params(
                self.grad_params.method,
                self.grad_params.scalar_out,
                self.grad_params.h,
                self.grad_params.argnum,
                len(args_data),
                in_tree,
                self.grad_params.with_value,
            )
            jaxpr, out_tree = _make_jaxpr_check_differentiable(fn, grad_params, *args)
            args_argnum = tuple(args[i] for i in grad_params.argnum)
            _, in_tree = tree_flatten(args_argnum)

            # It always returns list as required by catalyst control-flows
            results = grad_p.bind(*args_data, jaxpr=jaxpr, fn=fn, grad_params=grad_params)

            results = _unflatten_derivatives(
                results, in_tree, out_tree, grad_params, len(jaxpr.out_avals)
            )
        else:
            if argnums := self.grad_params.argnum is None:
                argnums = 0
            if self.grad_params.scalar_out:
                if self.grad_params.with_value:
                    results = jax.value_and_grad(self.fn, argnums=argnums)(*args)
                else:
                    results = jax.grad(self.fn, argnums=argnums)(*args)
            else:
                assert (
                    not self.grad_params.with_value
                ), "value_and_grad cannot be used with a Jacobian"
                results = jax.jacobian(self.fn, argnums=argnums)(*args)

        return results


## PRIVATE ##
# pylint: disable=too-many-arguments
def _check_grad_params(
    method: str,
    scalar_out: bool,
    h: Optional[float],
    argnum: Optional[Union[int, List[int]]],
    len_flatten_args: int,
    in_tree: PyTreeDef,
    with_value: bool = False,
) -> GradParams:
    """Check common gradient parameters and produce a class``GradParams`` object"""
    methods = {"fd", "auto"}
    if method is None:
        method = "auto"
    if method not in methods:
        raise ValueError(
            f"Invalid differentiation method '{method}'. "
            f"Supported methods are: {' '.join(sorted(methods))}"
        )
    if method == "fd" and h is None:
        h = 1e-7
    if not (h is None or isinstance(h, numbers.Number)):
        raise ValueError(f"Invalid h value ({h}). None or number was expected.")
    if argnum is None:
        argnum_list = [0]
    elif isinstance(argnum, int):
        argnum_list = [argnum]
    elif isinstance(argnum, tuple):
        argnum_list = list(argnum)
    elif isinstance(argnum, list) and all(isinstance(i, int) for i in argnum):
        argnum_list = argnum
    else:
        raise ValueError(f"argnum should be integer or a list of integers, not {argnum}")
    # Compute the argnums of the pytree arg
    total_argnums = list(range(0, len_flatten_args))
    argnum_unflatten = tree_unflatten(in_tree, total_argnums)
    argnum_selected = [argnum_unflatten[i] for i in argnum_list]
    argnum_expanded, _ = tree_flatten(argnum_selected)
    scalar_argnum = isinstance(argnum, int) or argnum is None
    return GradParams(
        method, scalar_out, h, argnum_list, scalar_argnum, argnum_expanded, with_value
    )


def _unflatten_derivatives(results, in_tree, out_tree, grad_params, num_results):
    """Unflatten the flat list of derivatives results given the out tree."""
    num_trainable_params = len(grad_params.expanded_argnum)
    results_final = []

    for i in range(0, num_results):
        intermediate_results = results[
            i * num_trainable_params : i * num_trainable_params + num_trainable_params
        ]
        intermediate_results = tree_unflatten(in_tree, intermediate_results)
        if grad_params.scalar_argnum:
            intermediate_results = intermediate_results[0]
        else:
            intermediate_results = tuple(intermediate_results)
        results_final.append(intermediate_results)

    results_final = tree_unflatten(out_tree, results_final)
    return results_final


def _ensure_differentiable(f: DifferentiableLike) -> Differentiable:
    """Narrows down the set of the supported differentiable objects."""

    # Unwrap the function from an existing QJIT object.
    if isinstance(f, catalyst.QJIT):
        f = f.user_function

    if isinstance(f, (Function, QNode)):
        return f
    elif isinstance(f, Callable):  # Keep at the bottom
        return Function(f)

    raise DifferentiableCompileError(f"Non-differentiable object passed: {type(f)}")


def _make_jaxpr_check_differentiable(f: Differentiable, grad_params: GradParams, *args) -> Jaxpr:
    """Gets the jaxpr of a differentiable function. Perform the required additional checks and
    return the output tree."""
    method = grad_params.method
    jaxpr, shape = jax.make_jaxpr(f, return_shape=True)(*args)
    _, out_tree = tree_flatten(shape)
    assert len(jaxpr.eqns) == 1, "Expected jaxpr consisting of a single function call."
    assert jaxpr.eqns[0].primitive == func_p, "Expected jaxpr consisting of a single function call."

    for pos, arg in enumerate(jaxpr.in_avals):
        if arg.dtype.kind != "f" and pos in grad_params.expanded_argnum:
            raise DifferentiableCompileError(
                "Catalyst.grad/jacobian only supports differentiation on floating-point "
                f"arguments, got '{arg.dtype}' at position {pos}."
            )

    if grad_params.scalar_out:
        if not (len(jaxpr.out_avals) == 1 and jaxpr.out_avals[0].shape == ()):
            raise DifferentiableCompileError(
                f"Catalyst.grad only supports scalar-output functions, got {jaxpr.out_avals}"
            )

    for pos, res in enumerate(jaxpr.out_avals):
        if res.dtype.kind != "f":
            raise DifferentiableCompileError(
                "Catalyst.grad/jacobian only supports differentiation on floating-point "
                f"results, got '{res.dtype}' at position {pos}."
            )

    _verify_differentiable_child_qnodes(jaxpr, method)

    return jaxpr, out_tree


def _verify_differentiable_child_qnodes(jaxpr, method):
    """Traverse QNodes being differentiated in the 'call graph' of the JAXPR to verify them."""
    visited = set()

    def traverse_children(jaxpr):
        for eqn in jaxpr.eqns:
            primitive = eqn.primitive
            if primitive is func_p:
                child_jaxpr = eqn.params.get("call_jaxpr")
            elif primitive is grad_p:
                child_jaxpr = eqn.params.get("jaxpr")
            else:
                continue

            _check_primitive_is_differentiable(primitive, method)

            py_callable = eqn.params.get("fn")
            if py_callable not in visited:
                if isinstance(py_callable, QNode):
                    _check_qnode_against_grad_method(py_callable, method, child_jaxpr)
                traverse_children(child_jaxpr)
                visited.add(py_callable)

    traverse_children(jaxpr)


def _check_primitive_is_differentiable(primitive, method):
    """Verify restriction on primitives in the call graph of a Grad operation."""

    if primitive is grad_p and method != "fd":
        raise DifferentiableCompileError(
            "Only finite difference can compute higher order derivatives."
        )


def _check_qnode_against_grad_method(f: QNode, method: str, jaxpr: Jaxpr):
    """Additional checks for the given jaxpr of a differentiable function."""
    if method == "fd":
        return

    return_ops = []
    for res in jaxpr.outvars:
        for eq in reversed(jaxpr.eqns):  # pragma: no branch
            if res in eq.outvars:
                return_ops.append(eq.primitive)
                break

    if f.diff_method is None:
        raise DifferentiableCompileError(
            "Cannot differentiate a QNode explicitly marked non-differentiable (with "
            "diff_method=None)."
        )

    if f.diff_method == "parameter-shift" and any(
        prim not in [expval_p, probs_p] for prim in return_ops
    ):
        raise DifferentiableCompileError(
            "The parameter-shift method can only be used for QNodes "
            "which return either qml.expval or qml.probs."
        )

    if f.diff_method == "adjoint" and any(prim not in [expval_p] for prim in return_ops):
        raise DifferentiableCompileError(
            "The adjoint method can only be used for QNodes which return qml.expval."
        )
