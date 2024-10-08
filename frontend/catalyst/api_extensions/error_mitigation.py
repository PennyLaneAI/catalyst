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
This module contains public API functions that provide error mitigation
capabilities for quantum programs. Error mitigation techniques improve the
reliability of noisy quantum computers without relying on error correction.
"""

import copy
import functools
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import pennylane as qml
from jax._src.tree_util import tree_flatten

from catalyst.jax_primitives import Folding, zne_p
from catalyst.jax_tracer import Function


def _is_odd_positive(numbers_list):
    return all(isinstance(i, int) and i > 0 and i % 2 != 0 for i in numbers_list)


## API ##
def mitigate_with_zne(
    fn=None, *, scale_factors=None, extrapolate=None, extrapolate_kwargs=None, folding="global"
):
    """A :func:`~.qjit` compatible error mitigation of an input circuit using zero-noise
    extrapolation.

    Error mitigation is a precursor to error correction and is compatible with near-term quantum
    devices. It aims to lower the impact of noise when evaluating a circuit on a quantum device by
    evaluating multiple variations of the circuit and post-processing the results into a
    noise-reduced estimate. This transform implements the zero-noise extrapolation (ZNE) method
    originally introduced by
    `Temme et al. <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.180509>`__ and
    `Li et al. <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.7.021050>`__.

    Args:
        fn (qml.QNode): the circuit to be mitigated.
        scale_factors (list[int]): the range of noise scale factors used.
        extrapolate (Callable): A qjit-compatible function taking two sequences as arguments (scale
            factors, and results), and returning a float by performing a fitting procedure.
            By default, perfect polynomial fitting :func:`~.polynomial_extrapolate` will be used,
            the :func:`~.exponential_extrapolate` function from PennyLane may also be used.
        extrapolate_kwargs (dict[str, Any]): Keyword arguments to be passed to the extrapolation
            function.
        folding (str): Unitary folding technique to be used to scale the circuit. Possible values:
            - global: the global unitary of the input circuit is folded
            - local-all: per-gate folding sequences replace original gates in-place in the circuit

    Returns:
        Callable: A callable object that computes the mitigated of the wrapped :class:`~.QNode`
        for the given arguments.

    **Example:**

    For example, given a noisy device (such as noisy hardware available through Amazon Braket):

    .. code-block:: python

        # replace "noisy.device" with your noisy device
        dev = qml.device("noisy.device", wires=2)

        @qml.qnode(device=dev)
        def circuit(x, n):
            @for_loop(0, n, 1)
            def loop_rx(i):
                qml.RX(x, wires=0)

            loop_rx()

            qml.Hadamard(wires=0)
            qml.RZ(x, wires=0)
            loop_rx()
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[1, 0])
            qml.Hadamard(wires=1)
            return qml.expval(qml.PauliY(wires=0))

        @qjit
        def mitigated_circuit(args, n):
            s = [1, 3, 5]
            return mitigate_with_zne(circuit, scale_factors=s)(args, n)

    Alternatively the `mitigate_with_zne` function can be applied directly on a qjitted
    function containing :class:`~.QNode`, the mitigation will be applied on each
    :class:`~.QNode` individually.

    Exponential extrapolation can also be performed via the
    :func:`~.exponential_extrapolate` function from PennyLane:

    .. code-block:: python

        from pennylane.transforms import exponential_extrapolate

        dev = qml.device("lightning.qubit", wires=2, shots=100000)

        @qml.qnode(dev)
        def circuit(weights):
            qml.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        @qjit
        def workflow(weights, s):
            zne_circuit = mitigate_with_zne(
                circuit, scale_factors=s, extrapolate=exponential_extrapolate
            )
            return zne_circuit(weights)

    >>> weights = jnp.ones([3, 2, 3])
    >>> scale_factors = [1, 3, 5]
    >>> workflow(weights, scale_factors)
    Array(-0.19946598, dtype=float64)
    """

    kwargs = copy.copy(locals())
    kwargs.pop("fn")

    if fn is None:
        return functools.partial(mitigate_with_zne, **kwargs)

    if extrapolate is None:
        extrapolate = polynomial_extrapolation(len(scale_factors) - 1)
    elif extrapolate_kwargs is not None:
        extrapolate = functools.partial(extrapolate, **extrapolate_kwargs)

    if not _is_odd_positive(scale_factors):
        raise ValueError("The scale factors must be positive odd integers: {scale_factors}")

    num_folds = jnp.array([jnp.floor((s - 1) / 2) for s in scale_factors], dtype=int)

    return ZNE(fn, num_folds, extrapolate, folding)


## IMPL ##


def _make_function(fn):
    if isinstance(fn, (Function, qml.QNode)):
        return fn
    elif isinstance(fn, Callable):  # Keep at the bottom
        return Function(fn)


class ZNE:
    """An object that specifies how a circuit is mitigated with ZNE.

    Args:
        fn (Callable): the circuit to be mitigated with ZNE.
        scale_factors (array[int]): the range of noise scale factors used.
        deg (int): the degree of the polymonial used for fitting.

    Raises:
        TypeError: Non-QNode object was passed as `fn`.
    """

    def __init__(
        self,
        fn: Callable,
        num_folds: jnp.ndarray,
        extrapolate: Callable[[Sequence[float], Sequence[float]], float],
        folding: str,
    ):
        self.fn = fn
        self.__name__ = f"zne.{getattr(fn, '__name__', 'unknown')}"
        self.num_folds = num_folds
        self.extrapolate = extrapolate
        self.folding = folding

    def __call__(self, *args, **kwargs):
        """Specifies the an actual call to the folded circuit."""
        callable_fn = _make_function(self.fn)
        jaxpr = jax.make_jaxpr(callable_fn)(*args)
        shapes = [out_val.shape for out_val in jaxpr.out_avals]
        dtypes = [out_val.dtype for out_val in jaxpr.out_avals]
        set_dtypes = set(dtypes)
        if any(shapes):
            raise TypeError("Only expectations values and classical scalar values can be returned.")
        if len(set_dtypes) != 1 or set_dtypes.pop().kind != "f":
            raise TypeError("All expectation and classical values dtypes must match and be float.")
        args_data, _ = tree_flatten(args)
        try:
            folding = Folding(self.folding)
        except ValueError as e:
            raise ValueError(f"Folding type must be one of {list(map(str, Folding))}") from e
        # TODO: remove the following check once #755 is completed
        if folding == Folding.RANDOM:
            raise NotImplementedError(f"Folding type {folding.value} is being developed")

        # Certain callables, like QNodes, may introduce additional wrappers during tracing.
        # Make sure to grab the top-level callable object in the traced function.
        callable_fn = jaxpr.eqns[0].params.get("fn", callable_fn)
        assert callable(callable_fn)

        results = zne_p.bind(
            *args_data, self.num_folds, folding=folding, jaxpr=jaxpr, fn=callable_fn
        )
        float_num_folds = jnp.array(self.num_folds, dtype=float)
        results = self.extrapolate(float_num_folds, results[0])
        # Single measurement
        if results.shape == ():
            return results
        # Multiple measurements
        return tuple(res for res in results)


def polynomial_extrapolation(degree):
    """utility to generate polynomial fitting functions of arbitrary degree"""
    return functools.partial(qml.transforms.poly_extrapolate, order=degree)
