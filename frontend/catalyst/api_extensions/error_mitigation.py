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
from typing import Callable

import jax
import jax.numpy as jnp
import pennylane as qml
from jax._src.tree_util import tree_flatten

from catalyst.jax_primitives import zne_p


## API ##
def mitigate_with_zne(fn=None, *, scale_factors=None, deg=None):
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
        scale_factors (array[int]): the range of noise scale factors used.
        deg (int): the degree of the polymonial used for fitting.

    Returns:
        Callable: A callable object that computes the mitigated of the wrapped :class:`qml.QNode`
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
            s = jax.numpy.array([1, 2, 3])
            return mitigate_with_zne(circuit, scale_factors=s)(args, n)
    """
    kwargs = copy.copy(locals())
    kwargs.pop("fn")

    if fn is None:
        return functools.partial(mitigate_with_zne, **kwargs)

    if deg is None:
        deg = len(scale_factors) - 1
    return ZNE(fn, scale_factors, deg)


## IMPL ##
class ZNE:
    """An object that specifies how a circuit is mitigated with ZNE.

    Args:
        fn (Callable): the circuit to be mitigated with ZNE.
        scale_factors (array[int]): the range of noise scale factors used.
        deg (int): the degree of the polymonial used for fitting.

    Raises:
        TypeError: Non-QNode object was passed as `fn`.
    """

    def __init__(self, fn: Callable, scale_factors: jnp.ndarray, deg: int):
        if not isinstance(fn, qml.QNode):
            raise TypeError(f"A QNode is expected, got the classical function {fn}")
        self.fn = fn
        self.__name__ = f"zne.{getattr(fn, '__name__', 'unknown')}"
        self.scale_factors = scale_factors
        self.deg = deg

    def __call__(self, *args, **kwargs):
        """Specifies the an actual call to the folded circuit."""
        jaxpr = jaxpr = jax.make_jaxpr(self.fn)(*args)
        shapes = [out_val.shape for out_val in jaxpr.out_avals]
        dtypes = [out_val.dtype for out_val in jaxpr.out_avals]
        set_dtypes = set(dtypes)
        if any(shapes):
            raise TypeError("Only expectations values and classical scalar values can be returned.")
        if len(set_dtypes) != 1 or set_dtypes.pop().kind != "f":
            raise TypeError("All expectation and classical values dtypes must match and be float.")
        args_data, _ = tree_flatten(args)
        results = zne_p.bind(*args_data, self.scale_factors, jaxpr=jaxpr, fn=self.fn)
        float_scale_factors = jnp.array(self.scale_factors, dtype=float)
        results = jnp.polyfit(float_scale_factors, results[0], self.deg)[-1]
        # Single measurement
        if results.shape == ():
            return results
        # Multiple measurements
        return tuple(res for res in results)
