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
import pennylane as qp
from jax._src.tree_util import tree_flatten

from catalyst.api_extensions.rem_postprocessing import (
    rem_apply_to_counts,
    rem_apply_to_probs,
    rem_apply_to_samples,
    rem_calibrate_counts,
    rem_calibrate_probs,
    rem_calibrate_samples,
)
from catalyst.jax_primitives import Folding, func_p, quantum_kernel_p, rem_p, zne_p
from catalyst.jax_tracer import Function
from catalyst.utils.callables import CatalystCallable


def _check_is_odd_positive(numbers_list):
    for n in numbers_list:
        if not isinstance(n, int):
            msg = f"Found non-integer {n} in scale_factors {numbers_list}.\n"
            msg += "Only odd positive integers are allowed in scale_factors"
            raise TypeError(msg)
        if n < 0:
            msg = "Found negative number {n} in scale_factors {numbers_list}.\n"
            msg += "Only odd positive integers are allowed in scale_factors"
            raise ValueError(msg)
        if n % 2 == 0:
            msg = f"Found even positive {n} in scale_factors {numbers_list}.\n"
            msg += "Only odd positive integers are allowed in scale_factors"
            raise ValueError(msg)


## API ##
def mitigate_with_zne(
    fn=None, *, scale_factors, extrapolate=None, extrapolate_kwargs=None, folding="global"
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
        fn (qp.QNode): the circuit to be mitigated.
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

        import pennylane as qp
        from catalyst import qjit, for_loop, mitigate_with_zne

        # replace "noisy.device" with your noisy device
        dev = qp.device("noisy.device", wires=2)

        @qp.qnode(device=dev)
        def circuit(x, n):
            @for_loop(0, n, 1)
            def loop_rx(i):
                qp.RX(x, wires=0)

            loop_rx()

            qp.Hadamard(wires=0)
            qp.RZ(x, wires=0)
            loop_rx()
            qp.RZ(x, wires=0)
            qp.CNOT(wires=[1, 0])
            qp.Hadamard(wires=1)
            return qp.expval(qp.PauliY(wires=0))

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

        from pennylane.noise import exponential_extrapolate

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev, shots=100000)
        def circuit(weights):
            qp.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qp.expval(qp.PauliZ(0) @ qp.PauliZ(1))

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

    _check_is_odd_positive(scale_factors)

    return ZNECallable(fn, scale_factors, extrapolate, folding)


def mitigate_with_rem(
    fn=None, *, calibration_matrices=None, return_calibration_matrices: bool = False
):
    """A qjit-compatible frontend that emits a `mitigation.rem` operation.

    When ``calibration_matrices`` is ``None`` the lowering pass emits the
    all-zeros and all-ones calibration circuits and this callable computes
    the confusion matrices itself. When a precomputed ``(n_qubits, 2, 2)``
    matrix stack is supplied, ``rem_calibrate_*`` is skipped and the matrices
    are passed straight into ``rem_apply_to_*``. Pass
    ``return_calibration_matrices=True`` to get the matrices back alongside
    the mitigated result so they can be cached for subsequent calls.
    """

    kwargs = copy.copy(locals())
    kwargs.pop("fn")

    if fn is None:
        return functools.partial(mitigate_with_rem, **kwargs)

    return RemCallable(
        fn,
        calibration_matrices=calibration_matrices,
        return_calibration_matrices=return_calibration_matrices,
    )


## IMPL ##
class ZNECallable(CatalystCallable):
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
        scale_factors: Sequence[int],
        extrapolate: Callable[[Sequence[float], Sequence[float]], float],
        folding: str,
    ):
        functools.update_wrapper(self, fn)
        self.fn = fn
        self.__name__ = f"zne.{getattr(fn, '__name__', 'unknown')}"
        self.scale_factors = scale_factors
        self.extrapolate = extrapolate
        self.folding = folding

        super().__init__("fn")

    def __call__(self, *args, **kwargs):
        """Specifies the an actual call to the folded circuit."""
        callable_fn = _wrap_callable(self.fn)
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
        assert jaxpr.eqns, "expected non-empty jaxpr for zne target"
        assert jaxpr.eqns[0].primitive in {
            func_p,
            quantum_kernel_p,
        }, "expected func_p or quantum_kernel_p as first operation in zne target"
        callable_fn = jaxpr.eqns[0].params.get("fn", callable_fn)
        assert callable(
            callable_fn
        ), "expected callable set as param on the first operation in zne target"

        fold_numbers = (jnp.asarray(self.scale_factors, dtype=int) - 1) // 2
        fold_results = zne_p.bind(
            *args_data, fold_numbers, folding=folding, jaxpr=jaxpr, fn=callable_fn
        )

        scale_factors = jnp.asarray(self.scale_factors, dtype=float)
        zne_results = self.extrapolate(scale_factors, fold_results)

        # if multiple measurement processes, split array back into tuple
        if len(zne_results.shape):
            zne_results = tuple(zne_results)
        return zne_results


class RemCallable(CatalystCallable):
    """An object that specifies how a circuit is mitigated with REM.

    Args:
        fn (Callable): the circuit to be mitigated with REM.
        calibration_matrices: optional precomputed per-qubit confusion stack
            of shape ``(n_qubits, 2, 2)``. When ``None`` the lowering pass
            emits the all-zeros and all-ones calibration circuits and this
            callable computes the matrices from their outputs via
            ``rem_calibrate_*``. When provided, ``rem_calibrate_*`` is not
            traced and the supplied matrices are passed straight into
            ``rem_apply_to_*``.
        return_calibration_matrices (bool): when ``True`` the callable
            returns ``(mitigated_result, calibration_matrices)`` instead of
            just the mitigated result.

    Raises:
        TypeError: Non-QNode object was passed as ``fn``.
    """

    def __init__(
        self,
        fn: Callable,
        calibration_matrices=None,
        return_calibration_matrices: bool = False,
    ):
        functools.update_wrapper(self, fn)
        self.fn = fn
        self.__name__ = f"rem.{getattr(fn, '__name__', 'unknown')}"
        self.calibration_matrices = calibration_matrices
        self.return_calibration_matrices = bool(return_calibration_matrices)
        super().__init__("fn")

    def __call__(self, *args, **kwargs):
        callable_fn = _wrap_callable(self.fn)
        jaxpr = jax.make_jaxpr(callable_fn)(*args)

        args_data, _ = tree_flatten(args)

        assert jaxpr.eqns, "expected non-empty jaxpr for rem target"
        assert jaxpr.eqns[0].primitive in {
            func_p,
            quantum_kernel_p,
        }, "expected func_p or quantum_kernel_p as first operation in rem target"
        callable_fn = jaxpr.eqns[0].params.get("fn", callable_fn)
        assert callable(
            callable_fn
        ), "expected callable set as param on the first operation in rem target"

        mp_kind = _detect_measurement_process(jaxpr)
        assert mp_kind is not None, (
            "measurement process must be one of CountsMP, ProbsMP or SampleMP. "
            "Other measurement processes such as observables are not supported yet."
        )

        run_calibration = self.calibration_matrices is None
        rem_results = rem_p.bind(
            *args_data,
            run_calibration=run_calibration,
            jaxpr=jaxpr,
            fn=callable_fn,
        )

        qnode_obj = jaxpr.eqns[0].params.get("qnode", None)
        assert qnode_obj is not None, "REM post-processing requires a QNode target"
        n_qubits = len(qnode_obj.device.wires)
        measured_qubits = jnp.array(list(qnode_obj.device.wires))

        handler = {
            "sample": self._apply_sample,
            "counts": self._apply_counts,
            "probs": self._apply_probs,
        }[mp_kind]
        mitigated_result, confusion_matrices = handler(
            rem_results, run_calibration, measured_qubits, n_qubits
        )

        if self.return_calibration_matrices:
            return mitigated_result, confusion_matrices
        return mitigated_result

    def _apply_sample(self, rem_results, run_calibration, measured_qubits, n_qubits):
        # quantum.sample returns f64 in catalyst; the REM helpers index
        # confusion matrices by bit value and need an integer dtype.
        mitigatee_samples = rem_results[0].astype(jnp.int32)
        if run_calibration:
            zeros_samples = rem_results[1].astype(jnp.int32)
            ones_samples = rem_results[2].astype(jnp.int32)
            confusion_matrices = rem_calibrate_samples(zeros_samples, ones_samples)
        else:
            confusion_matrices = jnp.asarray(self.calibration_matrices)
        unique_bitstrings, mitigated_counts = rem_apply_to_samples(
            mitigatee_samples, confusion_matrices, measured_qubits, n_qubits
        )
        return (unique_bitstrings, mitigated_counts), confusion_matrices

    def _apply_counts(self, rem_results, run_calibration, measured_qubits, n_qubits):
        mitigatee_eigvals = rem_results[0]
        mitigatee_counts = rem_results[1]
        if run_calibration:
            zeros_counts = rem_results[2]
            ones_counts = rem_results[3]
            confusion_matrices = rem_calibrate_counts(zeros_counts, ones_counts)
        else:
            confusion_matrices = jnp.asarray(self.calibration_matrices)
        mitigated_counts = rem_apply_to_counts(
            mitigatee_counts, confusion_matrices, measured_qubits, n_qubits
        )
        return (mitigatee_eigvals, mitigated_counts), confusion_matrices

    def _apply_probs(self, rem_results, run_calibration, measured_qubits, n_qubits):
        mitigatee_probs = rem_results[0]
        if run_calibration:
            zeros_probs = rem_results[1]
            ones_probs = rem_results[2]
            confusion_matrices = rem_calibrate_probs(zeros_probs, ones_probs)
        else:
            confusion_matrices = jnp.asarray(self.calibration_matrices)
        mitigated_probs = rem_apply_to_probs(
            mitigatee_probs, confusion_matrices, measured_qubits, n_qubits
        )
        return mitigated_probs, confusion_matrices


def polynomial_extrapolation(degree):
    """utility to generate polynomial fitting functions of arbitrary degree"""
    return functools.partial(qp.noise.poly_extrapolate, order=degree)


## PRIVATE ##
def _wrap_callable(fn):
    if isinstance(fn, (Function, qp.QNode)):
        return fn
    elif isinstance(fn, Callable):  # Keep at the bottom
        return Function(fn)
    raise TypeError(f"Target must be callable, got: {type(fn)}")


def _detect_measurement_process(jaxpr):
    """Walk the inner jaxpr for a probs/sample/counts primitive; return its name or None.

    Mirrors the detection logic in :func:`_rem_abstract_eval`.
    """
    for eqn in jaxpr.eqns:
        inner = eqn.params.get("call_jaxpr", None)
        if inner is None:
            continue
        for op_eq in inner.eqns:
            pname = str(op_eq.primitive)
            if pname in ("probs", "sample", "counts"):
                return pname
    return None
