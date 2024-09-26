# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This module contains the decomposition functions to pre-process tapes for
compilation & execution on devices.
"""
import copy
import logging
from functools import partial

import jax
import pennylane as qml
from pennylane import transform
from pennylane.devices.preprocess import decompose
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MidMeasureMP,
    ProbabilityMP,
    SampleMP,
    VarianceMP,
)

from catalyst.api_extensions import HybridCtrl
from catalyst.jax_tracer import HybridOpRegion, has_nested_tapes
from catalyst.logging import debug_logger
from catalyst.tracing.contexts import EvaluationContext
from catalyst.utils.exceptions import CompileError
from catalyst.utils.toml import DeviceCapabilities

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def check_alternative_support(op, capabilities):
    """Verify that aliased operations aren't supported via alternative definitions."""

    if isinstance(op, qml.ops.Controlled):
        # "Cast" away the specialized class for gates like Toffoli, ControlledQubitUnitary, etc.
        supported = capabilities.native_ops.get(op.base.name)
        if supported and supported.controllable:
            return [qml.ops.Controlled(op.base, op.control_wires, op.control_values, op.work_wires)]

    return None


def catalyst_decomposer(op, capabilities: DeviceCapabilities):
    """A decomposer for catalyst, to be passed to the decompose transform. Takes an operator and
    returns the default decomposition, unless the operator should decompose to a QubitUnitary.
    Raises a CompileError for MidMeasureMP"""
    if isinstance(op, MidMeasureMP):
        raise CompileError("Must use 'measure' from Catalyst instead of PennyLane.")

    alternative_decomp = check_alternative_support(op, capabilities)
    if alternative_decomp is not None:
        return alternative_decomp

    if capabilities.native_ops.get("QubitUnitary"):
        # If the device supports unitary matrices, apply the relevant conversions and fallbacks.
        if op.name in capabilities.to_matrix_ops or (
            op.has_matrix and isinstance(op, qml.ops.Controlled)
        ):
            return _decompose_to_matrix(op)

    return op.decomposition()


@transform
@debug_logger
def catalyst_decompose(tape: qml.tape.QuantumTape, ctx, capabilities):
    """Decompose operations until the stopping condition is met.

    In a single call of the catalyst_decompose function, the PennyLane operations are decomposed
    in the same manner as in PennyLane (for each operator on the tape, checking if the operator
    passes the stopping_condition, and using its `decompostion` method if not, called recursively
    until a supported operation is found or an error is hit, then moving on to the next operator
    on the tape.)

    Once all operators on the tape are supported operators, the resulting tape is iterated over,
    and for each HybridOp, the catalyst_decompose function is called on each of it's regions.
    This continues to call catalyst_decompose recursively until the tapes on all
    the HybridOps have been passed to the decompose function.
    """

    (toplevel_tape,), _ = decompose(
        tape,
        stopping_condition=lambda op: bool(catalyst_acceptance(op, capabilities)),
        skip_initial_state_prep=capabilities.initial_state_prep_flag,
        decomposer=partial(catalyst_decomposer, capabilities=capabilities),
        name="catalyst on this device",
        error=CompileError,
    )

    new_ops = []
    for op in toplevel_tape.operations:
        if has_nested_tapes(op):
            op = _decompose_nested_tapes(op, ctx, capabilities)
        new_ops.append(op)
    tape = qml.tape.QuantumScript(new_ops, tape.measurements, shots=tape.shots)

    return (tape,), lambda x: x[0]


def _decompose_to_matrix(op):
    try:
        mat = op.matrix()
    except Exception as e:
        raise CompileError(
            f"Operation {op} could not be decomposed, it might be unsupported."
        ) from e
    op = qml.QubitUnitary(mat, wires=op.wires)
    return [op]


def _decompose_nested_tapes(op, ctx, capabilities):
    new_regions = []
    for region in op.regions:
        if region.quantum_tape is None:
            new_tape = None
        else:
            with EvaluationContext.frame_tracing_context(ctx, region.trace):
                tapes, _ = catalyst_decompose(
                    region.quantum_tape, ctx=ctx, capabilities=capabilities
                )
                new_tape = tapes[0]
        new_regions.append(
            HybridOpRegion(
                region.trace, new_tape, region.arg_classical_tracers, region.res_classical_tracers
            )
        )

    new_op = copy.copy(op)
    new_op.regions = new_regions
    # new_op.apply_reverse_transform=op.apply_reverse_transform,
    # new_op.expansion_strategy=op.expansion_strategy,
    return new_op


@transform
@debug_logger
def decompose_ops_to_unitary(tape, convert_to_matrix_ops):
    r"""Quantum transform that decomposes operations to unitary given a list of operations name.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        convert_to_matrix_ops (list[str]): The list of operation names to be converted to unitary.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The
        transformed circuit as described in :func:`qml.transform <pennylane.transform>`.
    """
    new_operations = []

    for op in tape.operations:
        if op.name in convert_to_matrix_ops or isinstance(op, HybridCtrl):
            try:
                mat = op.matrix()
            except Exception as e:
                raise CompileError(
                    f"Operation {op} could not be decomposed, it might be unsupported."
                ) from e
            op = qml.QubitUnitary(mat, wires=op.wires)
        new_operations.append(op)

    new_tape = type(tape)(new_operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing


def catalyst_acceptance(op: qml.operation.Operation, capabilities: DeviceCapabilities) -> str:
    """Check whether or not an Operator is supported."""
    op_support = capabilities.native_ops

    if isinstance(op, qml.ops.Adjoint):
        match = catalyst_acceptance(op.base, capabilities)
        if not match or not op_support[match].invertible:
            return None
    elif type(op) is qml.ops.ControlledOp:
        match = catalyst_acceptance(op.base, capabilities)
        if not match or not op_support[match].controllable:
            return None
    else:
        match = op.name if op.name in op_support else None

    return match


@transform
@debug_logger
def measurements_from_counts(tape, device_wires):
    r"""Replace all measurements from a tape with a single count measurement, it adds postprocessing
    functions for each original measurement.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        device_wires (Wires): the wires from the device, for assessing what wires to use
            when a measurement applies to all wires.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The
        transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    .. note::

        Samples are not supported.
    """

    new_operations, measured_wires = _diagonalize_measurements(tape, device_wires=device_wires)

    new_tape = type(tape)(new_operations, [qml.counts(wires=measured_wires)], shots=tape.shots)

    def postprocessing_counts(results):
        """A processing function to get expecation values from counts."""
        states = results[0][0]
        counts_outcomes = results[0][1]
        results_processed = []
        for m in tape.measurements:
            wires = m.wires if m.wires else device_wires
            mapped_counts_outcome = _map_counts(
                counts_outcomes, wires, qml.wires.Wires(list(measured_wires))
            )
            if isinstance(m, ExpectationMP):
                probs = _probs_from_counts(mapped_counts_outcome)
                results_processed.append(_expval_from_probs(eigvals=m.eigvals(), prob_vector=probs))
            elif isinstance(m, VarianceMP):
                probs = _probs_from_counts(mapped_counts_outcome)
                results_processed.append(_var_from_probs(eigvals=m.eigvals(), prob_vector=probs))
            elif isinstance(m, ProbabilityMP):
                probs = _probs_from_counts(mapped_counts_outcome)
                results_processed.append(probs)
            elif isinstance(m, CountsMP):
                results_processed.append(
                    tuple([states[0 : 2 ** len(m.wires)], mapped_counts_outcome])
                )
            else:
                raise NotImplementedError(
                    f"Measurement type {type(m)} is not implemented with measurements_from_counts"
                )
        if len(tape.measurements) == 1:
            results_processed = results_processed[0]
        else:
            results_processed = tuple(results_processed)
        return results_processed

    return [new_tape], postprocessing_counts


@transform
@debug_logger
def measurements_from_samples(tape, device_wires):
    r"""Replace all measurements from a tape with sample measurements, and adds postprocessing
    functions for each original measurement.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The
        transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    .. note::

        Counts are not supported.
    """

    new_operations, measured_wires = _diagonalize_measurements(tape, device_wires=device_wires)

    new_tape = type(tape)(new_operations, [qml.sample(wires=measured_wires)], shots=tape.shots)

    def postprocessing_samples(results):
        """A processing function to get expecation values from samples."""
        samples = results[0]
        results_processed = []
        for m in tape.measurements:
            if isinstance(m, (ExpectationMP, VarianceMP, ProbabilityMP, SampleMP)):
                if len(tape.shots.shot_vector) > 1:
                    res = tuple(m.process_samples(s, measured_wires) for s in samples)
                else:
                    res = m.process_samples(samples, measured_wires)
                results_processed.append(res)
            else:
                raise NotImplementedError(
                    f"Measurement type {type(m)} is not implemented with measurements_from_samples"
                )
        if len(tape.measurements) == 1:
            results_processed = results_processed[0]
        else:
            results_processed = tuple(results_processed)
        return results_processed

    return [new_tape], postprocessing_samples


def _diagonalize_measurements(tape, device_wires):
    """Takes a tape and returns the information needed to create a new tape based on
    diagonalization and readout in the measurement basis.

    Args:
        tape (QuantumTape): A quantum circuit.

    Returns:
        new_operations (list): The original operations, plus the diagonalizing gates for the circuit
        measured_wires (list): A list of all wires that are measured on the tape

    """

    (diagonalized_tape,), _ = qml.transforms.diagonalize_measurements(tape)

    measured_wires = set()
    for m in diagonalized_tape.measurements:
        wires = m.wires if m.wires else device_wires
        measured_wires.update(wires.tolist())

    return diagonalized_tape.operations, list(measured_wires)


def _probs_from_counts(counts_outcome):
    """From the counts outcome, calculate the probability vector."""
    prob_vector = []
    num_shots = jax.numpy.sum(counts_outcome)
    for count in counts_outcome:
        prob = count / num_shots
        prob_vector.append(prob)
    return jax.numpy.array(prob_vector)


def _expval_from_probs(eigvals, prob_vector):
    """From the observable eigenvalues and the probability vector
    it calculates the expectation value."""
    expval = jax.numpy.dot(jax.numpy.array(eigvals), prob_vector)
    return expval


def _var_from_probs(eigvals, prob_vector):
    """From the observable eigenvalues and the probability vector
    it calculates the variance."""
    var = jax.numpy.dot(prob_vector, (eigvals**2)) - jax.numpy.dot(prob_vector, eigvals) ** 2
    return var


def _map_counts(counts, sub_wires, wire_order):
    """Map the count outcome given a wires and wire order."""
    wire_map = dict(zip(wire_order, range(len(wire_order))))
    mapped_wires = [wire_map[w] for w in sub_wires]

    mapped_counts = {}
    num_wires = len(wire_order)
    for outcome, occurrence in enumerate(counts):
        binary_outcome = format(outcome, f"0{num_wires}b")
        mapped_outcome = "".join(binary_outcome[i] for i in mapped_wires)
        mapped_counts[mapped_outcome] = mapped_counts.get(mapped_outcome, 0) + occurrence

    return jax.numpy.array(list(mapped_counts.values()))
