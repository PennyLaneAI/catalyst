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
"""This module contains the preprocessing functions.
"""
import jax
import pennylane as qml
from pennylane import transform
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MidMeasureMP,
    ProbabilityMP,
    VarianceMP,
)
from pennylane.tape.tape import (
    _validate_computational_basis_sampling,
    rotations_and_diagonal_measurements,
)

import catalyst
import catalyst.pennylane_extensions
from catalyst.utils.exceptions import CompileError


def _operator_decomposition_gen(
    op: qml.operation.Operator,
    acceptance_function,
    decomposer,
    max_expansion=None,
    current_depth=0,
):
    """A generator that yields the next operation that is accepted."""
    max_depth_reached = False
    if max_expansion is not None and max_expansion <= current_depth:
        max_depth_reached = True
    if acceptance_function(op) or max_depth_reached:
        yield op
    else:
        try:
            decomp = decomposer(op)
            current_depth += 1
        except qml.operation.DecompositionUndefinedError as e:
            raise CompileError(
                f"Operator {op} not supported on device and does not provide a decomposition."
            ) from e

        for sub_op in decomp:
            yield from _operator_decomposition_gen(
                sub_op,
                acceptance_function,
                decomposer=decomposer,
                max_expansion=max_expansion,
                current_depth=current_depth,
            )


@transform
def decompose(
    tape: qml.tape.QuantumTape,
    stopping_condition,
    max_expansion=None,
):
    """Decompose operations until the stopping condition is met."""

    def decomposer(op):
        if op.name in {"MultiControlledX", "BlockEncode"} or isinstance(op, qml.ops.Controlled):
            try:
                mat = op.matrix()
            except Exception as e:
                raise CompileError(
                    f"Operation {op} could not be decomposed, it might be unsupported."
                ) from e
            op = qml.QubitUnitary(mat, wires=op.wires)
            return [op]
        elif isinstance(
            op,
            (
                catalyst.pennylane_extensions.Adjoint,
                catalyst.pennylane_extensions.MidCircuitMeasure,
                catalyst.pennylane_extensions.ForLoop,
                catalyst.pennylane_extensions.WhileLoop,
                catalyst.pennylane_extensions.Cond,
            ),
        ):
            for r in op.regions:
                if r.quantum_tape:
                    tapes, _ = decompose(
                        r.quantum_tape,
                        stopping_condition=stopping_condition,
                        max_expansion=max_expansion,
                    )
                    r.quantum_tape = tapes[0]
            op.visited = True
            return [op]
        return op.decomposition()

    if len(tape) == 0:
        return (tape,), lambda x: x[0]

    new_ops = []
    for op in tape.operations:
        if isinstance(op, MidMeasureMP):
            raise CompileError("Must use 'measure' from Catalyst instead of PennyLane.")
        new_ops.extend(
            op
            for op in _operator_decomposition_gen(
                op,
                stopping_condition,
                decomposer=decomposer,
                max_expansion=max_expansion,
            )
        )
    tape = qml.tape.QuantumScript(new_ops, tape.measurements, shots=tape.shots)

    return (tape,), lambda x: x[0]


@transform
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
        if op.name in convert_to_matrix_ops or isinstance(op, catalyst.pennylane_extensions.QCtrl):
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


def catalyst_acceptance(op: qml.operation.Operator, operations) -> bool:
    """Specify whether or not an Operator is supported."""
    if isinstance(
        op,
        (
            catalyst.pennylane_extensions.Adjoint,
            catalyst.pennylane_extensions.MidCircuitMeasure,
            catalyst.pennylane_extensions.ForLoop,
            catalyst.pennylane_extensions.WhileLoop,
            catalyst.pennylane_extensions.Cond,
        ),
    ):
        return op.name in operations and op.visited

    return op.name in operations


@transform
def measurements_from_counts(tape):
    r"""Replace all measurements from a tape with a single count measurement, it adds postprocessing
    functions for each original measurement.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The
        transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    .. note::

        Samples are not supported.
    """
    if tape.samples_computational_basis and len(tape.measurements) > 1:
        _validate_computational_basis_sampling(tape.measurements)
    diagonalizing_gates, diagonal_measurements = rotations_and_diagonal_measurements(tape)
    for i, m in enumerate(diagonal_measurements):
        if m.obs is not None:
            diagonalizing_gates.extend(m.obs.diagonalizing_gates())
            diagonal_measurements[i] = type(m)(eigvals=m.eigvals(), wires=m.wires)
    # Add diagonalizing gates
    news_operations = tape.operations
    news_operations.extend(diagonalizing_gates)
    # Transform tape
    measured_wires = set()
    for m in diagonal_measurements:
        measured_wires.update(m.wires.tolist())

    new_measurements = [qml.counts(wires=list(measured_wires))]
    new_tape = type(tape)(news_operations, new_measurements, shots=tape.shots)

    def postprocessing_counts_to_expval(results):
        """A processing function to get expecation values from counts."""
        states = results[0][0]
        counts_outcomes = results[0][1]
        results_processed = []
        for m in tape.measurements:
            mapped_counts_outcome = _map_counts(
                counts_outcomes, m.wires, qml.wires.Wires(list(measured_wires))
            )
            if isinstance(m, ExpectationMP):
                probs = _get_probs(mapped_counts_outcome)
                results_processed.append(_get_expval(eigvals=m.eigvals(), prob_vector=probs))
            elif isinstance(m, VarianceMP):
                probs = _get_probs(mapped_counts_outcome)
                results_processed.append(_get_var(eigvals=m.eigvals(), prob_vector=probs))
            elif isinstance(m, ProbabilityMP):
                probs = _get_probs(mapped_counts_outcome)
                results_processed.append(probs)
            elif isinstance(m, CountsMP):
                results_processed.append(
                    tuple([states[0 : 2 ** len(m.wires)], mapped_counts_outcome])
                )
        if len(tape.measurements) == 1:
            results_processed = results_processed[0]
        else:
            results_processed = tuple(results_processed)
        return results_processed

    return [new_tape], postprocessing_counts_to_expval


def _get_probs(counts_outcome):
    """From the counts outcome, calculate the probability vector."""
    prob_vector = []
    num_shots = jax.numpy.sum(counts_outcome)
    for count in counts_outcome:
        prob = count / num_shots
        prob_vector.append(prob)
    return jax.numpy.array(prob_vector)


def _get_expval(eigvals, prob_vector):
    """From the observable eigenvalues and the probability vector
    it calculates the expectation value."""
    expval = jax.numpy.dot(jax.numpy.array(eigvals), prob_vector)
    return expval


def _get_var(eigvals, prob_vector):
    """From the observable eigenvalues and the probability vector
    it calculates the variance."""
    var = jax.numpy.dot(prob_vector, (eigvals**2)) - jax.numpy.dot(prob_vector, eigvals)
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
