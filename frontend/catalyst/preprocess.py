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

import catalyst
from pennylane.measurements import ExpectationMP, ProbabilityMP
from catalyst.utils.exceptions import CompileError


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
    return op.name in operations


@transform
def expval_from_counts(tape):
    r"""Replace expval from a tape with counts and postprocessing.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        convert_to_matrix_ops (list[str]): The list of operation names to be converted to unitary.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The
        transformed circuit as described in :func:`qml.transform <pennylane.transform>`.
    """
    # Add diagonalizing gates
    news_operations = tape.operations
    expval_position = [False * len(tape.measurements)]
    expval_eigvals = [False * len(tape.measurements)]
    for i, m in enumerate(tape.measurements):
        if m.obs:
            news_operations.extend(m.obs.diagonalizing_gates())
        if isinstance(m, ExpectationMP):
            expval_position[i] = True
            expval_eigvals[i] = m.eigvals()
    # Transform tape
    new_measurements = [
        qml.counts(wires=m.obs.wires.tolist()) if isinstance(m, ExpectationMP) else m
        for m in tape.measurements
    ]
    new_tape = type(tape)(news_operations, new_measurements, shots=tape.shots)

    def postprocessing_counts_to_expval(results):
        """A processing function to get expecation values from counts."""
        processed_results = []
        for r in results:
            process_tape_results(r)
        return processed_results

    return [new_tape], postprocessing_counts_to_expval

def process_results():
    for i, is_expval in enumerate(expval_position):
        if is_expval:
            prob_vector = []
            _, values = r[i]
            num_shots = jax.numpy.sum(values)
            for value in values:
                prob = value / num_shots
                prob_vector.append(prob)
            expval = jax.numpy.dot(
                jax.numpy.array(expval_eigvals[i]), jax.numpy.array(prob_vector)
            )
            processed_results.append(expval)
        else:
            processed_results.append(results[i])
    return processed_results