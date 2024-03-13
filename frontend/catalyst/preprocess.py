# def stopping_condition(op: qml.operation.Operator) -> bool:
#     """Specify whether or not an Operator object is supported by the device."""
#     if op.name == "QFT" and len(op.wires) >= 6:
#         return False
#     if op.name == "GroverOperator" and len(op.wires) >= 13:
#         return False
#     if op.name == "Snapshot":
#         return True
#     if op.__class__.__name__[:3] == "Pow" and qml.operation.is_trainable(op):
#         return False

#     return op.has_matrix

# def stopping_condition_shots(op: qml.operation.Operator) -> bool:
#     """Specify whether or not an Operator object is supported by the device with shots."""
#     return isinstance(op, (Conditional, MidMeasureMP)) or stopping_condition(op)

import pennylane as qml
from pennylane import transform
from catalyst.utils.exceptions import CompileError

@transform
def decompose_ops_to_unitary(tape, convert_to_matrix_ops):
    """
    """
    new_operations = []
    for op in tape.operations:
        if op in convert_to_matrix_ops or type(op) == qml.ops.Controlled:
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