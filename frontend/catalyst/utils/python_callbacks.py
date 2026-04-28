# TODO license

"""
This module provides the PoC for decomposition rules using compiler callbacks.

For now, decomposition rules using callbacks are expected to have signatures of the form
    (*op_data, wires) -> str
with the string returned expected to be a valid MLIR FuncOp in assembly or pretty-print format.

The input signature may be updated in the future to accomodate hyperparams or other more advanced
param types.
"""

import pennylane as qml

pauli_char_to_op = {
    "X": qml.RX,
    "Y": qml.RY,
    "Z": qml.RZ,
}


def test_rot_to_ppr(angle, pauli_string, wires):
    """Decomposition rule for pauli rot."""
    for pauli_char, wire in zip(pauli_string, wires):
        pauli_char_to_op[pauli_char](angle, wire)


def callback_wrapper(*op_args, op_wires=None, decomp_rule=None, capture=True, static_argnums=None):
    """Wraps paulirot decomp rule to enable compile-time lowering."""
    device = qml.device("null.qubit", wires=len(op_wires))
    qnode = qml.QNode(decomp_rule, device=device)
    circuit = qml.qjit(qnode, target="mlir", capture=capture, static_argnums=static_argnums)
    circuit(*op_args, op_wires)
    return circuit.mlir


if __name__ == "__main__":
    ir = callback_wrapper(
        0.2,
        "XYZ",
        op_wires=[1, 0, 2],
        decomp_rule=test_rot_to_ppr,
        static_argnums=1,
    )

    print(ir)
