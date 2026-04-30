# Copyright 2026 Xanadu Quantum Technologies Inc.

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
This module provides the infrastructure for lowering decomposition rules using compiler callbacks.
"""

import pennylane as qp


def paulirot_callback_wrapper(theta, pauli_word, wires):
    """Wraps paulirot decomp rule to enable compile-time lowering."""
    device = qp.device("null.qubit", wires=len(wires))
    qnode = qp.QNode(
        qp.ops.qubit.parametric_ops_multi_qubit._pauli_rot_decomposition._impl, device=device
    )
    circuit = qp.qjit(
        qnode,
        target="mlir",
        capture=True,
        static_argnums=2,
    )
    circuit(theta, wires, pauli_word)
    return circuit.mlir
