# Copyright 2026 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RUN: %PYTHON %s | FileCheck %s

"""
Lit tests for the split-to-single-terms pass.

Tests the split-to-single-terms pass for different types of Hamiltonian expectation values.
"""

import pennylane as qml

from catalyst import qjit
from catalyst.debug import get_compilation_stage
from catalyst.passes import apply_pass


def test_split_to_single_terms_basic():
    """
    Test basic Hamiltonian splitting
    H = Z(0) + X(1) + 2 * Y(2)
    """

    @qjit(keep_intermediate=True)
    @apply_pass("split-to-single-terms")
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def circuit():
        qml.Rot(0.3, 0.5, 0.7, wires=0)
        qml.Rot(0.2, 0.4, 0.6, wires=1)
        qml.Rot(0.1, 0.8, 0.9, wires=2)

        return qml.expval(qml.Z(0) + qml.X(1) + 2 * qml.Y(2))

    # CHECK: transform.apply_registered_pass "split-to-single-terms"
    print(circuit.mlir)

    # CHECK: func.func public @circuit.quantum
    # CHECK: %[[OBS_Z:.*]] = quantum.namedobs %{{.*}}[ PauliZ]
    # CHECK: %[[OBS_X:.*]] = quantum.namedobs %{{.*}}[ PauliX]
    # CHECK: %[[OBS_Y:.*]] = quantum.namedobs %{{.*}}[ PauliY]
    # CHECK: %[[EXPVAL_Z:.*]] = quantum.expval %[[OBS_Z]]
    # CHECK: %[[TENSOR_Z:.*]] = tensor.from_elements %[[EXPVAL_Z]]
    # CHECK: %[[EXPVAL_X:.*]] = quantum.expval %[[OBS_X]]
    # CHECK: %[[TENSOR_X:.*]] = tensor.from_elements %[[EXPVAL_X]]
    # CHECK: %[[EXPVAL_Y:.*]] = quantum.expval %[[OBS_Y]]

    # CHECK: func.func public @circuit
    # CHECK-NOT: quantum.custom.*
    # CHECK-NOT: quantum.namedobs.*
    # CHECK-NOT: quantum.expval.*
    # CHECK: %[[CALL:.*]]:3 = call @circuit.quantum
    # CHECK: %[[CONCAT:.*]] = stablehlo.concatenate
    # CHECK: %[[RESULT:.*]] = stablehlo.reduce(%[[CONCAT]] init: {{%.+}}) applies stablehlo.add
    # CHECK: return %[[RESULT]]
    print(get_compilation_stage(circuit, "QuantumCompilationStage"))


test_split_to_single_terms_basic()

# -----


def test_split_to_single_terms_tensor_product():
    """
    Test tensor product Hamiltonian splitting
    H = Z(0) @ X(1) + 2 * Y(2)
    """

    @qjit(keep_intermediate=True)
    @apply_pass("split-to-single-terms")
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def circuit():
        qml.Rot(0.3, 0.5, 0.7, wires=0)
        qml.Rot(0.2, 0.4, 0.6, wires=1)
        qml.Rot(0.1, 0.8, 0.9, wires=2)

        return qml.expval(qml.Z(0) @ qml.X(1) + 2 * qml.Y(2))

    # CHECK: transform.apply_registered_pass "split-to-single-terms"
    print(circuit.mlir)

    # CHECK: func.func public @circuit.quantum
    # CHECK: %[[OUT_QUBIT_0:.*]] = quantum.custom "Rot"{{.*}}
    # CHECK: %[[OUT_QUBIT_1:.*]] = quantum.custom "Rot"{{.*}}
    # CHECK: %[[OUT_QUBIT_2:.*]] = quantum.custom "Rot"{{.*}}
    # CHECK: %[[OBS_Z:.*]] = quantum.namedobs %[[OUT_QUBIT_0]][ PauliZ]
    # CHECK: %[[OBS_X:.*]] = quantum.namedobs %[[OUT_QUBIT_1]][ PauliX]
    # CHECK: %[[OBS_ZX:.*]] = quantum.tensor %[[OBS_Z]], %[[OBS_X]]
    # CHECK: %[[OBS_Y:.*]] = quantum.namedobs %[[OUT_QUBIT_2]][ PauliY]
    # CHECK: %[[EXPVAL_ZX:.*]] = quantum.expval %[[OBS_ZX]]
    # CHECK: %[[TENSOR_ZX:.*]] = tensor.from_elements %[[EXPVAL_ZX]]
    # CHECK: %[[EXPVAL_Y:.*]] = quantum.expval %[[OBS_Y]]
    # CHECK: %[[TENSOR_Y:.*]] = tensor.from_elements %[[EXPVAL_Y]]

    # CHECK: func.func public @circuit
    # CHECK-NOT: quantum.custom.*
    # CHECK-NOT: quantum.namedobs.*
    # CHECK-NOT: quantum.expval.*
    # CHECK: %[[CALL:.*]]:2 = call @circuit.quantum
    # CHECK: stablehlo.multiply
    # CHECK: stablehlo.broadcast_in_dim
    # CHECK: stablehlo.broadcast_in_dim
    # CHECK: %[[CONCAT:.*]] = stablehlo.concatenate
    # CHECK: %[[RESULT:.*]] = stablehlo.reduce(%[[CONCAT]] init: {{%.+}}) applies stablehlo.add
    # CHECK: return %[[RESULT]]
    print(get_compilation_stage(circuit, "QuantumCompilationStage"))


test_split_to_single_terms_tensor_product()

# -----


def test_split_to_single_terms_identity():
    """
    Test identity Hamiltonian splitting
    H = Z(0) + 2 * X(1) + 0.7 * Identity(2)
    """

    @qjit(keep_intermediate=True)
    @apply_pass("split-to-single-terms")
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def circuit():
        qml.Rot(0.3, 0.5, 0.7, wires=0)
        qml.Rot(0.2, 0.4, 0.6, wires=1)
        qml.Rot(0.1, 0.8, 0.9, wires=2)

        return qml.expval(qml.Z(0) + 2 * qml.X(1) + 0.7 * qml.Identity(2))

    # CHECK: transform.apply_registered_pass "split-to-single-terms"
    print(circuit.mlir)

    # CHECK: func.func public @circuit.quantum{{.*}}
    # CHECK: %[[ONE:.*]] = {{.*}}constant {{.*}}1.000000e+00
    # CHECK: quantum.device
    # CHECK: %[[OBS_Z:.*]] = quantum.namedobs %{{.*}}[ PauliZ]
    # CHECK: %[[OBS_X:.*]] = quantum.namedobs %{{.*}}[ PauliX]
    # CHECK: %[[EXPVAL_Z:.*]] = quantum.expval %[[OBS_Z]]
    # CHECK: %[[TENSOR_Z:.*]] = tensor.from_elements %[[EXPVAL_Z]]
    # CHECK: %[[EXPVAL_X:.*]] = quantum.expval %[[OBS_X]]
    # CHECK: %[[TENSOR_X:.*]] = tensor.from_elements %[[EXPVAL_X]]
    # CHECK-NOT: quantum.namedobs.*Identity
    # CHECK-NOT: quantum.expval.*Identity
    # CHECK: return {{.*}}, {{.*}}, %[[ONE]]

    # CHECK: func.func public @circuit{{.*}}
    # CHECK-NOT: quantum.custom.*
    # CHECK-NOT: quantum.namedobs.*
    # CHECK-NOT: quantum.expval.*
    # CHECK: %[[CALL:.*]]:3 = call @circuit.quantum{{.*}}
    # CHECK: stablehlo.multiply
    # CHECK: stablehlo.multiply
    # CHECK: stablehlo.multiply
    # CHECK: stablehlo.broadcast_in_dim
    # CHECK: stablehlo.broadcast_in_dim
    # CHECK: stablehlo.broadcast_in_dim
    # CHECK: %[[CONCAT:.*]] = stablehlo.concatenate
    # CHECK: %[[RESULT:.*]] = stablehlo.reduce(%[[CONCAT]] init: {{%.+}}) applies stablehlo.add
    # CHECK: return %[[RESULT]]
    print(get_compilation_stage(circuit, "QuantumCompilationStage"))


test_split_to_single_terms_identity()
