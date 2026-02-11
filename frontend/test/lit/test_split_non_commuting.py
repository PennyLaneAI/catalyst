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
Lit tests for the split-non-commuting pass.

Tests the split-non-commuting pass which splits quantum functions that measure
non-commuting observables into multiple executions, one group per observable.
"""

import pennylane as qml

from catalyst import qjit
from catalyst.debug import get_compilation_stage
from catalyst.passes import apply_pass


def test_split_non_commuting_multiple_expvals():
    """
    Test splitting multiple non-commuting expvals into separate groups.
    Returns expval(Z(0)), expval(X(1)), expval(Y(2)) -> 3 groups.
    """

    @qjit(keep_intermediate=True)
    @apply_pass("split-non-commuting")
    @qml.set_shots(100)
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def circuit():
        qml.Rot(0.3, 0.5, 0.7, wires=0)
        qml.Rot(0.2, 0.4, 0.6, wires=1)
        qml.Rot(0.1, 0.8, 0.9, wires=2)

        return qml.expval(qml.Z(0)), qml.expval(qml.X(1)), qml.expval(qml.Y(2))

    # CHECK: transform.apply_registered_pass "split-non-commuting"
    print(circuit.mlir)

    # CHECK: func.func public @circuit
    # CHECK: call @circuit.group
    # CHECK: call @circuit.group
    # CHECK: call @circuit.group

    # CHECK-LABEL: func.func private @circuit.group
    # CHECK: quantum.namedobs {{.*}}[ PauliZ]
    # CHECK: quantum.expval {{.*}}

    # CHECK-LABEL: func.func private @circuit.group
    # CHECK: quantum.namedobs {{.*}}[ PauliX]
    # CHECK: quantum.expval {{.*}}

    # CHECK-LABEL: func.func private @circuit.group
    # CHECK: quantum.namedobs {{.*}}[ PauliY]
    # CHECK: quantum.expval {{.*}}
    print(get_compilation_stage(circuit, "QuantumCompilationStage"))


test_split_non_commuting_multiple_expvals()

# -----


def test_split_non_commuting_hamiltonian():
    """
    Test split-non-commuting with Hamiltonian observable.
    H = Z(0) + X(1) + 2 * Y(2)
    Pass runs split-to-single-terms first, then splits into groups.
    """

    @qjit(keep_intermediate=True)
    @apply_pass("split-non-commuting")
    @qml.set_shots(100)
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def circuit():
        qml.Rot(0.3, 0.5, 0.7, wires=0)
        qml.Rot(0.2, 0.4, 0.6, wires=1)
        qml.Rot(0.1, 0.8, 0.9, wires=2)

        return qml.expval(qml.Z(0) + qml.X(1) + 2 * qml.Y(2))

    # CHECK: transform.apply_registered_pass "split-non-commuting"
    print(circuit.mlir)

    # CHECK-LABEL: func.func public @circuit.quantum
    # CHECK: call @circuit.quantum.group.0
    # CHECK: call @circuit.quantum.group.1
    # CHECK: call @circuit.quantum.group.2

    # CHECK-LABEL: func.func public @circuit
    # CHECK: call @circuit.quantum

    # CHECK-LABEL: func.func private @circuit.quantum.group.0
    # CHECK: quantum.namedobs {{.*}}[ PauliZ]
    # CHECK: quantum.expval {{.*}}

    # CHECK-LABEL: func.func private @circuit.quantum.group.1
    # CHECK: quantum.namedobs {{.*}}[ PauliX]
    # CHECK: quantum.expval {{.*}}

    # CHECK-LABEL: func.func private @circuit.quantum.group.2
    # CHECK: quantum.namedobs {{.*}}[ PauliY]
    # CHECK: quantum.expval {{.*}}
    print(get_compilation_stage(circuit, "QuantumCompilationStage"))


test_split_non_commuting_hamiltonian()

# -----


def test_split_non_commuting_identity():
    """
    Test split-non-commuting with Identity in Hamiltonian.
    H = Z(0) + 2 * X(1) + 0.7 * Identity(2)
    Identity observables are folded into group 0 as constant 1.0.
    """

    @qjit(keep_intermediate=True)
    @apply_pass("split-non-commuting")
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def circuit():
        qml.Rot(0.3, 0.5, 0.7, wires=0)
        qml.Rot(0.2, 0.4, 0.6, wires=1)
        qml.Rot(0.1, 0.8, 0.9, wires=2)

        return qml.expval(qml.Z(0) + 2 * qml.X(1) + 0.7 * qml.Identity(2))

    # CHECK: transform.apply_registered_pass "split-non-commuting"
    print(circuit.mlir)

    # CHECK-LABEL: func.func public @circuit.quantum
    # CHECK: call @circuit.quantum.group.0
    # CHECK: call @circuit.quantum.group.1

    # CHECK-LABEL: func.func public @circuit
    # CHECK: call @circuit.quantum

    # CHECK-LABEL: func.func private @circuit.quantum.group.0
    # CHECK: %[[ONE:.*]] = arith.constant dense<1.000000e+00>
    # CHECK: quantum.namedobs {{.*}}[ PauliZ]
    # CHECK-NOT: quantum.namedobs {{.*}}[ Identity]
    # CHECK: quantum.expval {{.*}}
    # CHECK: return {{.*}}, %[[ONE]]

    # CHECK-LABEL: func.func private @circuit.quantum.group.1
    # CHECK: quantum.namedobs {{.*}}[ PauliX]
    # CHECK: quantum.expval {{.*}}

    # CHECK-NOT: func.func private @circuit.quantum.group.2

    print(get_compilation_stage(circuit, "QuantumCompilationStage"))


test_split_non_commuting_identity()
