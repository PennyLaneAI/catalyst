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
Test decomposition rules registered on the special operations are correctly generated.

There are 5 special operators that do not lower to CustomOp or OperatorOp in MLIR, and instead lower
to their own operations:
- MultiRZ
- PauliRot
- GlobalPhase (TODO: migration to Operator2 not done yet)
- PCPhase (TODO: migration to Operator2 not done yet)
- QubitUnitary (TODO: migration to Operator2 not done yet)
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable = line-too-long,unused-argument

import pennylane as qp


def test_multirz():
    """
    Test that decomposing qp.MultiRZ works.
    """

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=3))
    def c():
        qp.MultiRZ(theta=0.1, wires=[0, 1])
        qp.MultiRZ(theta=0.2, wires=[2])
        return qp.state()

    print(c.mlir)


# CHECK: func.func public @c()
# CHECK: qref.multirz({{%.+}}) {{%.+}}, {{%.+}} : !qref.bit, !qref.bit
# CHECK: qref.multirz({{%.+}}) {{%.+}} : !qref.bit
# CHECK: func.func private @"__builtin__multi_rz_decomposition_MultiRZ{theta:[f64]}{wires:2}{}"
# CHECK-SAME:   target_gate = "MultiRZ{theta:[f64]}{wires:2}{}"
# CHECK: func.func private @"__builtin__multi_rz_decomposition_MultiRZ{theta:[f64]}{wires:1}{}"
# CHECK-SAME:   target_gate = "MultiRZ{theta:[f64]}{wires:1}{}"
test_multirz()


def test_paulirot():
    """
    Test that decomposing qp.PauliRot works.
    """

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=3))
    def c():
        qp.PauliRot(theta=0.1, pauli_word="XX", wires=[0, 1])
        qp.PauliRot(theta=0.2, pauli_word="Z", wires=[2])
        # TODO: decomposition rule of Y PauliRot involves RX, which has not been migrated to
        # Operator2 yet
        # qp.PauliRot(theta=0.2, pauli_word="YZX", wires=[0, 1, 2])
        return qp.state()

    print(c.mlir)


# CHECK: func.func public @c()
# CHECK: qref.paulirot ["X", "X"]({{%.+}}) {{%.+}}, {{%.+}} : !qref.bit, !qref.bit
# CHECK: qref.paulirot ["Z"]({{%.+}}) {{%.+}} : !qref.bit
# CHECK: func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:2}{pauli_word:XX}"
# CHECK-SAME:   target_gate = "PauliRot{theta:[f64]}{wires:2}{pauli_word:XX}"
# CHECK: func.func private @"__builtin__pauli_rot_decomposition_PauliRot{theta:[f64]}{wires:1}{pauli_word:Z}"
# CHECK-SAME:   target_gate = "PauliRot{theta:[f64]}{wires:1}{pauli_word:Z}"
test_paulirot()
