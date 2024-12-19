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

# RUN: %PYTHON %s | FileCheck %s

"""
Test quantum circuits with static (knonw at compile time) specifications.
"""

import pennylane as qml

from catalyst import qjit


def test_static_params():
    """Test operations with static params."""

    @qjit(target="mlir")
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def circuit():
        x = 3.14
        y = 0.6
        qml.Rot(x, y, x + y, wires=0)

        qml.RX(x, wires=0)
        qml.RY(y, wires=1)
        qml.RZ(x, wires=2)

        qml.IsingXX(x, wires=[0, 1])
        qml.IsingXX(y, wires=[1, 2])
        qml.IsingZZ(x, wires=[0, 1])

        qml.CRX(x, wires=[0, 1])
        qml.CRY(x, wires=[0, 1])
        qml.CRZ(x, wires=[0, 1])

        return qml.state()

    print(circuit.mlir)


# CHECK-LABEL: public @jit_circuit
# CHECK: %[[REG:.*]] = quantum.alloc( 4) : !quantum.reg
# CHECK: %[[BIT1:.*]] = quantum.extract %[[REG]][ 0] : !quantum.reg -> !quantum.bit
# CHECK: %[[ROT:.*]] = quantum.static_custom "Rot"
# CHECK: %[[RX:.*]] = quantum.static_custom "RX"
# CHECK: %[[BIT1:.*]] = quantum.extract %[[REG]][ 1]
# CHECK: %[[RY1:.*]] = quantum.static_custom "RY"
# CHECK: %[[XX1:.*]] = quantum.static_custom "IsingXX"
# CHECK: %[[BIT2:.*]] = quantum.extract %[[REG]][ 2]
# CHECK: %[[RZ:.*]] = quantum.static_custom "RZ"
# CHECK: %[[XX2:.*]] = quantum.static_custom "IsingXX"
# CHECK: %[[ZZ:.*]] = quantum.static_custom "IsingZZ"
# CHECK: %[[CRX:.*]] = quantum.static_custom "CRX"
# CHECK: %[[CRY:.*]] = quantum.static_custom "CRY"
# CHECK: %[[CRZ:.*]] = quantum.static_custom "CRZ"
test_static_params()
