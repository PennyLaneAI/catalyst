# Copyright 2025 Xanadu Quantum Technologies Inc.

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
This file performs the lit test checking MLIR output for qml.Snapshot support in Catalyst.
"""

import pennylane as qml
from pennylane import numpy as np

from catalyst import qjit

# pylint: disable=line-too-long


# CHECK-LABEL: public @jit_single_qubit_circuit
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=1))
def single_qubit_circuit():
    """Test MLIR output of all six single qubit basis states in qml.Snapshot without shots"""
    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot0:%.+]] = quantum.state [[compbasis]] : tensor<2xcomplex<f64>>
    qml.Snapshot()  # |0>

    qml.X(wires=0)

    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot1:%.+]] = quantum.state [[compbasis]] : tensor<2xcomplex<f64>>
    qml.Snapshot()  # |1>

    qml.Hadamard(wires=0)

    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot2:%.+]] = quantum.state [[compbasis]] : tensor<2xcomplex<f64>>
    qml.Snapshot()  # |->

    qml.PhaseShift(np.pi / 2, wires=0)

    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot3:%.+]] = quantum.state [[compbasis]] : tensor<2xcomplex<f64>>
    qml.Snapshot()  # |-i>

    qml.Z(wires=0)

    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot4:%.+]] = quantum.state [[compbasis]] : tensor<2xcomplex<f64>>
    qml.Snapshot()  # |+i>

    qml.PhaseShift(-np.pi / 2, wires=0)

    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot5:%.+]] = quantum.state [[compbasis]] : tensor<2xcomplex<f64>>
    qml.Snapshot()  # |+>

    # CHECK: return [[snapshot0]], [[snapshot1]], [[snapshot2]], [[snapshot3]], [[snapshot4]], [[snapshot5]], {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}} :
    # CHECK-SAME: tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>,
    # CHECK-SAME: tensor<2xcomplex<f64>>, tensor<2xf64>, tensor<f64>, tensor<f64>
    return qml.state(), qml.probs(), qml.expval(qml.X(0)), qml.var(qml.Z(0))


print(single_qubit_circuit.mlir)


# CHECK-LABEL: public @jit_two_qubit_circuit
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2, shots=5))
def two_qubit_circuit():
    """Test MLIR output of qml.Snapshot on two qubits with shots"""

    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot0:%.+]] = quantum.state [[compbasis]] : tensor<4xcomplex<f64>>
    qml.Snapshot()  # |00>

    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)

    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot1:%.+]] = quantum.state [[compbasis]] : tensor<4xcomplex<f64>>
    qml.Snapshot()  # |++>

    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)  # |00>
    qml.X(wires=0)
    qml.X(wires=1)  # |11> to measure in comp-basis

    # CHECK: return [[snapshot0]], [[snapshot1]], {{%.+}}, {{%.+}}, {{%.+}} : tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>, tensor<4xi64>, tensor<4xi64>, tensor<5x2xi64>
    return qml.counts(), qml.sample()


print(two_qubit_circuit.mlir)
