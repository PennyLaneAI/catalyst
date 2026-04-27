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
This file performs the lit test checking MLIR output for qp.Snapshot support in Catalyst.
"""

import pennylane as qp
from pennylane import numpy as np

from catalyst import qjit

# pylint: disable=line-too-long


# CHECK-LABEL: public @jit_single_qubit_circuit
@qjit(target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=1))
def single_qubit_circuit():
    """Test MLIR output of all six single qubit basis states in qp.Snapshot without shots"""
    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot0:%.+]] = quantum.state [[compbasis]] : tensor<2xcomplex<f64>>
    qp.Snapshot()  # |0>

    qp.X(wires=0)

    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot1:%.+]] = quantum.state [[compbasis]] : tensor<2xcomplex<f64>>
    qp.Snapshot()  # |1>

    qp.Hadamard(wires=0)

    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot2:%.+]] = quantum.state [[compbasis]] : tensor<2xcomplex<f64>>
    qp.Snapshot()  # |->

    qp.PhaseShift(np.pi / 2, wires=0)  # pylint: disable=no-member

    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot3:%.+]] = quantum.state [[compbasis]] : tensor<2xcomplex<f64>>
    qp.Snapshot()  # |-i>

    qp.Z(wires=0)

    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot4:%.+]] = quantum.state [[compbasis]] : tensor<2xcomplex<f64>>
    qp.Snapshot()  # |+i>

    qp.PhaseShift(-np.pi / 2, wires=0)  # pylint: disable=no-member

    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot5:%.+]] = quantum.state [[compbasis]] : tensor<2xcomplex<f64>>
    qp.Snapshot()  # |+>

    # CHECK: return [[snapshot0]], [[snapshot1]], [[snapshot2]], [[snapshot3]], [[snapshot4]], [[snapshot5]], {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}} :
    # CHECK-SAME: tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>,
    # CHECK-SAME: tensor<2xcomplex<f64>>, tensor<2xf64>, tensor<f64>, tensor<f64>
    return qp.state(), qp.probs(), qp.expval(qp.X(0)), qp.var(qp.Z(0))


print(single_qubit_circuit.mlir)


# CHECK-LABEL: public @jit_two_qubit_circuit
@qjit(target="mlir")
@qp.set_shots(5)
@qp.qnode(qp.device("lightning.qubit", wires=2), mcm_method="single-branch-statistics")
def two_qubit_circuit():
    """Test MLIR output of qp.Snapshot on two qubits with shots"""

    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot0:%.+]] = quantum.state [[compbasis]] : tensor<4xcomplex<f64>>
    qp.Snapshot()  # |00>

    qp.Hadamard(wires=0)
    qp.Hadamard(wires=1)

    # CHECK: [[compbasis:%.+]] = quantum.compbasis qreg {{%.+}} : !quantum.obs
    # CHECK: [[snapshot1:%.+]] = quantum.state [[compbasis]] : tensor<4xcomplex<f64>>
    qp.Snapshot()  # |++>

    qp.Hadamard(wires=0)
    qp.Hadamard(wires=1)  # |00>
    qp.X(wires=0)
    qp.X(wires=1)  # |11> to measure in comp-basis

    # CHECK: return [[snapshot0]], [[snapshot1]], {{%.+}}, {{%.+}}, {{%.+}} : tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>, tensor<4xi64>, tensor<4xi64>, tensor<5x2xi64>
    return qp.counts(), qp.sample()


print(two_qubit_circuit.mlir)
