# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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

import pennylane as qml
from pennylane import numpy as np

from catalyst import qjit


# CHECK-LABEL: public @jit_circuit
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit():
    # CHECK: %0 = quantum.alloc( 1) : !quantum.reg

    # CHECK: %1 = quantum.compbasis qreg %0 : !quantum.obs
    # CHECK: %2 = quantum.state %1 : tensor<2xcomplex<f64>>
    qml.Snapshot()  # |0>

    # CHECK: %3 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    # CHECK: %out_qubits = quantum.custom "PauliX"() %3 : !quantum.bit
    # CHECK: %4 = quantum.insert %0[ 0], %out_qubits : !quantum.reg, !quantum.bit
    qml.X(wires=0)

    # CHECK: %5 = quantum.compbasis qreg %4 : !quantum.obs
    # CHECK: %6 = quantum.state %5 : tensor<2xcomplex<f64>>
    qml.Snapshot()  # |1>

    # CHECK: %7 = quantum.extract %4[ 0] : !quantum.reg -> !quantum.bit
    # CHECK: %out_qubits_1 = quantum.custom "Hadamard"() %7 : !quantum.bit
    # CHECK: %8 = quantum.insert %4[ 0], %out_qubits_1 : !quantum.reg, !quantum.bit
    qml.Hadamard(wires=0)

    # CHCK: %9 = quantum.compbasis qreg %8 : !quantum.obs
    # CHECK: %10 = quantum.state %9 : tensor<2xcomplex<f64>>
    qml.Snapshot()  # |->

    # CHECK: %11 = quantum.extract %8[ 0] : !quantum.reg -> !quantum.bit
    # CHECK: %out_qubits_2 = quantum.custom "PhaseShift"(%cst_0) %11 : !quantum.bit
    # CHECK: %12 = quantum.insert %8[ 0], %out_qubits_2 : !quantum.reg, !quantum.bit
    qml.PhaseShift(np.pi / 2, wires=0)

    # CHECK: %13 = quantum.compbasis qreg %12 : !quantum.obs
    # CHECK: %14 = quantum.state %13 : tensor<2xcomplex<f64>>
    qml.Snapshot()  # |-i>

    # CHECK: %15 = quantum.extract %12[ 0] : !quantum.reg -> !quantum.bit
    # CHECK: %out_qubits_3 = quantum.custom "PauliZ"() %15 : !quantum.bit
    # CHECK: %16 = quantum.insert %12[ 0], %out_qubits_3 : !quantum.reg, !quantum.bit
    qml.Z(wires=0)

    # CHECK: %17 = quantum.compbasis qreg %16 : !quantum.obs
    # CHECK: %18 = quantum.state %17 : tensor<2xcomplex<f64>>
    qml.Snapshot()  # |+i>

    # CHECK: %19 = quantum.extract %16[ 0] : !quantum.reg -> !quantum.bit
    # CHECK: %out_qubits_4 = quantum.custom "PhaseShift"(%cst) %19 : !quantum.bit
    # CHECK: %20 = quantum.insert %16[ 0], %out_qubits_4 : !quantum.reg, !quantum.bit
    qml.PhaseShift(-np.pi / 2, wires=0)

    # CHECK: %21 = quantum.compbasis qreg %20 : !quantum.obs
    # CHECK: %22 = quantum.state %21 : tensor<2xcomplex<f64>>
    qml.Snapshot()  # |+>

    # CHECK: %23 = quantum.compbasis qreg %20 : !quantum.obs
    # CHECK: %24 = quantum.state %23 : tensor<2xcomplex<f64>>
    # CHECK: %25 = quantum.compbasis qreg %20 : !quantum.obs
    # CHECK: %26 = quantum.probs %25 : tensor<2xf64>
    # CHECK: %27 = quantum.extract %20[ 0] : !quantum.reg -> !quantum.bit
    # CHECK: %28 = quantum.namedobs %27[ PauliX] : !quantum.obs
    # CHECK: %29 = quantum.expval %28 : f64
    # CHECK: %from_elements = tensor.from_elements %29 : tensor<f64>
    # CHECK: %30 = quantum.extract %20[ 0] : !quantum.reg -> !quantum.bit
    # CHECK: %31 = quantum.namedobs %30[ PauliZ] : !quantum.obs
    # CHECK: %32 = quantum.var %31 : f64
    # CHECK: %from_elements_5 = tensor.from_elements %32 : tensor<f64>
    # CHECK: quantum.dealloc %20 : !quantum.reg
    # CHECK: quantum.device_release
    # CHECK: return %2, %6, %10, %14, %18, %22, %24, %26, %from_elements, %from_elements_5 : tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>, tensor<2xf64>, tensor<f64>, tensor<f64>
    return qml.state(), qml.probs(), qml.expval(qml.X(0)), qml.var(qml.Z(0))


print(circuit.mlir)


# CHECK-LABEL: public @jit_circuit
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2, shots=5))
def circuit():
    # CHECK: %0 = quantum.alloc( 2) : !quantum.reg

    # CHECK: %1 = quantum.compbasis qreg %0 : !quantum.obs
    # CHECK: %2 = quantum.state %1 : tensor<4xcomplex<f64>>
    qml.Snapshot()  # |00>

    # CHECK: %3 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    # CHECK: %out_qubits = quantum.custom "Hadamard"() %3 : !quantum.bit
    qml.Hadamard(wires=0)

    # CHECK: %4 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    # CHECK: %out_qubits_0 = quantum.custom "Hadamard"() %4 : !quantum.bit
    qml.Hadamard(wires=1)
    # CHECK: %5 = quantum.insert %0[ 0], %out_qubits_0 : !quantum.reg, !quantum.bit
    # CHECK: %6 = quantum.insert %5[ 1], %out_qubits : !quantum.reg, !quantum.bit

    # CHECK: %7 = quantum.compbasis qreg %6 : !quantum.obs
    # CHECK: %8 = quantum.state %7 : tensor<4xcomplex<f64>>
    qml.Snapshot()  # |++>

    # CHECK: %9 = quantum.extract %6[ 0] : !quantum.reg -> !quantum.bit
    # CHECK: %out_qubits_1 = quantum.custom "Hadamard"() %9 : !quantum.bit
    # CHECK: %out_qubits_2 = quantum.custom "PauliX"() %out_qubits_1 : !quantum.bit
    # CHECK: %10 = quantum.extract %6[ 1] : !quantum.reg -> !quantum.bit
    # CHECK: %out_qubits_3 = quantum.custom "Hadamard"() %10 : !quantum.bit
    # CHECK: %out_qubits_4 = quantum.custom "PauliX"() %out_qubits_3 : !quantum.bit
    # CHECK: %11 = quantum.insert %6[ 0], %out_qubits_2 : !quantum.reg, !quantum.bit
    # CHECK: %12 = quantum.insert %11[ 1], %out_qubits_4 : !quantum.reg, !quantum.bit
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)  # |00>
    qml.X(wires=0)
    qml.X(wires=1)  # |11> to measure in comp-basis

    # CHECK: %13 = quantum.compbasis qreg %12 : !quantum.obs
    # CHECK: %eigvals, %counts = quantum.counts %13 : tensor<4xf64>, tensor<4xi64>
    # CHECK: %14 = stablehlo.convert %eigvals : (tensor<4xf64>) -> tensor<4xi64>
    # CHECK: %15 = quantum.compbasis qreg %12 : !quantum.obs
    # CHECK: %16 = quantum.sample %15 : tensor<5x2xf64>
    # CHECK: %17 = stablehlo.convert %16 : (tensor<5x2xf64>) -> tensor<5x2xi64>
    # CHECK: quantum.dealloc %12 : !quantum.reg
    # CHECK: return %2, %8, %14, %counts, %17 : tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>, tensor<4xi64>, tensor<4xi64>, tensor<5x2xi64>
    return qml.counts(), qml.sample()


print(circuit.mlir)
