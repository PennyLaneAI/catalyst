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

"""
Unit tests for the dynamic work wire allocation.
Note that this feature is only available under the plxpr pipeline.
"""

# RUN: %PYTHON %s | FileCheck %s

import pennylane as qml
from pennylane.allocation import allocate, deallocate
from catalyst import qjit


qml.capture.enable()


@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def test_basic_dynalloc():
    qml.X(1)
    qml.X(1)
    wires = allocate(1)
    qml.X(wires[0])
    qml.Z(wires[0])
    deallocate(wires[0])

    qml.X(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)

    wires = allocate(2)
    qml.Y(wires[0])
    qml.Z(wires[1])
    deallocate(wires[:])

    return qml.probs()


#print(test_basic_dynalloc.jaxpr)
print(test_basic_dynalloc.mlir)

# CHECK: func.func public @test_basic_dynalloc() -> tensor<8xf64>
# CHECK:   %c0_i64 = arith.constant 0 : i64
# CHECK:   quantum.device shots(%c0_i64)
# CHECK:   %0 = quantum.alloc( 3) : !quantum.reg
# CHECK:   %1 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
# CHECK:   %out_qubits = quantum.custom "PauliX"() %1 : !quantum.bit
# CHECK:   %out_qubits_0 = quantum.custom "PauliX"() %out_qubits : !quantum.bit
# CHECK:   %2 = quantum.alloc( 1) : !quantum.reg
# CHECK:   %3 = quantum.extract %2[ 0] : !quantum.reg -> !quantum.bit
# CHECK:   %out_qubits_1 = quantum.custom "PauliX"() %3 : !quantum.bit
# CHECK:   %out_qubits_2 = quantum.custom "PauliZ"() %out_qubits_1 : !quantum.bit
# CHECK:   %4 = quantum.insert %2[ 0], %out_qubits_2 : !quantum.reg, !quantum.bit
# CHECK:   quantum.dealloc %4 : !quantum.reg
# CHECK:   %5 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
# CHECK:   %out_qubits_3 = quantum.custom "PauliX"() %5 : !quantum.bit
# CHECK:   %out_qubits_4 = quantum.custom "Hadamard"() %out_qubits_0 : !quantum.bit
# CHECK:   %6 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
# CHECK:   %out_qubits_5 = quantum.custom "Hadamard"() %6 : !quantum.bit
# CHECK:   %7 = quantum.alloc( 2) : !quantum.reg
# CHECK:   %8 = quantum.extract %7[ 0] : !quantum.reg -> !quantum.bit
# CHECK:   %9 = quantum.extract %7[ 1] : !quantum.reg -> !quantum.bit
# CHECK:   %out_qubits_6 = quantum.custom "PauliY"() %8 : !quantum.bit
# CHECK:   %out_qubits_7 = quantum.custom "PauliZ"() %9 : !quantum.bit
# CHECK:   %10 = quantum.insert %7[ 0], %out_qubits_6 : !quantum.reg, !quantum.bit
# CHECK:   %11 = quantum.insert %10[ 1], %out_qubits_7 : !quantum.reg, !quantum.bit
# CHECK:   quantum.dealloc %11 : !quantum.reg
# CHECK:   %12 = quantum.insert %0[ 1], %out_qubits_4 : !quantum.reg, !quantum.bit
# CHECK:   %13 = quantum.insert %12[ 0], %out_qubits_3 : !quantum.reg, !quantum.bit
# CHECK:   %14 = quantum.insert %13[ 2], %out_qubits_5 : !quantum.reg, !quantum.bit
# CHECK:   %15 = quantum.compbasis qreg %14 : !quantum.obs
# CHECK:   %16 = quantum.probs %15 : tensor<8xf64>
# CHECK:   quantum.dealloc %14 : !quantum.reg
# CHECK:   quantum.device_release
# CHECK:   return %16 : tensor<8xf64>

qml.capture.disable()
