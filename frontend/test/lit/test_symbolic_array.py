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
Tests for using symobolic_array in a QNode.
"""

# RUN: %PYTHON %s | FileCheck %s

import pennylane as qp


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device('null.qubit', wires=2))
def c():
    # CHECK-LABEL: func.func public @c

    # CHECK: [[mat:%.+]] = catalyst.symbolic_array : tensor<4x4xcomplex<f64>>
    mat = qp.capture.symbolic_array((4,4), complex)
    # CHECK: qref.unitary([[mat]] : tensor<4x4xcomplex<f64>>) {{%.+}}, {{%.+}} : !qref.bit, !qref.bit
    qp.QubitUnitary(mat, wires=(0,1))

    # CHECK: {{%.+}} = catalyst.symbolic_array : tensor<2x1x3xi64>
    _ = qp.capture.symbolic_array((2,1,3), int)

    # CHECK: [[phi:%.+]] = catalyst.symbolic_array : tensor<f64>
    a = qp.capture.symbolic_array((), float)

    # can use in arithmetic
    # CHECK: [[theta:%.+]] = stablehlo.multiply {{%.+}}, [[phi]] : tensor<f64>
    b = 2 * a

    # CHECK: [[wire:%.+]] = catalyst.symbolic_array : tensor<i64>
    # CHECK: [[extracted_wire:%.+]] = tensor.extract [[wire]][] : tensor<i64>
    # CHECK: [[qbit:%.+]] = qref.get {{%.+}}[[[extracted_wire]]] : !qref.reg<2>, i64 -> !qref.bit
    wire = qp.capture.symbolic_array((), int)

    # CHECK: [[extracted_theta:%.+]] = tensor.extract [[theta]][] : tensor<f64>
    # CHECK: qref.custom "RX"([[extracted_theta]]) [[qbit]] : !qref.bit
    qp.RX(b, wire)
    
    return qp.state()

print(c.mlir)