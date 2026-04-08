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

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable=line-too-long

import pennylane as qp


# CHECK: func.func public @test_state0() -> tensor<8xcomplex<f64>>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_state0():
    """
    Test an empty circuit with state terminal measurement.
    """
    # CHECK:   quantum.device
    # CHECK:   [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK:   [[compbasis:%.+]] = qref.compbasis(qreg [[alloc]] : !qref.reg<3>) : !quantum.obs
    # CHECK:   [[state:%.+]] = quantum.state [[compbasis]] : tensor<8xcomplex<f64>>
    # CHECK:   qref.dealloc [[alloc]] : !qref.reg<3>
    # CHECK:   quantum.device_release
    # CHECK:   return [[state]] : tensor<8xcomplex<f64>>
    return qp.state()


print(test_state0.mlir)


# CHECK: func.func public @test_probs0() -> tensor<8xf64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_probs0():
    """
    Test an empty circuit with probs terminal measurement on all wires.
    """
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[compbasis:%.+]] = qref.compbasis(qreg [[alloc]] : !qref.reg<3>) : !quantum.obs
    # CHECK: [[probs:%.+]] = quantum.probs [[compbasis]] : tensor<8xf64>
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<3>
    # CHECK: quantum.device_release
    # CHECK: return [[probs]] : tensor<8xf64>
    return qp.probs()


print(test_probs0.mlir)


# CHECK: func.func public @test_probs1() -> tensor<4xf64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_probs1():
    """
    Test an empty circuit with probs terminal measurement on static wires.
    """
    # CHECK: [[one:%.+]] = arith.constant 1 : i64
    # CHECK: [[zero:%.+]] = arith.constant 0 : i64
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][[[one]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[compbasis:%.+]] = qref.compbasis qubits [[q0]], [[q1]] : !quantum.obs
    # CHECK: [[probs:%.+]] = quantum.probs [[compbasis]] : tensor<4xf64>
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<3>
    # CHECK: quantum.device_release
    # CHECK: return [[probs]] : tensor<4xf64>
    return qp.probs(wires=[0, 1])


print(test_probs1.mlir)


# CHECK: func.func public @test_probs2(%arg0: tensor<i64>) -> tensor<4xf64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def test_probs2(i: int):
    """
    Test an empty circuit with probs terminal measurement on dynamic wires.
    """
    # CHECK: [[zero:%.+]] = arith.constant 0 : i64
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[i:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK: [[qi:%.+]] = qref.get [[alloc]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[compbasis:%.+]] = qref.compbasis qubits [[q0]], [[qi]] : !quantum.obs
    # CHECK: [[probs:%.+]] = quantum.probs [[compbasis]] : tensor<4xf64>
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<3>
    # CHECK: quantum.device_release
    # CHECK: return [[probs]] : tensor<4xf64>
    return qp.probs(wires=[0, i])


print(test_probs2.mlir)


# CHECK: func.func public @test_sample0() -> tensor<1x3xi64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3), shots=1)
def test_sample0():
    """
    Test an empty circuit with sample terminal measurement on all wires.
    """
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[compbasis:%.+]] = qref.compbasis(qreg [[alloc]] : !qref.reg<3>) : !quantum.obs
    # CHECK: [[sample:%.+]] = quantum.sample [[compbasis]] : tensor<1x3xf64>
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<3>
    # CHECK: quantum.device_release
    return qp.sample()


print(test_sample0.mlir)


# CHECK: func.func public @test_sample1() -> tensor<1x2xi64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3), shots=1)
def test_sample1():
    """
    Test an empty circuit with sample terminal measurement on static wires.
    """
    # CHECK: [[zero:%.+]] = arith.constant 0 : i64
    # CHECK: [[one:%.+]] = arith.constant 1 : i64
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][[[one]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[compbasis:%.+]] = qref.compbasis qubits [[q0]], [[q1]] : !quantum.obs
    # CHECK: [[sample:%.+]] = quantum.sample [[compbasis]] : tensor<1x2xf64>
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<3>
    # CHECK: quantum.device_release
    return qp.sample(wires=[0, 1])


print(test_sample1.mlir)


# CHECK: func.func public @test_sample2(%arg0: tensor<i64>) -> tensor<1x2xi64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3), shots=1)
def test_sample2(i: int):
    """
    Test an empty circuit with sample terminal measurement on dynamic wires.
    """
    # CHECK: [[zero:%.+]] = arith.constant 0 : i64
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[i:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK: [[qi:%.+]] = qref.get [[alloc]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[compbasis:%.+]] = qref.compbasis qubits [[q0]], [[qi]] : !quantum.obs
    # CHECK: [[sample:%.+]] = quantum.sample [[compbasis]] : tensor<1x2xf64>
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<3>
    # CHECK: quantum.device_release
    return qp.sample(wires=[0, i])


print(test_sample2.mlir)


# CHECK: func.func public @test_counts0() -> (tensor<8xi64>, tensor<8xi64>)
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3), shots=1)
def test_counts0():
    """
    Test an empty circuit with counts terminal measurement on all wires.
    """
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[compbasis:%.+]] = qref.compbasis(qreg [[alloc]] : !qref.reg<3>) : !quantum.obs
    # CHECK: [[eigens:%.+]], [[counts:%.+]] = quantum.counts [[compbasis]] : tensor<8xf64>, tensor<8xi64>
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<3>
    # CHECK: quantum.device_release
    return qp.counts()


print(test_counts0.mlir)


# CHECK: func.func public @test_counts1() -> (tensor<4xi64>, tensor<4xi64>)
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3), shots=1)
def test_counts1():
    """
    Test an empty circuit with counts terminal measurement on static wires.
    """
    # CHECK: [[zero:%.+]] = arith.constant 0 : i64
    # CHECK: [[one:%.+]] = arith.constant 1 : i64
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][[[one]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[compbasis:%.+]] = qref.compbasis qubits [[q0]], [[q1]] : !quantum.obs
    # CHECK: [[eigens:%.+]], [[counts:%.+]] = quantum.counts [[compbasis]] : tensor<4xf64>, tensor<4xi64>
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<3>
    # CHECK: quantum.device_release
    return qp.counts(wires=[0, 1])


print(test_counts1.mlir)


# CHECK: func.func public @test_counts2(%arg0: tensor<i64>) -> (tensor<4xi64>, tensor<4xi64>)
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3), shots=1)
def test_counts2(i: int):
    """
    Test an empty circuit with counts terminal measurement on dynamic wires.
    """
    # CHECK: [[zero:%.+]] = arith.constant 0 : i64
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[i:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK: [[qi:%.+]] = qref.get [[alloc]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[compbasis:%.+]] = qref.compbasis qubits [[q0]], [[qi]] : !quantum.obs
    # CHECK: [[eigens:%.+]], [[counts:%.+]] = quantum.counts [[compbasis]] : tensor<4xf64>, tensor<4xi64>
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<3>
    # CHECK: quantum.device_release
    return qp.counts(wires=[0, i])


print(test_counts2.mlir)
