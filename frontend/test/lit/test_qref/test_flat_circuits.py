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

import numpy as np
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


# CHECK: func.func public @expval1() -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=2))
def expval1():
    # CHECK: [[zero:%.+]] = arith.constant 0 : i64
    # CHECK: [[alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][[[zero]]] : !qref.reg<2>, i64 -> !qref.bit
    # CHECK: [[obs:%.+]] = qref.namedobs [[q0]][ PauliX] : !quantum.obs
    # CHECK: [[expval:%.+]] = quantum.expval [[obs]] : f64
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<2>
    return qp.expval(qp.PauliX(0))


print(expval1.mlir)


# CHECK: func.func public @expval2() -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def expval2():
    # CHECK: [[two:%.+]] = arith.constant 2 : i64
    # CHECK: [[one:%.+]] = arith.constant 1 : i64
    # CHECK: [[zero:%.+]] = arith.constant 0 : i64
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[obs0:%.+]] = qref.namedobs [[q0]][ PauliX] : !quantum.obs
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][[[one]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[obs1:%.+]] = qref.namedobs [[q1]][ PauliZ] : !quantum.obs
    # CHECK: [[q2:%.+]] = qref.get [[alloc]][[[two]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[obs2:%.+]] = qref.namedobs [[q2]][ Hadamard] : !quantum.obs
    # CHECK: [[obs_tensor:%.+]] = quantum.tensor [[obs0]], [[obs1]], [[obs2]] : !quantum.obs
    # CHECK: [[expval:%.+]] = quantum.expval [[obs_tensor]] : f64
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<3>
    return qp.expval(qp.PauliX(0) @ qp.PauliZ(1) @ qp.Hadamard(2))


print(expval2.mlir)


# CHECK: func.func public @expval3(%arg0: tensor<2x2xcomplex<f64>>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=2))
def expval3():
    A = np.array([[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]])

    # CHECK: [[zero:%.+]] = arith.constant 0 : i64
    # CHECK: [[alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][[[zero]]] : !qref.reg<2>, i64 -> !qref.bit
    # CHECK: [[obs:%.+]] = qref.hermitian(%arg0 : tensor<2x2xcomplex<f64>>) [[q0]] : !quantum.obs
    # CHECK: [[expval:%.+]] = quantum.expval [[obs]] : f64
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<2>
    return qp.expval(qp.Hermitian(A, wires=0))


print(expval3.mlir)


# CHECK: func.func public @expval4(%arg0: tensor<4x4xcomplex<f64>>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=2))
def expval4():
    B = np.array(
        [
            [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
        ]
    )

    # CHECK: [[one:%.+]] = arith.constant 1 : i64
    # CHECK: [[zero:%.+]] = arith.constant 0 : i64
    # CHECK: [[alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][[[zero]]] : !qref.reg<2>, i64 -> !qref.bit
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][[[one]]] : !qref.reg<2>, i64 -> !qref.bit
    # CHECK: [[obs:%.+]] = qref.hermitian(%arg0 : tensor<4x4xcomplex<f64>>) [[q0]], [[q1]] : !quantum.obs
    # CHECK: [[expval:%.+]] = quantum.expval [[obs]] : f64
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<2>
    return qp.expval(qp.Hermitian(B, wires=[0, 1]))


print(expval4.mlir)


# CHECK: func.func public @expval5(%arg0: tensor<4x4xcomplex<f64>>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def expval5():
    B = np.array(
        [
            [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
        ]
    )

    # CHECK: [[two:%.+]] = arith.constant 2 : i64
    # CHECK: [[one:%.+]] = arith.constant 1 : i64
    # CHECK: [[zero:%.+]] = arith.constant 0 : i64
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][[[one]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[obs1:%.+]] = qref.namedobs [[q1]][ PauliX] : !quantum.obs
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[alloc]][[[two]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[obs2:%.+]] = qref.hermitian(%arg0 : tensor<4x4xcomplex<f64>>) [[q0]], [[q2]] : !quantum.obs
    # CHECK: [[obs_tensor:%.+]] = quantum.tensor [[obs1]], [[obs2]] : !quantum.obs
    # CHECK: [[expval:%.+]] = quantum.expval [[obs_tensor]] : f64
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<3>
    return qp.expval(qp.PauliX(1) @ qp.Hermitian(B, wires=[0, 2]))


print(expval5.mlir)


# CHECK: func.func public @expval6() -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def expval6():
    # CHECK: [[two:%.+]] = arith.constant 2 : i64
    # CHECK: [[one:%.+]] = arith.constant 1 : i64
    # CHECK: [[zero:%.+]] = arith.constant 0 : i64
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[obs0x:%.+]] = qref.namedobs [[q0]][ PauliX] : !quantum.obs
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][[[one]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[obs1z:%.+]] = qref.namedobs [[q1]][ PauliZ] : !quantum.obs
    # CHECK: [[t0:%.+]] = quantum.tensor [[obs0x]], [[obs1z]] : !quantum.obs
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][[[zero]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[obs0z:%.+]] = qref.namedobs [[q0]][ PauliZ] : !quantum.obs
    # CHECK: [[q2:%.+]] = qref.get [[alloc]][[[two]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[obs2h:%.+]] = qref.namedobs [[q2]][ Hadamard] : !quantum.obs
    # CHECK: [[t1:%.+]] = quantum.tensor [[obs0z]], [[obs2h]] : !quantum.obs
    # CHECK: [[obs:%.+]] = quantum.hamiltonian({{%.+}} : tensor<2xf64>) [[t0]], [[t1]]
    # CHECK: [[expval:%.+]] = quantum.expval [[obs]] : f64
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<3>

    coeffs = np.array([0.2, -0.543])
    obs = [qp.PauliX(0) @ qp.PauliZ(1), qp.PauliZ(0) @ qp.Hadamard(2)]
    return qp.expval(qp.Hamiltonian(coeffs, obs))


print(expval6.mlir)
