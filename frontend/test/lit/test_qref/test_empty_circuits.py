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
Unit tests for lowering allocation and observable primitives to reference semantics MLIR during
PLxPR conversion.

All circuits in this test file are empty, aka only an allocation at the front, and observables
for the terminal measurements. There are no gates.
"""

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
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][ 1] : !qref.reg<3> -> !qref.bit
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
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<3> -> !qref.bit
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
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][ 1] : !qref.reg<3> -> !qref.bit
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
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<3> -> !qref.bit
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
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][ 1] : !qref.reg<3> -> !qref.bit
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
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<3> -> !qref.bit
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
    """
    Test an empty circuit with expval terminal measurement on a simple named observable.
    """
    # CHECK: [[alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<2> -> !qref.bit
    # CHECK: [[obs:%.+]] = qref.namedobs [[q0]][ PauliX] : !quantum.obs
    # CHECK: [[expval:%.+]] = quantum.expval [[obs]] : f64
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<2>
    return qp.expval(qp.PauliX(0))


print(expval1.mlir)


# CHECK: func.func public @expval2() -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def expval2():
    """
    Test an empty circuit with expval terminal measurement on a tensor product observable.
    """
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[obs0:%.+]] = qref.namedobs [[q0]][ PauliX] : !quantum.obs
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][ 1] : !qref.reg<3> -> !qref.bit
    # CHECK: [[obs1:%.+]] = qref.namedobs [[q1]][ PauliZ] : !quantum.obs
    # CHECK: [[q2:%.+]] = qref.get [[alloc]][ 2] : !qref.reg<3> -> !qref.bit
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
    """
    Test an empty circuit with expval terminal measurement on a Hermitian observable.
    """
    A = np.array([[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]])

    # CHECK: [[alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<2> -> !qref.bit
    # CHECK: [[obs:%.+]] = qref.hermitian(%arg0 : tensor<2x2xcomplex<f64>>) [[q0]] : !quantum.obs
    # CHECK: [[expval:%.+]] = quantum.expval [[obs]] : f64
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<2>
    return qp.expval(qp.Hermitian(A, wires=0))


print(expval3.mlir)


# CHECK: func.func public @expval4(%arg0: tensor<4x4xcomplex<f64>>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=2))
def expval4():
    """
    Test an empty circuit with expval terminal measurement on a Hermitian observable on multiple
    wires.
    """
    B = np.array(
        [
            [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
        ]
    )

    # CHECK: [[alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<2> -> !qref.bit
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][ 1] : !qref.reg<2> -> !qref.bit
    # CHECK: [[obs:%.+]] = qref.hermitian(%arg0 : tensor<4x4xcomplex<f64>>) [[q0]], [[q1]] : !quantum.obs
    # CHECK: [[expval:%.+]] = quantum.expval [[obs]] : f64
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<2>
    return qp.expval(qp.Hermitian(B, wires=[0, 1]))


print(expval4.mlir)


# CHECK: func.func public @expval5(%arg0: tensor<4x4xcomplex<f64>>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def expval5():
    """
    Test an empty circuit with expval terminal measurement on a tensor product between named and
    Hermitian observables.
    """
    B = np.array(
        [
            [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
        ]
    )

    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][ 1] : !qref.reg<3> -> !qref.bit
    # CHECK: [[obs1:%.+]] = qref.namedobs [[q1]][ PauliX] : !quantum.obs
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[q2:%.+]] = qref.get [[alloc]][ 2] : !qref.reg<3> -> !qref.bit
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
    """
    Test an empty circuit with expval terminal measurement on a Hamiltonian observable, with the
    terms being tensor products.
    """
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[obs0x:%.+]] = qref.namedobs [[q0]][ PauliX] : !quantum.obs
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][ 1] : !qref.reg<3> -> !qref.bit
    # CHECK: [[obs1z:%.+]] = qref.namedobs [[q1]][ PauliZ] : !quantum.obs
    # CHECK: [[t0:%.+]] = quantum.tensor [[obs0x]], [[obs1z]] : !quantum.obs
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[obs0z:%.+]] = qref.namedobs [[q0]][ PauliZ] : !quantum.obs
    # CHECK: [[q2:%.+]] = qref.get [[alloc]][ 2] : !qref.reg<3> -> !qref.bit
    # CHECK: [[obs2h:%.+]] = qref.namedobs [[q2]][ Hadamard] : !quantum.obs
    # CHECK: [[t1:%.+]] = quantum.tensor [[obs0z]], [[obs2h]] : !quantum.obs
    # CHECK: [[obs:%.+]] = quantum.hamiltonian({{%.+}} : tensor<2xf64>) [[t0]], [[t1]]
    # CHECK: [[expval:%.+]] = quantum.expval [[obs]] : f64
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<3>

    coeffs = np.array([0.2, -0.543])
    obs = [qp.PauliX(0) @ qp.PauliZ(1), qp.PauliZ(0) @ qp.Hadamard(2)]
    return qp.expval(qp.Hamiltonian(coeffs, obs))


print(expval6.mlir)


# CHECK: func.func public @expval7(%arg0: tensor<4x4xcomplex<f64>>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=2))
def expval7():
    """
    Test an empty circuit with expval terminal measurement on a Hamiltonian observable, with the
    terms being Hermitian and named observables.
    """
    coeff = np.array([0.8, 0.2])
    obs_matrix = np.array(
        [
            [0.5, 1.0j, 0.0, -3j],
            [-1.0j, -1.1, 0.0, -0.1],
            [0.0, 0.0, -0.9, 12.0],
            [3j, -0.1, 12.0, 0.0],
        ]
    )

    # CHECK: [[alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<2> -> !qref.bit
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][ 1] : !qref.reg<2> -> !qref.bit
    # CHECK: [[hermitian:%.+]] = qref.hermitian(%arg0 : tensor<4x4xcomplex<f64>>) [[q0]], [[q1]] : !quantum.obs
    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<2> -> !qref.bit
    # CHECK: [[obs0x:%.+]] = qref.namedobs [[q0]][ PauliX] : !quantum.obs
    # CHECK: [[hamiltonian:%.+]] = quantum.hamiltonian({{%.+}} : tensor<2xf64>) [[hermitian]], [[obs0x]] : !quantum.obs
    # CHECK: [[expval:%.+]] = quantum.expval [[hamiltonian]] : f64
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<2>

    obs = qp.Hermitian(obs_matrix, wires=[0, 1])
    return qp.expval(qp.Hamiltonian(coeff, [obs, qp.PauliX(0)]))


print(expval7.mlir)


# CHECK: func.func public @var1(%arg0: tensor<i64>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=2))
def var1(i: int):
    """
    Test an empty circuit with variance terminal measurement on a dynamic wires.
    """
    # CHECK: [[alloc:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK: [[i:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK: [[qi:%.+]] = qref.get [[alloc]][[[i]]] : !qref.reg<2>, i64 -> !qref.bit
    # CHECK: [[obs:%.+]] = qref.namedobs [[qi]][ PauliX] : !quantum.obs
    # CHECK: [[var:%.+]] = quantum.var [[obs]] : f64
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<2>
    return qp.var(qp.PauliX(i))


print(var1.mlir)


# CHECK: func.func public @var2(%arg0: tensor<4x4xcomplex<f64>>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<f64>
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def var2(i: int, j: int):
    """
    Test an empty circuit with variance terminal measurement on a dynamic wires, with a tensor
    product of named and Hermitian observables.
    """
    B = np.array(
        [
            [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
        ]
    )

    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>
    # CHECK: [[q1:%.+]] = qref.get [[alloc]][ 1] : !qref.reg<3> -> !qref.bit
    # CHECK: [[obs1x:%.+]] = qref.namedobs [[q1]][ PauliX] : !quantum.obs
    # CHECK: [[i:%.+]] = tensor.extract %arg1[] : tensor<i64>
    # CHECK: [[qi:%.+]] = qref.get [[alloc]][[[i]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[j:%.+]] = tensor.extract %arg2[] : tensor<i64>
    # CHECK: [[qj:%.+]] = qref.get [[alloc]][[[j]]] : !qref.reg<3>, i64 -> !qref.bit
    # CHECK: [[hermitian:%.+]] = qref.hermitian(%arg0 : tensor<4x4xcomplex<f64>>) [[qi]], [[qj]] : !quantum.obs
    # CHECK: [[tensor:%.+]] = quantum.tensor [[obs1x]], [[hermitian]] : !quantum.obs
    # CHECK: [[var:%.+]] = quantum.var [[tensor]] : f64
    # CHECK: qref.dealloc [[alloc]] : !qref.reg<3>
    return qp.var(qp.PauliX(1) @ qp.Hermitian(B, wires=[i, j]))


print(var2.mlir)


# CHECK: func.func public @test_multiple_terminal_measurements() -> (tensor<8xf64>, tensor<1000x1xi64>, tensor<f64>, tensor<f64>)
@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3), shots=1000)
def test_multiple_terminal_measurements():
    """
    Test an empty circuit with multiple terminal measurements.
    """
    # CHECK: quantum.device
    # CHECK: [[alloc:%.+]] = qref.alloc( 3) : !qref.reg<3>

    # CHECK: [[compbasis:%.+]] = qref.compbasis(qreg [[alloc]] : !qref.reg<3>) : !quantum.obs
    # CHECK: [[probs:%.+]] = quantum.probs [[compbasis]] : tensor<8xf64>

    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[compbasis:%.+]] = qref.compbasis qubits [[q0]] : !quantum.obs
    # CHECK: [[sample:%.+]] = quantum.sample [[compbasis]] : tensor<1000x1xf64>

    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[Xobs:%.+]] = qref.namedobs [[q0]][ PauliX] : !quantum.obs
    # CHECK: [[expval:%.+]] = quantum.expval [[Xobs]] : f64

    # CHECK: [[q0:%.+]] = qref.get [[alloc]][ 0] : !qref.reg<3> -> !qref.bit
    # CHECK: [[Yobs:%.+]] = qref.namedobs [[q0]][ PauliY] : !quantum.obs
    # CHECK: [[var:%.+]] = quantum.var [[Yobs]] : f64

    # CHECK: qref.dealloc [[alloc]] : !qref.reg<3>
    # CHECK: quantum.device_release

    return qp.probs(), qp.sample(wires=[0]), qp.expval(qp.X(0)), qp.var(qp.Y(0))


print(test_multiple_terminal_measurements.mlir)


# CHECK: func.func public @jit_test_pre_post_processing(%arg0: tensor<i64>)
@qp.qjit(capture=True, target="mlir")
def test_pre_post_processing(i: int):
    """
    Test converting a workflow with pre and post processing.
    """
    # CHECK: [[one:%.+]] = stablehlo.constant dense<1> : tensor<i64>
    # CHECK: [[arg0_plus_one:%.+]] = stablehlo.add %arg0, [[one]] : tensor<i64>
    # CHECK: [[circuit_out:%.+]]:2 = catalyst.launch_kernel @module_circuit::@circuit(%arg0, [[arg0_plus_one]])
    # CHECK-SAME:   (tensor<i64>, tensor<i64>) -> (tensor<f64>, tensor<f64>)
    # CHECK: [[add:%.+]] = stablehlo.add [[circuit_out]]#0, [[circuit_out]]#1 : tensor<f64>
    # CHECK: return [[add]] : tensor<f64>

    # CHECK: func.func public @circuit(%arg0: tensor<i64>, %arg1: tensor<i64>) -> (tensor<f64>, tensor<f64>)
    # CHECK:   [[reg:%.+]] = qref.alloc( 2) : !qref.reg<2>
    # CHECK:   [[j:%.+]] = tensor.extract %arg0[] : tensor<i64>
    # CHECK:   [[qj:%.+]] = qref.get [[reg]][[[j]]] : !qref.reg<2>, i64 -> !qref.bit
    # CHECK:   [[Xobs:%.+]] = qref.namedobs [[qj]][ PauliX] : !quantum.obs
    # CHECK:   [[Xexpval:%.+]] = quantum.expval [[Xobs]] : f64
    # CHECK:   [[Xexpval_tensor:%.+]] = tensor.from_elements [[Xexpval]] : tensor<f64>

    # CHECK:   [[k:%.+]] = tensor.extract %arg1[] : tensor<i64>
    # CHECK:   [[qk:%.+]] = qref.get [[reg]][[[k]]] : !qref.reg<2>, i64 -> !qref.bit
    # CHECK:   [[Yobs:%.+]] = qref.namedobs [[qk]][ PauliY] : !quantum.obs
    # CHECK:   [[Yexpval:%.+]] = quantum.expval [[Yobs]] : f64
    # CHECK:   [[Yexpval_tensor:%.+]] = tensor.from_elements [[Yexpval]] : tensor<f64>

    # CHECK:   qref.dealloc [[reg]] : !qref.reg<2>
    # CHECK:   return [[Xexpval_tensor]], [[Yexpval_tensor]] : tensor<f64>, tensor<f64>

    @qp.qnode(qp.device("null.qubit", wires=2))
    def circuit(j: int, k: int):
        return qp.expval(qp.X(j)), qp.expval(qp.Y(k))

    a, b = circuit(i, i + 1)
    return a + b


print(test_pre_post_processing.mlir)
