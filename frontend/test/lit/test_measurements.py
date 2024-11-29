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

import jax
import numpy as np
import pennylane as qml

from catalyst import CompileError, qjit
from catalyst.jax_primitives import (
    compbasis_p,
    counts_p,
    sample_p
)

# TODO: NOTE:
# The tests sample1 and sample2 below used to pass, before verification steps were added in the
# device preprocessing. Now that the measurement validation is run, the circuit below complains
# (observables with MeasurementProcess types other than ExpectationMP and VarianceMP are not
# currently supported).
#
# These tests are commented out and the expected output is also commented out using the FileCheck
# comments (COM:).

try:
    # COM: CHECK-LABEL: public @sample1(
    @qjit(target="mlir")
    @qml.qnode(qml.device("lightning.qubit", wires=2, shots=1000))
    def sample1(x: float, y: float):
        qml.RX(x, wires=0)
        qml.RY(y, wires=1)
        # COM: CHECK: [[q0:%.+]] = quantum.custom "RZ"
        qml.RZ(0.1, wires=0)

        # COM: CHECK: [[obs:%.+]] = quantum.namedobs [[q0]][ PauliZ]
        # COM: CHECK: quantum.sample [[obs]] {shots = 1000 : i64} : tensor<1000xf64>
        return qml.sample(qml.PauliZ(0))

    print(sample1.mlir)

    # COM: CHECK-LABEL: public @sample2(
    @qjit(target="mlir")
    @qml.qnode(qml.device("lightning.qubit", wires=2, shots=1000))
    def sample2(x: float, y: float):
        qml.RX(x, wires=0)
        # COM: CHECK: [[q1:%.+]] = quantum.custom "RY"
        qml.RY(y, wires=1)
        # COM: CHECK: [[q0:%.+]] = quantum.custom "RZ"
        qml.RZ(0.1, wires=0)

        # COM: CHECK: [[obs1:%.+]] = quantum.namedobs [[q1]][ PauliX]
        # COM: CHECK: [[obs2:%.+]] = quantum.namedobs [[q0]][ Identity]
        # COM: CHECK: [[obs3:%.+]] = quantum.tensor [[obs1]], [[obs2]]
        # COM: CHECK: quantum.sample [[obs3]] {shots = 1000 : i64} : tensor<1000xf64>
        return qml.sample(qml.PauliX(1) @ qml.Identity(0))

    print(sample2.mlir)
except CompileError:
    ...


# CHECK-LABEL: public @sample3(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2, shots=1000))
# CHECK: [[shots:%.+]] = arith.constant 1000 : i64
# CHECK: quantum.device shots([[shots]]) [{{.+}}]
def sample3(x: float, y: float):
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = quantum.custom "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = quantum.custom "RZ"
    qml.RZ(0.1, wires=0)

    # CHECK: [[obs:%.+]] = quantum.compbasis [[q0]], [[q1]]
    # CHECK: quantum.sample [[obs]] : tensor<1000x2xf64>
    return qml.sample()


print(sample3.mlir)


# CHECK-LABEL: public @test_sample_static(
@qjit
@qml.qnode(
    qml.device("null.qubit", wires=1)
)  # SampleOp is only legal if there is a device in the same scope
def test_sample_static():
    """Test that the sample primitive can be correctly compiled to mlir."""
    obs = compbasis_p.bind()
    return sample_p.bind(obs, shots=5, num_qubits=0)

# CHECK: [[obs:%.+]] = quantum.compbasis  : !quantum.obs
# CHECK: [[sample:%.+]] = quantum.sample [[obs]] : tensor<5x0xf64>
# CHECK: return [[sample]] : tensor<5x0xf64>
print(test_sample_static.mlir)


# TODO: convert the device to have a dynamic shots value when core PennyLane device supports it
# CHECK-LABEL: public @test_sample_dynamic(
@qjit
@qml.qnode(
    qml.device("null.qubit", wires=1)
)  # SampleOp is only legal if there is a device in the same scope
def test_sample_dynamic(shots: int):
    """Test that the sample primitive with dynamic shape can be correctly compiled to mlir."""
    obs = compbasis_p.bind()
    x = shots + 1
    sample = sample_p.bind(obs, x, num_qubits=0)
    return sample + jax.numpy.zeros((x, 0))

# CHECK: [[one:%.+]] = stablehlo.constant dense<1> : tensor<i64>
# CHECK: [[obs:%.+]] = quantum.compbasis  : !quantum.obs
# CHECK: [[plusOne:%.+]] = stablehlo.add %arg0, [[one]] : tensor<i64>
# CHECK: [[sample:%.+]] = quantum.sample [[obs]] : tensor<?x0xf64>
# CHECK: [[zeroVec:%.+]] = stablehlo.dynamic_broadcast_in_dim {{.+}} -> tensor<?x0xf64>
# CHECK: [[outVecSum:%.+]] = stablehlo.add [[sample]], [[zeroVec]] : tensor<?x0xf64>
# CHECK: return [[plusOne]], [[outVecSum]] : tensor<i64>, tensor<?x0xf64>
print(test_sample_dynamic.mlir)



# TODO: NOTE:
# The tests below used to pass before the compiler driver (in the case of counts2) and device
# preprocessing verification (in the case of counts1). Now that the validation is run, the circuits
# below complain.
#
# These tests are commented out and the expected output is also commented out using the FileCheck
# comments (COM:).
#
try:

    # COM: CHECK-LABEL: public @counts1(
    @qjit(target="mlir")
    @qml.qnode(qml.device("lightning.qubit", wires=2, shots=1000))
    def counts1(x: float, y: float):
        qml.RX(x, wires=0)
        qml.RY(y, wires=1)
        # COM: CHECK: [[q0:%.+]] = quantum.custom "RZ"
        qml.RZ(0.1, wires=0)

        # COM: CHECK: [[obs:%.+]] = quantum.namedobs [[q0]][ PauliZ]
        # COM: CHECK: quantum.counts [[obs]] {shots = 1000 : i64} : tensor<2xf64>, tensor<2xi64>
        return qml.counts(qml.PauliZ(0))

    print(counts1.mlir)

    @qjit(target="mlir")
    @qml.qnode(qml.device("lightning.qubit", wires=2, shots=1000))
    def counts2(x: float, y: float):
        qml.RX(x, wires=0)
        # COM: CHECK: [[q1:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RY"
        qml.RY(y, wires=1)
        # COM: CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
        qml.RZ(0.1, wires=0)

        # COM: CHECK: [[obs1:%.+]] = "quantum.namedobs"([[q1]]) {type = #quantum<named_observable PauliX>}
        # COM: CHECK: [[obs2:%.+]] = "quantum.namedobs"([[q0]]) {type = #quantum<named_observable Identity>}
        # COM: CHECK: [[obs3:%.+]] = "quantum.tensor"([[obs1]], [[obs2]])
        # COM: CHECK: "quantum.counts"([[obs3]]) {{.*}}shots = 1000 : i64{{.*}} : (!quantum.obs) -> (tensor<2xf64>, tensor<2xi64>)
        return qml.counts(qml.PauliX(1) @ qml.Identity(0))

    print(counts2.mlir)
except:
    ...


# CHECK-LABEL: public @counts3(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2, shots=1000))
# CHECK: [[shots:%.+]] = arith.constant 1000 : i64
# CHECK: quantum.device shots([[shots]]) [{{.+}}]
def counts3(x: float, y: float):
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = quantum.custom "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = quantum.custom "RZ"
    qml.RZ(0.1, wires=0)

    # CHECK: [[obs:%.+]] = quantum.compbasis [[q0]], [[q1]]
    # CHECK: quantum.counts [[obs]] : tensor<4xf64>, tensor<4xi64>
    return qml.counts()


print(counts3.mlir)


# CHECK-LABEL: public @expval1(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def expval1(x: float, y: float):
    qml.RX(x, wires=0)
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = quantum.custom "RZ"
    qml.RZ(0.1, wires=0)

    # CHECK: [[obs:%.+]] = quantum.namedobs [[q0]][ PauliX]
    # CHECK: quantum.expval [[obs]] : f64
    return qml.expval(qml.PauliX(0))


print(expval1.mlir)


# CHECK-LABEL: public @expval2(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def expval2(x: float, y: float):
    # CHECK: [[q0:%.+]] = quantum.custom "RX"
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = quantum.custom "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q2:%.+]] = quantum.custom "RZ"
    qml.RZ(0.1, wires=2)

    # CHECK: [[p1:%.+]] = quantum.namedobs [[q0]][ PauliX]
    # CHECK: [[p2:%.+]] = quantum.namedobs [[q1]][ PauliZ]
    # CHECK: [[p3:%.+]] = quantum.namedobs [[q2]][ Hadamard]
    # CHECK: [[t0:%.+]] = quantum.tensor [[p1]], [[p2]], [[p3]]
    # CHECK: quantum.expval [[t0]] : f64
    return qml.expval(qml.PauliX(0) @ qml.PauliZ(1) @ qml.Hadamard(2))


print(expval2.mlir)


# CHECK-LABEL: public @expval3(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def expval3():
    A = np.array([[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]])

    # CHECK: [[obs:%.+]] = quantum.hermitian
    # CHECK: quantum.expval [[obs]] : f64
    return qml.expval(qml.Hermitian(A, wires=0))


print(expval3.mlir)


# CHECK-LABEL: public @expval4(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def expval4():
    B = np.array(
        [
            [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
        ]
    )

    # CHECK: [[obs:%.+]] = quantum.hermitian
    # CHECK: quantum.expval [[obs]] : f64
    return qml.expval(qml.Hermitian(B, wires=[0, 1]))


print(expval4.mlir)


# CHECK-LABEL: public @expval5(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def expval5(x: float, y: float):
    # CHECK: [[q0:%.+]] = quantum.custom "RX"
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = quantum.custom "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q2:%.+]] = quantum.custom "RZ"
    qml.RZ(0.1, wires=2)

    B = np.array(
        [
            [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
        ]
    )

    # CHECK: [[p0:%.+]] = quantum.namedobs [[q1]][ PauliX]
    # CHECK: [[h0:%.+]] = quantum.hermitian({{%.+}} : tensor<4x4xcomplex<f64>>) [[q0]], [[q2]]
    # CHECK: [[obs:%.+]] = quantum.tensor [[p0]], [[h0]]
    # CHECK: quantum.expval [[obs]] : f64
    return qml.expval(qml.PauliX(1) @ qml.Hermitian(B, wires=[0, 2]))


print(expval5.mlir)


# CHECK-LABEL: public @expval5(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def expval5(x: float, y: float):
    # CHECK: [[q0:%.+]] = quantum.custom "RX"
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = quantum.custom "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q2:%.+]] = quantum.custom "RZ"
    qml.RZ(0.1, wires=2)

    coeffs = np.array([0.2, -0.543])
    obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]

    # CHECK: [[n0:%.+]] = quantum.namedobs [[q0]][ PauliX]
    # CHECK: [[n1:%.+]] = quantum.namedobs [[q1]][ PauliZ]
    # CHECK: [[t0:%.+]] = quantum.tensor [[n0]], [[n1]]
    # CHECK: [[n2:%.+]] = quantum.namedobs [[q0]][ PauliZ]
    # CHECK: [[n3:%.+]] = quantum.namedobs [[q2]][ Hadamard]
    # CHECK: [[t1:%.+]] = quantum.tensor [[n2]], [[n3]]
    # CHECK: [[obs:%.+]] = quantum.hamiltonian({{%.+}} : tensor<2xf64>) [[t0]], [[t1]]
    # CHECK: quantum.expval [[obs]] : f64
    return qml.expval(qml.Hamiltonian(coeffs, obs))


print(expval5.mlir)


# CHECK-LABEL: public @expval6(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def expval6(x: float):
    # CHECK: [[q0:%.+]] = quantum.custom "RX"
    qml.RX(x, wires=0)

    coeff = np.array([0.8, 0.2])
    obs_matrix = np.array(
        [
            [0.5, 1.0j, 0.0, -3j],
            [-1.0j, -1.1, 0.0, -0.1],
            [0.0, 0.0, -0.9, 12.0],
            [3j, -0.1, 12.0, 0.0],
        ]
    )

    # CHECK: [[h0:%.+]] = quantum.hermitian
    obs = qml.Hermitian(obs_matrix, wires=[0, 1])

    # CHECK: [[n0:%.+]] = quantum.namedobs [[q0]][ PauliX]
    # CHECK: [[obs:%.+]] = quantum.hamiltonian({{%.+}} : tensor<2xf64>) [[h0]], [[n0]]
    # CHECK: quantum.expval [[obs]] : f64
    return qml.expval(qml.Hamiltonian(coeff, [obs, qml.PauliX(0)]))


print(expval6.mlir)


# CHECK-LABEL: public @expval7(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def expval7():
    A = np.array([[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]])

    # CHECK: [[obs:%.+]] = quantum.hermitian
    # CHECK: quantum.expval [[obs]] : f64
    return qml.expval(qml.Hermitian(A, wires=0))


print(expval7.mlir)


# CHECK-LABEL: public @expval8(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def expval8():
    B = np.array(
        [
            [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
        ]
    )

    # CHECK: [[obs:%.+]] = quantum.hermitian
    # CHECK: quantum.expval [[obs]] : f64
    return qml.expval(qml.Hermitian(B, wires=[0, 1]))


print(expval8.mlir)


# CHECK-LABEL: public @expval9(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def expval9(x: float, y: float):
    # CHECK: [[q0:%.+]] = quantum.custom "RX"
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = quantum.custom "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q2:%.+]] = quantum.custom "RZ"
    qml.RZ(0.1, wires=2)

    # CHECK: [[p1:%.+]] = quantum.namedobs [[q0]][ PauliX]
    # CHECK: [[p2:%.+]] = quantum.namedobs [[q1]][ PauliZ]
    # CHECK: [[p3:%.+]] = quantum.namedobs [[q2]][ Hadamard]
    # CHECK: [[obs:%.+]] = quantum.tensor [[p1]], [[p2]], [[p3]]
    # CHECK: quantum.expval [[obs]] : f64
    return qml.expval(qml.PauliX(0) @ qml.PauliZ(1) @ qml.Hadamard(2))


print(expval9.mlir)


# CHECK-LABEL: public @expval10(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def expval10(x: float, y: float):
    # CHECK: [[q0:%.+]] = quantum.custom "RX"
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = quantum.custom "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q2:%.+]] = quantum.custom "RZ"
    qml.RZ(0.1, wires=2)

    B = np.array(
        [
            [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
        ]
    )

    # CHECK: [[p0:%.+]] = quantum.namedobs [[q1]][ PauliX]
    # CHECK: [[h0:%.+]] = quantum.hermitian({{%.+}} : tensor<4x4xcomplex<f64>>) [[q0]], [[q2]]
    # CHECK: [[obs:%.+]] = quantum.tensor [[p0]], [[h0]]
    # CHECK: quantum.expval [[obs]] : f64
    return qml.expval(qml.PauliX(1) @ qml.Hermitian(B, wires=[0, 2]))


print(expval10.mlir)


# CHECK-LABEL: public @var1(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def var1(x: float, y: float):
    qml.RX(x, wires=0)
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = quantum.custom "RZ"
    qml.RZ(0.1, wires=0)

    # CHECK: [[obs:%.+]] = quantum.namedobs [[q0]][ PauliX]
    # CHECK: quantum.var [[obs]] : f64
    return qml.var(qml.PauliX(0))


print(var1.mlir)


# CHECK-LABEL: public @var2(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def var2(x: float, y: float):
    qml.RX(x, wires=0)
    qml.RY(y, wires=1)
    qml.RZ(0.1, wires=2)

    B = np.array(
        [
            [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
        ]
    )

    # CHECK: [[obs:%.+]] = quantum.tensor
    # CHECK: quantum.var [[obs]] : f64
    return qml.var(qml.PauliX(1) @ qml.Hermitian(B, wires=[0, 2]))


print(var2.mlir)


# CHECK-LABEL: public @probs1(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def probs1(x: float, y: float):
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = quantum.custom "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = quantum.custom "RZ"
    qml.RZ(0.1, wires=0)

    # qml.probs()  # unsupported by PennyLane
    # qml.probs(op=qml.PauliX(0))  # unsupported by the compiler

    # CHECK: [[obs:%.+]] = quantum.compbasis [[q0]], [[q1]]
    # CHECK: quantum.probs [[obs]] : tensor<4xf64>
    return qml.probs(wires=[0, 1])


print(probs1.mlir)


# CHECK-LABEL: public @state1(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def state1(x: float, y: float):
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = quantum.custom "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = quantum.custom "RZ"
    qml.RZ(0.1, wires=0)

    # qml.state(wires=[0])  # unsupported by PennyLane

    # CHECK: [[obs:%.+]] = quantum.compbasis [[q0]], [[q1]]
    # CHECK: quantum.state [[obs]] : tensor<4xcomplex<f64>>
    return qml.state()


print(state1.mlir)
