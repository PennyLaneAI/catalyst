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

import numpy as np
import pennylane as qml

from catalyst import qjit


# CHECK-LABEL: private @sample1(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2, shots=1000))
def sample1(x: float, y: float):
    qml.RX(x, wires=0)
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
    qml.RZ(0.1, wires=0)

    # CHECK: [[obs:%.+]] = "quantum.namedobs"([[q0]]) {type = #quantum<named_observable PauliZ>}
    # CHECK: "quantum.sample"([[obs]]) {shots = 1000 : i64} {{.+}} -> tensor<1000xf64>
    return qml.sample(qml.PauliZ(0))


print(sample1.mlir)


# CHECK-LABEL: private @sample2(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2, shots=1000))
def sample2(x: float, y: float):
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
    qml.RZ(0.1, wires=0)

    # CHECK: [[obs1:%.+]] = "quantum.namedobs"([[q1]]) {type = #quantum<named_observable PauliX>}
    # CHECK: [[obs2:%.+]] = "quantum.namedobs"([[q0]]) {type = #quantum<named_observable Identity>}
    # CHECK: [[obs3:%.+]] = "quantum.tensor"([[obs1]], [[obs2]])
    # CHECK: "quantum.sample"([[obs3]]) {shots = 1000 : i64} {{.+}} -> tensor<1000xf64>
    return qml.sample(qml.PauliX(1) @ qml.Identity(0))


print(sample2.mlir)


# CHECK-LABEL: private @sample3(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2, shots=1000))
def sample3(x: float, y: float):
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
    qml.RZ(0.1, wires=0)

    # CHECK: [[obs:%.+]] = "quantum.compbasis"([[q0]], [[q1]])
    # CHECK: "quantum.sample"([[obs]]) {shots = 1000 : i64} {{.+}} -> tensor<1000x2xf64>
    return qml.sample()


print(sample3.mlir)


# CHECK-LABEL: private @counts1(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2, shots=1000))
def counts1(x: float, y: float):
    qml.RX(x, wires=0)
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
    qml.RZ(0.1, wires=0)

    # CHECK: [[obs:%.+]] = "quantum.namedobs"([[q0]]) {type = #quantum<named_observable PauliZ>}
    # CHECK: "quantum.counts"([[obs]]) {{.*}}shots = 1000 : i64{{.*}} : (!quantum.obs) -> (tensor<2xf64>, tensor<2xi64>)
    return qml.counts(qml.PauliZ(0))


print(counts1.mlir)


# CHECK-LABEL: private @counts2(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2, shots=1000))
def counts2(x: float, y: float):
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
    qml.RZ(0.1, wires=0)

    # CHECK: [[obs1:%.+]] = "quantum.namedobs"([[q1]]) {type = #quantum<named_observable PauliX>}
    # CHECK: [[obs2:%.+]] = "quantum.namedobs"([[q0]]) {type = #quantum<named_observable Identity>}
    # CHECK: [[obs3:%.+]] = "quantum.tensor"([[obs1]], [[obs2]])
    # CHECK: "quantum.counts"([[obs3]]) {{.*}}shots = 1000 : i64{{.*}} : (!quantum.obs) -> (tensor<2xf64>, tensor<2xi64>)
    return qml.counts(qml.PauliX(1) @ qml.Identity(0))


print(counts2.mlir)


# CHECK-LABEL: private @counts3(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2, shots=1000))
def counts3(x: float, y: float):
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
    qml.RZ(0.1, wires=0)

    # CHECK: [[obs:%.+]] = "quantum.compbasis"([[q0]], [[q1]])
    # CHECK: "quantum.counts"([[obs]]) {{.*}}shots = 1000 : i64{{.*}} : (!quantum.obs) -> (tensor<4xf64>, tensor<4xi64>)
    return qml.counts()


print(counts3.mlir)


# CHECK-LABEL: private @expval1(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def expval1(x: float, y: float):
    qml.RX(x, wires=0)
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
    qml.RZ(0.1, wires=0)

    # CHECK: [[obs:%.+]] = "quantum.namedobs"([[q0]]) {type = #quantum<named_observable PauliX>}
    # CHECK: "quantum.expval"([[obs]]) {{.+}} -> f64
    return qml.expval(qml.PauliX(0))


print(expval1.mlir)


# CHECK-LABEL: private @expval2(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def expval2(x: float, y: float):
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RX"
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q2:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
    qml.RZ(0.1, wires=2)

    # CHECK: [[p1:%.+]] = "quantum.namedobs"([[q0]]) {type = #quantum<named_observable PauliX>}
    # CHECK: [[p2:%.+]] = "quantum.namedobs"([[q1]]) {type = #quantum<named_observable PauliZ>}
    # CHECK: [[p3:%.+]] = "quantum.namedobs"([[q2]]) {type = #quantum<named_observable Hadamard>}
    # CHECK: [[t0:%.+]] = "quantum.tensor"([[p1]], [[p2]], [[p3]])
    # CHECK: "quantum.expval"([[t0]]) {{.+}} -> f64
    return qml.expval(qml.PauliX(0) @ qml.PauliZ(1) @ qml.Hadamard(2))


print(expval2.mlir)


# CHECK-LABEL: private @expval3(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def expval3():
    A = np.array([[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]])

    # CHECK: [[obs:%.+]] = "quantum.hermitian"({{%.+}}, {{%.+}}) : (tensor<2x2xcomplex<f64>>, !quantum.bit) -> !quantum.obs
    # CHECK: "quantum.expval"([[obs]]) {{.+}} -> f64
    return qml.expval(qml.Hermitian(A, wires=0))


print(expval3.mlir)


# CHECK-LABEL: private @expval4(
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

    # CHECK: [[obs:%.+]] = "quantum.hermitian"({{%.+}}, {{%.+}}, {{%.+}}) : (tensor<4x4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> !quantum.obs
    # CHECK: "quantum.expval"([[obs]]) {{.+}} -> f64
    return qml.expval(qml.Hermitian(B, wires=[0, 1]))


print(expval4.mlir)


# CHECK-LABEL: private @expval5(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def expval5(x: float, y: float):
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RX"
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q2:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
    qml.RZ(0.1, wires=2)

    B = np.array(
        [
            [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
        ]
    )

    # CHECK: [[p0:%.+]] = "quantum.namedobs"([[q1]]) {type = #quantum<named_observable PauliX>}
    # CHECK: [[h0:%.+]] = "quantum.hermitian"({{%.+}}, [[q0]], [[q2]]) : (tensor<4x4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> !quantum.obs
    # CHECK: [[obs:%.+]] = "quantum.tensor"([[p0]], [[h0]])
    # CHECK: "quantum.expval"([[obs]]) {{.+}} -> f64
    return qml.expval(qml.PauliX(1) @ qml.Hermitian(B, wires=[0, 2]))


print(expval5.mlir)


# CHECK-LABEL: private @expval5(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def expval5(x: float, y: float):
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RX"
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q2:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
    qml.RZ(0.1, wires=2)

    coeffs = np.array([0.2, -0.543])
    obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]

    # CHECK: [[n0:%.+]] = "quantum.namedobs"([[q0]]) {type = #quantum<named_observable PauliX>}
    # CHECK: [[n1:%.+]] = "quantum.namedobs"([[q1]]) {type = #quantum<named_observable PauliZ>}
    # CHECK: [[t0:%.+]] = "quantum.tensor"([[n0]], [[n1]])
    # CHECK: [[n2:%.+]] = "quantum.namedobs"([[q0]]) {type = #quantum<named_observable PauliZ>}
    # CHECK: [[n3:%.+]] = "quantum.namedobs"([[q2]]) {type = #quantum<named_observable Hadamard>}
    # CHECK: [[t1:%.+]] = "quantum.tensor"([[n2]], [[n3]])
    # CHECK: [[obs:%.+]] = "quantum.hamiltonian"({{%.+}}, [[t0]], [[t1]]) : (tensor<2xf64>, !quantum.obs, !quantum.obs) -> !quantum.obs
    # CHECK: "quantum.expval"([[obs]]) {{.+}} -> f64
    return qml.expval(qml.Hamiltonian(coeffs, obs))


print(expval5.mlir)


# CHECK-LABEL: private @expval6(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def expval6(x: float):
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RX"
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

    # CHECK: [[h0:%.+]] = "quantum.hermitian"({{%.+}}, {{%.+}}, {{%.+}}) : (tensor<4x4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> !quantum.obs
    obs = qml.Hermitian(obs_matrix, wires=[0, 1])

    # CHECK: [[n0:%.+]] = "quantum.namedobs"([[q0]]) {type = #quantum<named_observable PauliX>}
    # CHECK: [[obs:%.+]] = "quantum.hamiltonian"({{%.+}}, [[h0]], [[n0]]) : (tensor<2xf64>, !quantum.obs, !quantum.obs) -> !quantum.obs
    # CHECK: "quantum.expval"([[obs]]) {{.+}} -> f64
    return qml.expval(qml.Hamiltonian(coeff, [obs, qml.PauliX(0)]))


print(expval6.mlir)


# CHECK-LABEL: private @expval7(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def expval7():
    A = np.array([[complex(1.0, 0.0), complex(2.0, 0.0)], [complex(2.0, 0.0), complex(1.0, 0.0)]])

    # CHECK: [[obs:%.+]] = "quantum.hermitian"({{%.+}}, {{%.+}}) : (tensor<2x2xcomplex<f64>>, !quantum.bit) -> !quantum.obs
    # CHECK: "quantum.expval"([[obs]]) {{.+}} -> f64
    return qml.expval(qml.Hermitian(A, wires=0))


print(expval7.mlir)


# CHECK-LABEL: private @expval8(
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

    # CHECK: [[obs:%.+]] = "quantum.hermitian"({{%.+}}, {{%.+}}, {{%.+}}) : (tensor<4x4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> !quantum.obs
    # CHECK: "quantum.expval"([[obs]]) {{.+}} -> f64
    return qml.expval(qml.Hermitian(B, wires=[0, 1]))


print(expval8.mlir)


# CHECK-LABEL: private @expval9(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def expval9(x: float, y: float):
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RX"
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q2:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
    qml.RZ(0.1, wires=2)

    # CHECK: [[p1:%.+]] = "quantum.namedobs"([[q0]]) {type = #quantum<named_observable PauliX>}
    # CHECK: [[p2:%.+]] = "quantum.namedobs"([[q1]]) {type = #quantum<named_observable PauliZ>}
    # CHECK: [[p3:%.+]] = "quantum.namedobs"([[q2]]) {type = #quantum<named_observable Hadamard>}
    # CHECK: [[obs:%.+]] = "quantum.tensor"([[p1]], [[p2]], [[p3]])
    # CHECK: "quantum.expval"([[obs]]) {{.+}} -> f64
    return qml.expval(qml.PauliX(0) @ qml.PauliZ(1) @ qml.Hadamard(2))


print(expval9.mlir)


# CHECK-LABEL: private @expval10(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=3))
def expval10(x: float, y: float):
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RX"
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q2:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
    qml.RZ(0.1, wires=2)

    B = np.array(
        [
            [complex(1.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(1.0, 0.0), complex(1.0, 0.0), complex(1.0, 0.0), complex(2.0, 0.0)],
            [complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0), complex(2.0, 0.0)],
        ]
    )

    # CHECK: [[p0:%.+]] = "quantum.namedobs"([[q1]]) {type = #quantum<named_observable PauliX>}
    # CHECK: [[h0:%.+]] = "quantum.hermitian"({{%.+}}, [[q0]], [[q2]]) : (tensor<4x4xcomplex<f64>>, !quantum.bit, !quantum.bit) -> !quantum.obs
    # CHECK: [[obs:%.+]] = "quantum.tensor"([[p0]], [[h0]])
    # CHECK: "quantum.expval"([[obs]]) {{.+}} -> f64
    return qml.expval(qml.PauliX(1) @ qml.Hermitian(B, wires=[0, 2]))


print(expval10.mlir)


# CHECK-LABEL: private @var1(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def var1(x: float, y: float):
    qml.RX(x, wires=0)
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
    qml.RZ(0.1, wires=0)

    # CHECK: [[obs:%.+]] = "quantum.namedobs"([[q0]]) {type = #quantum<named_observable PauliX>}
    # CHECK: "quantum.var"([[obs]]) {{.+}} -> f64
    return qml.var(qml.PauliX(0))


print(var1.mlir)


# CHECK-LABEL: private @var2(
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

    # CHECK: [[obs:%.+]] = "quantum.tensor"({{.+}}, {{.+}})
    # CHECK: "quantum.var"([[obs]]) {{.+}} -> f64
    return qml.var(qml.PauliX(1) @ qml.Hermitian(B, wires=[0, 2]))


print(var2.mlir)


# CHECK-LABEL: private @probs1(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def probs1(x: float, y: float):
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
    qml.RZ(0.1, wires=0)

    # qml.probs()  # unsupported by PennyLane
    # qml.probs(op=qml.PauliX(0))  # unsupported by the compiler

    # CHECK: [[obs:%.+]] = "quantum.compbasis"([[q0]], [[q1]])
    # CHECK: "quantum.probs"([[obs]]) {{.+}} -> tensor<4xf64>
    return qml.probs(wires=[0, 1])


print(probs1.mlir)


# CHECK-LABEL: private @state1(
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def state1(x: float, y: float):
    qml.RX(x, wires=0)
    # CHECK: [[q1:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RY"
    qml.RY(y, wires=1)
    # CHECK: [[q0:%.+]] = "quantum.custom"({{%.+}}, {{%.+}}) {gate_name = "RZ"
    qml.RZ(0.1, wires=0)

    # qml.state(wires=[0])  # unsupported by PennyLane

    # CHECK: [[obs:%.+]] = "quantum.compbasis"([[q0]], [[q1]])
    # CHECK: "quantum.state"([[obs]]) {{.+}} -> tensor<4xcomplex<f64>>
    return qml.state()


print(state1.mlir)
