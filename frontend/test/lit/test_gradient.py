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

from catalyst import qjit, grad
import pennylane as qml
import jax
import numpy as np


# CHECK-LABEL: public @jit_grad_default
@qjit(target="mlir")
def grad_default(x: float):
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    # CHECK: "gradient.grad"({{%[0-9]+}}) {callee = @f, diffArgIndices = dense<0> : tensor<1xi64>, finiteDiffParam = 9.9999999999999995E-8 : f64, method = "fd"} : (tensor<f64>) -> tensor<f64>
    g = grad(f)
    return g(jax.numpy.pi)


print(grad_default.mlir)


# CHECK-LABEL: public @jit_override_method
@qjit(target="mlir")
def override_method(x: float):
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    # CHECK: "gradient.grad"({{%[0-9]+}}) {callee = @f, diffArgIndices = dense<0> : tensor<1xi64>, method = "ps"} : (tensor<f64>) -> tensor<f64>
    g = grad(f, method="ps")
    return g(jax.numpy.pi)


print(override_method.mlir)


# CHECK-LABEL: public @jit_override_h
@qjit(target="mlir")
def override_h(x: float):
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    # CHECK: "gradient.grad"({{%[0-9]+}}) {callee = @f, diffArgIndices = dense<0> : tensor<1xi64>, finiteDiffParam = 2.000000e+00 : f64, method = "fd"} : (tensor<f64>) -> tensor<f64>
    g = grad(f, h=2.0)
    return g(jax.numpy.pi)


print(override_h.mlir)


# CHECK-LABEL: public @jit_override_diff_arg
@qjit(target="mlir")
def override_diff_arg(x: float):
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float, y: float):
        qml.RX(x**y, wires=0)
        return qml.expval(qml.PauliY(0))

    # CHECK: "gradient.grad"({{%[0-9]+}}, {{%[0-9]+}}) {callee = @f, diffArgIndices = dense<1> : tensor<1xi64>, finiteDiffParam = 9.9999999999999995E-8 : f64, method = "fd"} : (tensor<f64>, tensor<f64>) -> tensor<f64>
    g = grad(f, argnum=1)
    return g(jax.numpy.pi, 2.0)


print(override_diff_arg.mlir)


# CHECK-LABEL: public @jit_second_grad
@qjit(target="mlir")
def second_grad(x: float):
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    # CHECK: "gradient.grad"({{%[0-9]+}}) {callee = @grad.f, diffArgIndices = dense<0> : tensor<1xi64>, finiteDiffParam = 9.9999999999999995E-8 : f64, method = "fd"} : (tensor<f64>) -> tensor<f64>
    g = grad(f)
    # CHECK-LABEL: private @grad.f
    h = grad(g)
    return h(jax.numpy.pi)


print(second_grad.mlir)


# CHECK-LABEL: public @jit_grad_range_change
@qjit(target="mlir")
def grad_range_change():
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float, y: float):
        qml.RX(x, wires=0)
        qml.RY(y, wires=1)
        return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(1))

    # CHECK: "gradient.grad"({{%[0-9]+}}, {{%[0-9]+}}) {callee = @f, diffArgIndices = dense<[0, 1]> : tensor<2xi64>, finiteDiffParam = 9.9999999999999995E-8 : f64, method = "fd"} : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>)
    g = grad(f, argnum=[0, 1])
    return g(jax.numpy.pi, jax.numpy.pi)


print(grad_range_change.mlir)


# CHECK-LABEL: public @jit_grad_hoist_constant(%arg0
@qjit(target="mlir")
def grad_hoist_constant(params: jax.core.ShapedArray([2], float)):
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def circuit(params):
        qml.CRX(params[0], wires=[0, 1])
        qml.CRX(params[0], wires=[0, 2])
        # CHECK-NEXT: [[const:%.+]] = mhlo.constant dense<[2.{{0+}}e-01, -5.3{{0+}}e-01]
        h_coeffs = np.array([0.2, -0.53])
        h_obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]
        return qml.expval(qml.Hamiltonian(h_coeffs, h_obs))

    # CHECK-NEXT {{%.+}} = "gradient.grad"([[const]], %arg0)
    h = grad(circuit, method="fd", argnum=[0])
    return h(params)


print(grad_hoist_constant.mlir)
