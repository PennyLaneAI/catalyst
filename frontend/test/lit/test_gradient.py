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

# pylint: disable=line-too-long

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml

from catalyst import grad, jacobian, qjit


# CHECK-LABEL: public @jit_grad_default
@qjit(target="mlir")
def grad_default(x: float):
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    # CHECK: gradient.grad "fd" @module_f::@f({{%[a-zA-Z0-9_]+}}) {diffArgIndices = dense<0> : tensor<1xi64>, finiteDiffParam = 9.9999999999999995E-8 : f64} : (tensor<f64>) -> tensor<f64>
    g = grad(f, method="fd")
    return g(jax.numpy.pi)


print(grad_default.mlir)


# CHECK-LABEL: public @jit_override_method
@qjit(target="mlir")
def override_method(x: float):
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    # CHECK: gradient.grad "auto" @module_f::@f({{%[a-zA-Z0-9_]+}}) {diffArgIndices = dense<0> : tensor<1xi64>} : (tensor<f64>) -> tensor<f64>
    g = grad(f, method="auto")
    return g(jax.numpy.pi)


print(override_method.mlir)


# CHECK-LABEL: public @jit_override_h
@qjit(target="mlir")
def override_h(x: float):
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    # CHECK: gradient.grad "fd" @module_f::@f({{%[a-zA-Z0-9_]+}}) {diffArgIndices = dense<0> : tensor<1xi64>, finiteDiffParam = 2.000000e+00 : f64} : (tensor<f64>) -> tensor<f64>
    g = grad(f, method="fd", h=2.0)
    return g(jax.numpy.pi)


print(override_h.mlir)


# CHECK-LABEL: public @jit_override_diff_arg
@qjit(target="mlir")
def override_diff_arg(x: float):
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float, y: float):
        qml.RX(x**y, wires=0)
        return qml.expval(qml.PauliY(0))

    # CHECK: gradient.grad "auto" @module_f::@f({{%[a-zA-Z0-9_]+}}, {{%[a-zA-Z0-9_]+}}_0) {diffArgIndices = dense<1> : tensor<1xi64>} : (tensor<f64>, tensor<f64>) -> tensor<f64>
    g = grad(f, argnums=1)
    return g(jax.numpy.pi, 2.0)


print(override_diff_arg.mlir)


# CHECK-LABEL: public @jit_second_grad
@qjit(target="mlir")
def second_grad(x: float):
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def f(x: float):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliY(0))

    # CHECK: gradient.grad "fd" @grad.f({{%[a-zA-Z0-9_]+}}) {diffArgIndices = dense<0> : tensor<1xi64>, finiteDiffParam = 9.9999999999999995E-8 : f64} : (tensor<f64>) -> tensor<f64>
    g = grad(f)
    # CHECK-LABEL: private @grad.f
    h = grad(g, method="fd")
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

    # CHECK: gradient.grad "auto" @module_f::@f({{%[a-zA-Z0-9_]+}}, {{%[a-zA-Z0-9_]+}}) {diffArgIndices = dense<[0, 1]> : tensor<2xi64>} : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>, tensor<f64>)
    g = jacobian(f, argnums=[0, 1])
    return g(jax.numpy.pi, jax.numpy.pi)


print(grad_range_change.mlir)


# CHECK-LABEL: public @jit_grad_hoist_constant(%arg0
@qjit(target="mlir")
def grad_hoist_constant(params: jax.core.ShapedArray([2], float)):
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def circuit(params):
        qml.CRX(params[0], wires=[0, 1])
        qml.CRX(params[0], wires=[0, 2])
        # CHECK-NEXT: [[const:%.+]] = stablehlo.constant dense<[2.{{0+}}e-01, -5.3{{0+}}e-01]
        h_coeffs = np.array([0.2, -0.53])
        h_obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]
        return qml.expval(qml.Hamiltonian(h_coeffs, h_obs))

    # CHECK-NEXT {{%.+}} = gradient.grad "fd" @module_circuit::@circuit([[const]], %arg0)
    h = grad(circuit, method="fd", argnums=[0])
    return h(params)


print(grad_hoist_constant.mlir)


# CHECK-LABEL: @test_gradient_used_twice
@qjit(target="mlir")
def test_gradient_used_twice(x: float):
    """This tests that calling the return of grad
    more than once does not define multiple functions.
    """

    # CHECK-NOT: @identity_0
    # CHECK-LABEL: @identity
    # CHECK-NOT: @identity_0
    def identity(x):
        return x

    diff_identity = grad(identity)
    return diff_identity(x) + diff_identity(x)


print(test_gradient_used_twice.mlir)


# CHECK-LABEL: @test_gradient_taken_twice
@qjit(target="mlir")
def test_gradient_taken_twice(x: float):
    """This tests that calling grad
    more than once does not define multiple functions.
    """

    # CHECK-NOT: @identity_0
    # CHECK-LABEL: @identity
    # CHECK-NOT: @identity_0
    def identity(x):
        return x

    diff_identity0 = grad(identity)
    diff_identity1 = grad(identity)
    return diff_identity0(x) + diff_identity1(x)


print(test_gradient_taken_twice.mlir)


# CHECK-LABEL: @test_higher_order_used_twice
@qjit(target="mlir")
def test_higher_order_used_twice(x: float):
    """Test that a single function is generated when using higher order derivatives"""

    # CHECK-NOT: @identity_0
    # CHECK-LABEL: @identity
    # CHECK-NOT: @identity_0
    def identity(x):
        return x

    dydx_identity = grad(identity, method="fd")
    dy2dx2_identity = grad(dydx_identity, method="fd")
    return dy2dx2_identity(x) + dy2dx2_identity(x)


print(test_higher_order_used_twice.mlir)

# ---


# CHECK-LABEL: @test_non_diff
@qjit(target="mlir")
def test_non_diff(params: jax.core.ShapedArray([2], float)):
    """Test non-differentiable operations are not decomposed when grad is not used."""

    @qml.qnode(qml.device("lightning.qubit", wires=4), diff_method="adjoint")
    def cost(params: jax.core.ShapedArray([2], float)):
        qml.StatePrep(jnp.ones((2**2,), dtype=float) / 2, wires=range(2))
        qml.Rot(*params, 0.5, wires=0)
        qml.QubitUnitary(jnp.stack([params, params]), wires=[1])
        qml.DoubleExcitation(params[1], wires=[0, 1, 2, 3])
        return qml.expval(qml.PauliZ(0))

    # CHECK: set_state
    # CHECK: "Rot"
    # CHECK: unitary
    return cost(params)


print(test_non_diff.mlir)


# ---


# CHECK-LABEL: @test_diff_const_paramshift
@qjit(target="mlir")
def test_diff_const_paramshift(params: jax.core.ShapedArray([2], float)):
    """Test non-differentiable operations are not decomposed when constant with param-shift."""

    @qml.qnode(qml.device("lightning.qubit", wires=4), diff_method="parameter-shift")
    def cost(params: jax.core.ShapedArray([2], float)):
        qml.StatePrep(np.ones((2**2,), dtype=float) / 2, wires=range(2))
        qml.Rot(0.3, 0.4, 0.5, wires=0)
        qml.QubitUnitary(np.ones((2, 2)), wires=[1])
        qml.DoubleExcitation(params[1], wires=[0, 1, 2, 3])
        return qml.expval(qml.PauliZ(0))

    # CHECK: set_state
    # CHECK: "Rot"
    # CHECK: unitary
    return grad(cost)(params)


print(test_diff_const_paramshift.mlir)

# ---


# CHECK-LABEL: @test_diff_const_adjoint
@qjit(target="mlir")
def test_diff_const_adjoint(params: jax.core.ShapedArray([2], float)):
    """Test non-differentiable operations *are* decomposed even when constant with adjoint.
    State preparations & unitaries are exempted from this and can be skipped if constant.
    """

    @qml.qnode(qml.device("lightning.qubit", wires=4), diff_method="adjoint")
    def cost(params: jax.core.ShapedArray([2], float)):
        qml.StatePrep(np.ones((2**2,), dtype=float) / 2, wires=range(2))
        qml.Rot(0.3, 0.4, 0.5, wires=0)
        qml.QubitUnitary(np.ones((2, 2)), wires=[1])
        qml.DoubleExcitation(params[1], wires=[0, 1, 2, 3])
        return qml.expval(qml.PauliZ(0))

    # CHECK: set_state
    # CHECK-NOT: "Rot"
    # CHECK: unitary
    return grad(cost)(params)


print(test_diff_const_adjoint.mlir)

# ---


# CHECK-LABEL: @test_diff_dynamic_paramshift
@qjit(target="mlir")
def test_diff_dynamic_paramshift(params: jax.core.ShapedArray([2], float)):
    """Test non-differentiable operations are decomposed when active with param-shift."""

    @qml.qnode(qml.device("lightning.qubit", wires=4), diff_method="parameter-shift")
    def cost(params: jax.core.ShapedArray([2], float)):
        qml.StatePrep(jnp.ones((2**2,), dtype=float) / 2, wires=range(2))
        qml.Rot(*params, 0.5, wires=0)
        qml.QubitUnitary(jnp.ones((2, 2)), wires=[1])
        qml.DoubleExcitation(params[1], wires=[0, 1, 2, 3])
        return qml.expval(qml.PauliZ(0))

    # CHECK-NOT: set_state
    # COM: Rot is supported by the two-term shift rule
    # CHECK: "Rot"
    # CHECK-NOT: unitary
    return grad(cost)(params)


print(test_diff_dynamic_paramshift.mlir)

# ---


# CHECK-LABEL: @test_diff_dynamic_adjoint
@qjit(target="mlir")
def test_diff_dynamic_adjoint(params: jax.core.ShapedArray([2], float)):
    """Test non-differentiable operations are decomposed when active with adjoint."""

    @qml.qnode(qml.device("lightning.qubit", wires=4), diff_method="adjoint")
    def cost(params: jax.core.ShapedArray([2], float)):
        qml.StatePrep(jnp.ones((2**2,), dtype=float) / 2, wires=range(2))
        qml.Rot(*params, 0.5, wires=0)
        qml.QubitUnitary(jnp.ones((2, 2)), wires=[1])
        qml.DoubleExcitation(params[1], wires=[0, 1, 2, 3])
        return qml.expval(qml.PauliZ(0))

    # CHECK-NOT: set_state
    # CHECK-NOT: "Rot"
    # CHECK-NOT: unitary
    return grad(cost)(params)


print(test_diff_dynamic_adjoint.mlir)

# ---


# CHECK-LABEL: @test_non_diff_ops_in_cost_and_grad
@qjit(target="mlir")
def test_non_diff_ops_in_cost_and_grad(params: jax.core.ShapedArray([2], float)):
    """Test decomposition when invoking both the function and its derivative."""

    @qml.qnode(qml.device("lightning.qubit", wires=4), diff_method="adjoint")
    def cost(params: jax.core.ShapedArray([2], float)):
        qml.StatePrep(jnp.ones((2**2,), dtype=float) / 2, wires=range(2))
        qml.Rot(*params, 0.5, wires=0)
        qml.QubitUnitary(jnp.ones((2, 2)), wires=[1])
        qml.DoubleExcitation(params[1], wires=[0, 1, 3, 4])
        return qml.expval(qml.PauliZ(0))

    # CHECK func.func public @cost
    # CHECK: set_state
    # CHECK: "Rot"
    # CHECK: unitary

    # CHECK func.func public @cost_1
    # CHECK-NOT: set_state
    # CHECK-NOT: "Rot"
    # CHECK-NOT: unitary
    return cost(params), grad(cost)(params)


print(test_non_diff_ops_in_cost_and_grad.mlir)
