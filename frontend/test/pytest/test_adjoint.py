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
"""Unit tests for the Catalyst adjoint function.
"""

from functools import partial

import jax.numpy as jnp
import pennylane as qml
import pennylane.numpy as pnp
import pytest
from numpy.testing import assert_allclose
from pennylane import adjoint as PL_adjoint

from catalyst import adjoint as C_adjoint
from catalyst import qjit

# pylint: disable=missing-function-docstring


def test_adjoint_func():
    def func():
        qml.PauliX(wires=0)
        qml.PauliY(wires=0)
        qml.PauliZ(wires=1)

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def C_workflow():
        qml.PauliX(wires=0)
        C_adjoint(func)()
        qml.PauliY(wires=0)
        return qml.state()

    @qml.qnode(qml.device("default.qubit", wires=2))
    def PL_workflow():
        qml.PauliX(wires=0)
        PL_adjoint(func)()
        qml.PauliY(wires=0)
        return qml.state()

    actual = C_workflow()
    desired = PL_workflow()
    assert_allclose(actual, desired)


@pytest.mark.parametrize("theta, val", [(jnp.pi, 0), (-100.0, 1)])
def test_adjoint_op(theta, val):
    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def C_workflow(theta, val):
        C_adjoint(qml.RY)(jnp.pi, val)
        C_adjoint(qml.RZ)(theta, wires=val)
        return qml.state()

    @qml.qnode(qml.device("default.qubit", wires=2))
    def PL_workflow(theta, val):
        PL_adjoint(qml.RY)(jnp.pi, val)
        PL_adjoint(qml.RZ)(theta, wires=val)
        return qml.state()

    actual = C_workflow(theta, val)
    desired = PL_workflow(theta, val)
    assert_allclose(actual, desired)


@pytest.mark.parametrize("theta, val", [(pnp.pi, 0), (-100.0, 2)])
def test_adjoint_bound_op(theta, val):
    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def C_workflow(theta, val):
        C_adjoint(qml.RX(jnp.pi, val))
        C_adjoint(qml.PauliY(val))
        C_adjoint(qml.RZ(theta, wires=val))
        return qml.state()

    @qml.qnode(qml.device("default.qubit", wires=3))
    def PL_workflow(theta, val):
        PL_adjoint(qml.RX(jnp.pi, val))
        PL_adjoint(qml.PauliY(val))
        PL_adjoint(qml.RZ(theta, wires=val))
        return qml.state()

    actual = C_workflow(theta, val)
    desired = PL_workflow(theta, val)
    assert_allclose(actual, desired, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("w, p", [(0, 0.5), (0, -100.0), (1, 123.22)])
def test_adjoint_param_fun(w, p):
    def func(w, theta1, theta2, theta3):
        qml.RX(theta1 * pnp.pi / 2, wires=w)
        qml.RY(theta2 / 2, wires=w)
        qml.RZ(theta3, wires=1)

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def C_workflow(w, theta):
        qml.PauliX(wires=0)
        C_adjoint(func)(w, theta, theta, theta)
        qml.PauliY(wires=0)
        return qml.state()

    @qml.qnode(qml.device("default.qubit", wires=2))
    def PL_workflow(w, theta):
        qml.PauliX(wires=0)
        PL_adjoint(func)(w, theta, theta, theta)
        qml.PauliY(wires=0)
        return qml.state()

    actual = C_workflow(w, p)
    desired = PL_workflow(w, p)
    assert_allclose(actual, desired)


def test_adjoint_nested_fun():
    def func(A, I):
        qml.RX(I, wires=1)
        qml.RY(I, wires=1)
        if I < 5:
            I = I + 1
            A(partial(func, A=A, I=I))()

    @qjit(verbose=True, keep_intermediate=True)
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def C_workflow():
        qml.RX(pnp.pi / 2, wires=0)
        C_adjoint(partial(func, A=C_adjoint, I=0))()
        qml.RZ(pnp.pi / 2, wires=0)
        return qml.state()

    @qml.qnode(qml.device("default.qubit", wires=2))
    def PL_workflow():
        qml.RX(pnp.pi / 2, wires=0)
        PL_adjoint(partial(func, A=PL_adjoint, I=0))()
        qml.RZ(pnp.pi / 2, wires=0)
        return qml.state()

    actual = C_workflow()
    desired = PL_workflow()
    assert_allclose(actual, desired)


def test_adjoint_qubitunitary():
    def func():
        qml.QubitUnitary(
            jnp.array(
                [
                    [0.99500417 - 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 + 0.09983342j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.99500417 - 0.09983342j],
                ]
            ),
            wires=[0, 1],
        )

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def C_workflow():
        C_adjoint(func)()
        return qml.state()

    @qml.qnode(qml.device("default.qubit", wires=2))
    def PL_workflow():
        PL_adjoint(func)()
        return qml.state()

    actual = C_workflow()
    desired = PL_workflow()
    assert_allclose(actual, desired)


def test_adjoint_nomeasurements():
    def func():
        qml.RX(pnp.pi / 2, wires=0)
        qml.sample()

    with pytest.raises(ValueError, match="no measurements"):

        @qjit()
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def C_workflow():
            C_adjoint(func)()
            return qml.state()

        C_workflow()
