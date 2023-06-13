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

from typing import Iterable, Tuple, TypeVar, Union

import jax.numpy as jnp
import pennylane as qml
import pennylane.numpy as pnp
import pytest

from catalyst import qjit, adjoint as adjoint_C
from pennylane import adjoint as adjoint_PL
from numpy.testing import assert_allclose

# pylint: disable=missing-function-docstring


def test_adjoint_func_simple():
    def func():
        qml.PauliX(wires=0)
        qml.PauliY(wires=0)
        qml.PauliZ(wires=1)

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def workflow_C():
        qml.PauliX(wires=0)
        adjoint_C(func)()
        qml.PauliY(wires=0)
        return qml.state()

    @qml.qnode(qml.device("default.qubit", wires=2))
    def workflow_PL():
        qml.PauliX(wires=0)
        adjoint_PL(func)()
        qml.PauliY(wires=0)
        return qml.state()

    actual = workflow_C()
    desired = workflow_PL()
    assert_allclose(actual, desired)


def test_adjoint_singleop():
    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def workflow_C():
        adjoint_C(qml.PauliZ)(wires=0)
        return qml.state()

    @qml.qnode(qml.device("default.qubit", wires=2))
    def workflow_PL():
        adjoint_PL(qml.PauliZ)(wires=0)
        return qml.state()

    actual = workflow_C()
    desired = workflow_PL()
    assert_allclose(actual, desired)


@pytest.mark.parametrize("w, p", [(0, 0.5), (0, -100.0), (1, 123.22)])
def test_adjoint_func_paramethrised(w, p):

    def func(w, theta1, theta2, theta3):
        qml.RX(theta1*pnp.pi/2, wires=w)
        qml.RY(theta2/2, wires=w)
        qml.RZ(theta3, wires=1)

    @qjit()
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def workflow_C(w, theta):
        adjoint_C(func)(w, theta, theta, theta)
        return qml.state()

    @qml.qnode(qml.device("default.qubit", wires=2))
    def workflow_PL(w, theta):
        adjoint_PL(func)(w, theta, theta, theta)
        return qml.state()

    actual = workflow_C(w, p)
    desired = workflow_PL(w, p)
    assert_allclose(actual, desired)

