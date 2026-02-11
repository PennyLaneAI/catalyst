# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pytests for integration with the qml.templates.Subroutine class."""
from functools import partial

import numpy as np
import pennylane as qml


def test_basic_subroutine():
    """Test the execution of a simple subroutine."""

    @qml.templates.Subroutine
    def f(x, wires):
        qml.RX(x, wires)

    @qml.qjit(capture=True)
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def c(x):
        f(x, 0)
        f(2 * x, 1)
        f(x + 1, 2)
        return [qml.expval(qml.Z(i)) for i in range(3)]

    r1, r2, r3 = c(0.5)
    assert qml.math.allclose(r1, np.cos(0.5))
    assert qml.math.allclose(r2, np.cos(1))
    assert qml.math.allclose(r3, np.cos(1.5))


def test_subroutine_with_metadata():
    """Test that catalyst can handle a subroutine with metadata."""

    @partial(qml.templates.Subroutine, static_argnames="kind")
    def f(wires, kind):
        if kind == "X":
            qml.X(wires)
        else:
            qml.Z(wires)

    @qml.qjit(capture=True)
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def c():
        f(0, "X")
        f(1, "X")
        f(2, "Z")
        return [qml.expval(qml.Z(i)) for i in range(3)]

    r1, r2, r3 = c()
    assert qml.math.allclose(r1, -1)
    assert qml.math.allclose(r2, -1)
    assert qml.math.allclose(r3, 1)


def test_different_wire_name():
    """Test that the input for wires can be named something different."""

    @partial(qml.templates.Subroutine, wire_argnames="register")
    def f(register):
        @qml.for_loop(register.shape[0])
        def l(i):
            qml.X(i)

        l()

    @qml.qjit(capture=True)
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def c():
        f(register=(0, 1, 3))
        return [qml.expval(qml.Z(i)) for i in range(4)]

    r0, r1, r2, r3 = c()
    assert qml.math.allclose(r0, -1)
    assert qml.math.allclose(r1, -1)
    assert qml.math.allclose(r2, 1)
    assert qml.math.allclose(r3, -1)
