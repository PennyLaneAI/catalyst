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

"""Integration tests for quantum subroutine"""

import jax
import numpy as np
import pennylane as qml

from catalyst.jax_primitives import subroutine


def test_classical_subroutine():
    """Dummy test"""

    @subroutine
    def identity(x):
        return x

    @qml.qjit
    def subroutine_test():
        return identity(1)

    assert subroutine_test() == 1


def test_quantum_subroutine():
    """Test quantum subroutine"""

    @subroutine
    def Hadamard0(wire):
        qml.Hadamard(wire)

    qml.capture.enable()

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def subroutine_test(c: int):
        Hadamard0(c)
        return qml.state()

    assert np.allclose(subroutine_test(0), jax.numpy.array([0.70710678 + 0.0j, 0.70710678 + 0.0j]))
    qml.capture.disable()


def test_quantum_subroutine_self_inverses():
    """Test quantum subroutine multiple calls"""

    @subroutine
    def Hadamard0(wire):
        qml.Hadamard(wire)

    qml.capture.enable()

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def subroutine_test(c: int):
        Hadamard0(c)
        Hadamard0(c)
        return qml.state()

    assert np.allclose(
        subroutine_test(0), jax.numpy.array([complex(1.0, 0), complex(0.0, 0.0)], dtype=complex)
    )

    qml.capture.disable()


def test_quantum_subroutine_conditional():
    """Test quantum subroutine control flow"""

    @subroutine
    def Hadamard0(wire):
        def true_path():
            qml.Hadamard(wires=[0])

        def false_path(): ...

        qml.cond(wire != 0, true_path, false_path)()

    qml.capture.enable()

    @qml.qjit(autograph=False)
    @qml.qnode(qml.device("lightning.qubit", wires=1), autograph=False)
    def subroutine_test(c: int):
        Hadamard0(c)
        return qml.state()

    assert np.allclose(subroutine_test(0), jax.numpy.array([1.0, 0.0], dtype=complex))
    assert np.allclose(subroutine_test(1), jax.numpy.array([0.70710678 + 0.0j, 0.70710678 + 0.0j]))
    qml.capture.disable()
