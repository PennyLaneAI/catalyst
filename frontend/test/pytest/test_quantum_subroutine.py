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

from functools import partial

import jax
import numpy as np
import pennylane as qp
import pytest
from pennylane.capture import subroutine

pytestmark = pytest.mark.usefixtures("disable_capture")


class TestSubroutineHOP:
    """Integration tests for qp.capture.subroutine"""

    def test_classical_subroutine(self):
        """Dummy test"""

        @subroutine
        def identity(x):
            return x

        @qp.qjit(capture=True)
        def subroutine_test():
            return identity(1)

        assert subroutine_test() == 1

    def test_quantum_subroutine(self):
        """Test quantum subroutine"""

        @subroutine
        def Hadamard0(wire):
            qp.Hadamard(wire)

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def subroutine_test(c: int):
            Hadamard0(c)
            return qp.state()

        expected = jax.numpy.array([0.70710678 + 0.0j, 0.70710678 + 0.0j])
        assert np.allclose(subroutine_test(0), expected)

    def test_quantum_subroutine_self_inverses(self):
        """Test quantum subroutine multiple calls"""

        @subroutine
        def Hadamard0(wire):
            qp.Hadamard(wire)

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def subroutine_test(c: int):
            Hadamard0(c)
            Hadamard0(c)
            return qp.state()

        assert np.allclose(
            subroutine_test(0), jax.numpy.array([complex(1.0, 0), complex(0.0, 0.0)], dtype=complex)
        )

    def test_quantum_subroutine_error_message(self):
        """Test error message for quantum operations outside of qnode."""

        @subroutine
        def Hadamard0():
            qp.Hadamard(wires=[0])

        msg = "inside subroutine"
        with pytest.raises(NotImplementedError, match=msg):

            @qp.qjit(autograph=False, capture=True)
            def subroutine_test():
                Hadamard0()

    def test_quantum_subroutine_conditional(self):
        """Test quantum subroutine control flow"""

        @subroutine
        def Hadamard0(wire):
            def true_path():
                qp.Hadamard(wires=[0])

            def false_path(): ...

            qp.cond(wire != 0, true_path, false_path)()

        @qp.qjit(autograph=False, capture=True)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def subroutine_test(c: int):
            Hadamard0(c)
            return qp.state()

        assert np.allclose(subroutine_test(0), jax.numpy.array([1.0, 0.0], dtype=complex))
        assert np.allclose(
            subroutine_test(1), jax.numpy.array([0.70710678 + 0.0j, 0.70710678 + 0.0j])
        )


class TestSubroutineClass:
    """integration tests for qp.templates.Subroutine"""

    def test_basic_subroutine(self):
        """Test the execution of a simple subroutine."""

        @qp.templates.Subroutine
        def f(x, wires):
            qp.RX(x, wires)

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def c(x):
            f(x, 0)
            f(2 * x, 1)
            f(x + 1, 2)
            return [qp.expval(qp.Z(i)) for i in range(3)]

        r1, r2, r3 = c(0.5)
        assert qp.math.allclose(r1, np.cos(0.5))
        assert qp.math.allclose(r2, np.cos(1))
        assert qp.math.allclose(r3, np.cos(1.5))

    def test_subroutine_with_metadata(self):
        """Test that catalyst can handle a subroutine with metadata."""

        @partial(qp.templates.Subroutine, static_argnames="kind")
        def f(wires, kind):
            if kind == "X":
                qp.X(wires)
            else:
                qp.Z(wires)

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def c():
            f(0, "X")
            f(1, "X")
            f(2, "Z")
            return [qp.expval(qp.Z(i)) for i in range(3)]

        r1, r2, r3 = c()
        assert qp.math.allclose(r1, -1)
        assert qp.math.allclose(r2, -1)
        assert qp.math.allclose(r3, 1)

    def test_different_wire_name(self):
        """Test that the input for wires can be named something different."""

        @partial(qp.templates.Subroutine, wire_argnames="register")
        def f(register):
            @qp.for_loop(len(register))
            def l(i):
                qp.X(register[i])

            l()

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("lightning.qubit", wires=4))
        def c():
            f(register=(0, 1, 3))
            return [qp.expval(qp.Z(i)) for i in range(4)]

        r0, r1, r2, r3 = c()
        assert qp.math.allclose(r0, -1)
        assert qp.math.allclose(r1, -1)
        assert qp.math.allclose(r2, 1)
        assert qp.math.allclose(r3, -1)
