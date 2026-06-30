# Copyright 2022-2026 Xanadu Quantum Technologies Inc.

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
Tests for the new Operator2 class.
"""

# pylint: disable = useless-parent-delegation, missing-function-docstring, missing-class-docstring
import pennylane as qp
import pytest
from jax import numpy as jnp

class DummyOp(qp.core.Operator2):

    def __init__(self, wires):
        super().__init__(wires=wires)

class PauliX(qp.core.Operator2):

    def __init__(self, wires):
        super().__init__(wires=wires)

class RX(qp.core.Operator2):

    dynamic_argnames = ("phi",)

    def __init__(self, phi, wires):
        super().__init__(phi, wires)

class CRX(qp.core.Operator2):

    dynamic_argnames = ("phi", )
    
    def __init__(self, phi, wires):
        super().__init__(phi, wires=wires)

class MultiRZ(qp.core.Operator2):

    dynamic_argnames = ("phi",)

    def __init__(self, phi, wires):
        super().__init__(phi, wires)


class Hadamard(qp.core.Operator2):

    def __init__(self, wires):
        super().__init__(wires=wires)


class PauliRot(qp.core.Operator2):

    dynamic_argnames = ("phi",)
    compilable_argnames = ("pauli_word", )

    def __init__(self, phi, pauli_word, wires):
        super().__init__(phi, pauli_word, wires)



class GlobalPhase(qp.core.Operator2):

    dynamic_argnames = ("phi", )
    wire_argnames = ()

    def __init__(self, phi):
        super().__init__(phi=phi)

class QubitUnitary(qp.core.Operator2):

    dynamic_argnames = ("matrix", )

    def __init__(self, matrix, wires):
        super().__init__(matrix, wires)

class PCPhase(qp.core.Operator2):

    dynamic_argnames = ("phi", "dim")

    def __init__(self, phi, dim, wires):
        super().__init__(phi, dim, wires)


class TestOperator2Execution:

    def test_custom_op_supported(self):
        """Test that Operator2 versions of core ops are supported and can be executed."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device('lightning.qubit', wires=3))
        def c(x):
            PauliX(0)
            RX(x, 1)
            CRX(2*x, (0,2))
            return qp.expval(qp.Z(0)), qp.expval(qp.Z(1)), qp.expval(qp.Z(2))

        res1, res2, res3 = c(0.5)

        assert qp.math.allclose(res1, -1)
        assert qp.math.allclose(res2, jnp.cos(0.5))
        assert qp.math.allclose(res3, jnp.cos(1.0))

    def test_MultiRZ(self):
        """Test that MultiRZ can be executed."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device('lightning.qubit', wires=3))
        def c(x):
            Hadamard(0)
            Hadamard(1)
            # skip on 2 for comparison
            MultiRZ(x, (0,1,2))
            return qp.expval(qp.X(0)), qp.expval(qp.X(1)), qp.expval(qp.X(2))

        r1, r2, r3 = c(0.5)
        assert qp.math.allclose(r1, jnp.cos(0.5))
        assert qp.math.allclose(r2, jnp.cos(0.5))
        assert qp.math.allclose(r3, 0)

    def test_paulirot(self):
        """Test that PauliRot can be executed."""
        @qp.qjit(capture=True)
        @qp.qnode(qp.device('lightning.qubit', wires=3))
        def c(x):
            Hadamard(2)
            PauliRot(x, "XYZ", (0,1,2))
            return qp.expval(qp.Z(0)), qp.expval(qp.Z(1)), qp.expval(qp.X(2))

        r1, r2, r3 = c(1.2)
        assert qp.math.allclose(r1, jnp.cos(1.2))
        assert qp.math.allclose(r2, jnp.cos(1.2))
        assert qp.math.allclose(r3, jnp.cos(1.2))

    def test_globalphase(self):
        """Test that global phase can be executed."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device('lightning.qubit', wires=1))
        def c(x):
            GlobalPhase(x)
            return qp.state()

        state = c(0.5)
        assert qp.math.allclose(state, jnp.exp(0.5 * -1j) * jnp.array([1, 0]))

    def test_QubitUnitary(self):
        """Test that QubitUnitary can be executed."""

        @qp.qjit(capture=True)
        @qp.qjitqnode(qp.device('lightning.qubit', wires=3))
        def c():
            QubitUnitary(jnp.array([[0,1],[1,0]]), 0)
            QubitUnitary(qp.CNOT.compute_matrix(), (0,1))
            return qp.expval(qp.Z(0)), qp.expval(qp.Z(0))

        r1, r2 = c()
        assert qp.math.allclose(r1, -1)
        assert qp.math.allclose(r2, -1)

    def test_PCPhase(self):
        """Test that PCPhase can be executed."""

        @qp.qjit(capture=True)
        @qp.qnode(qp.device('lightning.qubit', wires=3))
        def c(x, dim):
            Hadamard(0)
            Hadamard(1)
            PCPhase(x, dim, (0,1))
            return qp.state()

        state2 = c(0.5, 2)
        plus = jnp.exp(0.5j)/jnp.sqrt(2)
        minus = jnp.exp(-0.5j)/jnp.sqrt(2)
        expected2 = jnp.array([plus, plus, minus, minus])
        assert qp.math.allclose(state2, expected2)

        state3 = c(0.5, 3)
        expected3 = jnp.array([plus, plus, plus, minus])
        assert qp.math.allclose(state3, expected3)


def test_hybrid_not_supported_yet():
    """Test that hybrid arguments are not yet supported."""

    class OperatorArgument(qp.core.Operator2):

        hybrid_argnames = ("op",)
        wire_argnames = ()

        def __init__(self, op):
            super().__init__(op)

    with pytest.raises(NotImplementedError):

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=3))
        def c():
            OperatorArgument(DummyOp(0))
            return qp.state()


def test_static_argnames():
    """Test that static arguments are not yet supported."""

    class StaticArgsOp(qp.core.Operator2):

        static_argnames = ("thing",)

        def __init__(self, thing, wires):
            super().__init__(thing, wires)

    with pytest.raises(NotImplementedError):

        @qp.qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=2))
        def c():
            StaticArgsOp("hello", 0)
            return qp.state()
