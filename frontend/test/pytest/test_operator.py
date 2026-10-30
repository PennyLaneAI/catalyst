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

This file tests the execution of gates that are lowered through the operator2 primitive.
As execution unit tests for these primitives' lowering, we do not include any decomposition rules.
"""

# pylint: disable = useless-parent-delegation, missing-function-docstring, missing-class-docstring

import pennylane as qp
import pytest
from jax import numpy as jnp


class TestOperator2Execution:

    def test_custom_op_supported(self):
        """Test that Operator2 versions of core ops are supported and can be executed."""

        @qp.qjit(collect_decomp_rules=False, capture=True)
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def c(x):
            qp.PauliX(0)
            qp.RX(x, 1)
            qp.CRX(2 * x, (0, 2))
            return qp.expval(qp.Z(0)), qp.expval(qp.Z(1)), qp.expval(qp.Z(2))

        res1, res2, res3 = c(0.5)

        assert qp.math.allclose(res1, -1)
        assert qp.math.allclose(res2, jnp.cos(0.5))
        assert qp.math.allclose(res3, jnp.cos(1.0))

    def test_MultiRZ(self):
        """Test that MultiRZ can be executed."""

        @qp.qjit(collect_decomp_rules=False, capture=True)
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def c(x):
            qp.Hadamard(0)
            qp.Hadamard(1)
            # skip on 2 for comparison
            qp.MultiRZ(x, (0, 1, 2))
            return qp.expval(qp.X(0)), qp.expval(qp.X(1)), qp.expval(qp.X(2))

        r1, r2, r3 = c(0.5)
        assert qp.math.allclose(r1, jnp.cos(0.5))
        assert qp.math.allclose(r2, jnp.cos(0.5))
        assert qp.math.allclose(r3, 0)

    def test_paulirot(self):
        """Test that PauliRot can be executed."""

        @qp.qjit(collect_decomp_rules=False, capture=True)
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def c(x):
            qp.Hadamard(2)
            qp.PauliRot(x, "XYZ", (0, 1, 2))
            return qp.expval(qp.Z(0)), qp.expval(qp.Z(1)), qp.expval(qp.X(2))

        r1, r2, r3 = c(1.2)
        assert qp.math.allclose(r1, jnp.cos(1.2))
        assert qp.math.allclose(r2, jnp.cos(1.2))
        assert qp.math.allclose(r3, jnp.cos(1.2))

    def test_globalphase(self):
        """Test that global phase can be executed."""

        @qp.qjit(collect_decomp_rules=False, capture=True)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def c(x):
            qp.GlobalPhase(x)
            return qp.state()

        state = c(0.5)
        assert qp.math.allclose(state, jnp.exp(0.5 * -1j) * jnp.array([1, 0]))

    def test_QubitUnitary(self):
        """Test that QubitUnitary can be executed."""

        @qp.qjit(collect_decomp_rules=False, capture=True)
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def c():
            qp.QubitUnitary(jnp.array([[0, 1], [1, 0]]), 0)
            qp.QubitUnitary(qp.CNOT.compute_matrix(), (0, 1))
            return qp.expval(qp.Z(0)), qp.expval(qp.Z(0))

        r1, r2 = c()
        assert qp.math.allclose(r1, -1)
        assert qp.math.allclose(r2, -1)

    @pytest.mark.parametrize(
        "dim, expected",
        [
            (
                2,
                jnp.array(
                    [jnp.exp(0.5j) / 2, jnp.exp(0.5j) / 2, jnp.exp(-0.5j) / 2, jnp.exp(-0.5j) / 2]
                ),
            ),
            (
                3,
                jnp.array(
                    [jnp.exp(0.5j) / 2, jnp.exp(0.5j) / 2, jnp.exp(0.5j) / 2, jnp.exp(-0.5j) / 2]
                ),
            ),
        ],
    )
    def test_PCPhase(self, dim, expected):
        """Test that PCPhase can be executed."""

        @qp.qjit(collect_decomp_rules=False, capture=True)
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def c(x):
            qp.Hadamard(0)
            qp.Hadamard(1)
            qp.PCPhase(x, dim, (0, 1))
            return qp.state()

        state = c(0.5)
        assert qp.math.allclose(state, expected)
