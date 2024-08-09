# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test skipping iniitial state prep"""

import jax.numpy as jnp
import pennylane as qml
import pytest

from catalyst import DifferentiableCompileError as DiffErr
from catalyst import grad


class TestExamplesFromWebsite:
    """Test the easiest examples from the website"""

    def test_state_prep(self, backend):
        """Test example from
        https://docs.pennylane.ai/en/stable/code/api/pennylane.StatePrep.html
        as of July 31st 2024.

        Modified to use jax.numpy and a non trivial StatePrep
        """

        @qml.qnode(qml.device(backend, wires=2))
        def example_circuit():
            qml.StatePrep(jnp.array([0, 1, 0, 0]), wires=range(2))
            return qml.state()

        expected = example_circuit()
        observed = qml.qjit(example_circuit)()
        assert jnp.allclose(expected, observed)

    def test_basis_state(self, backend):
        """Test example from
        https://docs.pennylane.ai/en/stable/code/api/pennylane.BasisState.html
        as of July 31st 2024.

        Modified to use jax.numpy
        """

        @qml.qnode(qml.device(backend, wires=2))
        def example_circuit():
            qml.BasisState(jnp.array([1, 1]), wires=range(2))
            return qml.state()

        expected = example_circuit()
        observed = qml.qjit(example_circuit)()
        assert jnp.allclose(expected, observed)

    @pytest.mark.parametrize("wires", [(0), (1), (2)])
    def test_array_less_than_size_state_prep(self, wires, backend):
        """Test what happens when the array is less than the size required.
        This is the same error as reported by pennylane
        """

        @qml.qnode(qml.device(backend, wires=3))
        def example_circuit():
            qml.StatePrep(jnp.array([0, 1]), wires=wires)
            return qml.state()

        expected = example_circuit()
        observed = qml.qjit(example_circuit)()
        assert jnp.allclose(expected, observed)

    def test_wires_with_less_than_all_basis_state(self, backend):
        """Test what happens when not all wires are included.

        This is not the same behaviour as PennyLane, but for expediency,
        let's submit this and we can fix it later.
        """

        @qml.qnode(qml.device(backend, wires=3))
        def example_circuit():
            qml.BasisState(jnp.array([0, 1]), wires=range(2))
            return qml.state()

        expected = example_circuit()
        observed = qml.qjit(example_circuit)()
        assert jnp.allclose(expected, observed)


class TestDynamicWires:
    """Test dynamic wires"""

    @pytest.mark.parametrize("inp", [(0), (1)])
    def test_state_prep(self, inp):
        """Test example from
        https://docs.pennylane.ai/en/stable/code/api/pennylane.StatePrep.html
        as of July 31st 2024.

        Modified to use jax.numpy and a non trivial StatePrep
        Modified to use dynamic wires.

        Limit this test to lightning.qubit since kokkos is old device
        and this optimization does not yet work for it.
        """

        @qml.qjit
        @qml.qnode(qml.device("lightning.qubit", wires=3))
        def example_circuit(a: int):
            qml.StatePrep(jnp.array([0, 1, 0, 0]), wires=[a, a + 1])
            return qml.state()

        expected = example_circuit(inp)
        observed = qml.qjit(example_circuit)(inp)
        assert jnp.allclose(expected, observed)

    @pytest.mark.parametrize("inp", [(0), (1)])
    def test_basis_state(self, inp, backend):
        """Test example from
        https://docs.pennylane.ai/en/stable/code/api/pennylane.BasisState.html
        as of July 31st 2024.

        Modified to use jax.numpy
        Modified to use dynamic wires.
        """

        @qml.qnode(qml.device(backend, wires=3))
        def example_circuit(a: int):
            qml.BasisState(jnp.array([1, 1]), wires=[a, a + 1])
            return qml.state()

        expected = example_circuit(inp)
        observed = qml.qjit(example_circuit)(inp)
        assert jnp.allclose(expected, observed)


class TestPossibleErrors:
    """What happens when there is bad user input?"""

    def test_array_less_than_size_basis_state(self):
        """Test what happens when the array is less than the size required.
        This is the same error as reported by pennylane
        """

        with pytest.raises(ValueError, match="must be of equal length"):

            @qml.qjit
            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def example_circuit():
                qml.BasisState(jnp.array([1]), wires=range(2))
                return qml.state()

            example_circuit()

    def test_domain_invalid_basis_state(self):
        """Test what happens when BasisState operand is not between {0, 1}.
        This is the same error message, but different error class.
        In PennyLane the error raised is a ValueError, but all errors
        in Catalyst that happen during runtime are RuntimeErrors.
        """
        msg = "BasisState parameter must consist of 0 or 1 integers"
        with pytest.raises(RuntimeError, match=msg):

            @qml.qjit
            @qml.qnode(qml.device("lightning.qubit", wires=2))
            def example_circuit():
                qml.BasisState(jnp.array([0, 2]), wires=range(2))
                return qml.state()

            example_circuit()


class TestGrad:
    """What happens if grad?"""

    def test_state_prep_grad(self, backend):
        """Test error happens with gradient"""

        with pytest.raises(DiffErr):

            @qml.qjit
            @grad
            @qml.qnode(qml.device(backend, wires=2))
            def example_circuit(a: float):
                qml.StatePrep(jnp.array([0, 1]), wires=[0])
                qml.RX(a, wires=[0])
                return qml.state()

            example_circuit(0.0)

    def test_basis_state_grad(self, backend):
        """Test error happens with gradient"""

        with pytest.raises(DiffErr):

            @qml.qjit
            @grad
            @qml.qnode(qml.device(backend, wires=2))
            def example_circuit(a: float):
                qml.BasisState(jnp.array([0, 1]), wires=range(2))
                qml.RX(a, wires=[0])
                return qml.state()

            example_circuit(0.0)


class TestControlled:
    """What happens if ctrl?"""

    def test_state_prep_ctrl(self):
        """Test state prep with ctrl"""

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def example_circuit():
            """Example provided by Tom in Slack"""
            qml.ctrl(qml.StatePrep, 0)([0, 1], 1)
            return qml.state()

        expected = example_circuit()
        observed = qml.qjit(example_circuit)()
        assert jnp.allclose(expected, observed)

    def test_basis_state_ctrl(self):
        """Test basis state with ctrl"""

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def example_circuit():
            """Changed from above to use qml.BasisState"""
            qml.ctrl(qml.BasisState, 0)([1], 1)
            return qml.state()

        expected = example_circuit()
        observed = qml.qjit(example_circuit)()
        assert jnp.allclose(expected, observed)


class TestAdjoint:
    """What happens if adjoint?"""

    def test_state_prep_ctrl(self, backend):
        """Test state prep with adjoint"""

        @qml.qnode(qml.device(backend, wires=2))
        def example_circuit():
            """Example provided by Tom in Slack"""
            qml.adjoint(qml.StatePrep, 0)([0, 1], 1)
            return qml.state()

        expected = example_circuit()
        observed = qml.qjit(example_circuit)()
        assert jnp.allclose(expected, observed)

    def test_basis_state_ctrl(self, backend):
        """Test basis state with adjoint"""

        @qml.qnode(qml.device(backend, wires=2))
        def example_circuit():
            """Changed from above to use qml.BasisState"""
            qml.adjoint(qml.BasisState, 0)([1], 1)
            return qml.state()

        expected = example_circuit()
        observed = qml.qjit(example_circuit)()
        assert jnp.allclose(expected, observed)
