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
"""CUDA Integration testing."""

import jax
import pennylane as qp
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

import catalyst
from catalyst import measure, qjit
from catalyst.utils.exceptions import CompileError

# This import is here on purpose. We shouldn't ever import CUDA
# when we are running kokkos. Importing CUDA before running any kokkos
# kernel polutes the environment and will create a segfault.
# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-public-methods


@pytest.mark.cuda
class TestCudaQ:
    """CUDA Quantum integration tests. Skip if kokkos."""

    def test_valid_device(self):
        """Test that we cannot pass lightning qubit as a compiler to @qjit decorator."""

        from catalyst.third_party.cuda import cudaqjit as cjit

        @cjit()
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit_foo():
            return qp.state()

        with pytest.raises(ValueError, match="Unavailable target"):
            circuit_foo()

    def test_measurement_return(self):
        """Test the measurement code is added."""

        from catalyst.third_party.cuda import cudaqjit as cjit

        with pytest.raises(NotImplementedError, match="cannot return measurements directly"):

            @cjit
            @qp.set_shots(30)
            @qp.qnode(qp.device("softwareq.qpp", wires=1))
            def circuit():
                qp.RX(jnp.pi / 4, wires=[0])
                return measure(0)

            circuit()

    def test_measurement_side_effect(self):
        """Test the measurement code is added."""

        from catalyst.third_party.cuda import cudaqjit as cjit

        @cjit
        @qp.qnode(qp.device("softwareq.qpp", wires=1))
        def circuit():
            qp.RX(jnp.pi / 4, wires=[0])
            measure(0)
            return qp.state()

        circuit()

    def test_pytrees(self):
        """Test that we can return a dictionary."""

        @qp.qnode(qp.device("softwareq.qpp", wires=1))
        def circuit_a(a):
            qp.RX(a, wires=[0])
            return {"a": qp.state()}

        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit_b(a):
            qp.RX(a, wires=[0])
            return {"a": qp.state()}

        cuda_compiled = catalyst.third_party.cuda.cudaqjit()(circuit_a)
        observed = cuda_compiled(3.14)
        catalyst_compiled = qjit(circuit_b)
        expected = catalyst_compiled(3.14)
        assert_allclose(expected["a"], observed["a"])

    def test_cuda_device(self):
        """Test SoftwareQQPP."""

        @qp.qnode(qp.device("softwareq.qpp", wires=1))
        def circuit(a):
            qp.RX(a, wires=[0])
            return qp.state()

        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit_lightning(a):
            qp.RX(a, wires=[0])
            return qp.state()

        cuda_compiled = catalyst.third_party.cuda.cudaqjit()(circuit)
        catalyst_compiled = qjit(circuit_lightning)
        expected = catalyst_compiled(3.14)
        observed = cuda_compiled(3.14)
        assert_allclose(expected, observed)

    def test_samples(self):
        """Test SoftwareQQPP."""

        @qp.set_shots(100)
        @qp.qnode(qp.device("softwareq.qpp", wires=1))
        def circuit(a):
            qp.RX(a, wires=[0])
            return qp.sample()

        @qp.set_shots(100)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit_lightning(a):
            qp.RX(a, wires=[0])
            return qp.sample()

        cuda_compiled = catalyst.third_party.cuda.cudaqjit()(circuit)
        catalyst_compiled = qjit(circuit_lightning)
        expected = catalyst_compiled(3.14)
        observed = cuda_compiled(3.14)
        assert_allclose(expected, observed)

    def test_counts(self):
        """Test SoftwareQQPP."""

        @qp.set_shots(100)
        @qp.qnode(qp.device("softwareq.qpp", wires=1))
        def circuit(a):
            qp.RX(a, wires=[0])
            return qp.counts()

        @qp.set_shots(100)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit_lightning(a):
            qp.RX(a, wires=[0])
            return qp.counts()

        cuda_compiled = catalyst.third_party.cuda.cudaqjit()(circuit)
        catalyst_compiled = qjit(circuit_lightning)
        expected = catalyst_compiled(3.14)
        observed = cuda_compiled(3.14)
        assert_allclose(expected, observed)

    def test_qjit_cuda_device(self):
        """Test SoftwareQQPP."""

        @qp.qnode(qp.device("softwareq.qpp", wires=1))
        def circuit(a):
            qp.RX(a, wires=[0])
            return qp.state()

        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit_lightning(a):
            qp.RX(a, wires=[0])
            return qp.state()

        cuda_compiled = catalyst.third_party.cuda.cudaqjit(fn=circuit)
        catalyst_compiled = qjit(circuit_lightning)
        expected = catalyst_compiled(3.14)
        observed = cuda_compiled(3.14)
        assert_allclose(expected, observed)

    def test_abstract_variable(self):
        """Test abstract variable."""

        @qp.qnode(qp.device("softwareq.qpp", wires=1))
        def circuit(a: float):
            qp.RX(a, wires=[0])
            return qp.state()

        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit_lightning(a):
            qp.RX(a, wires=[0])
            return qp.state()

        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
        catalyst_compiled = qjit(circuit_lightning)
        expected = catalyst_compiled(3.14)
        observed = cuda_compiled(3.14)
        assert_allclose(expected, observed)

    def test_arithmetic(self):
        """Test arithmetic."""

        @qp.qnode(qp.device("softwareq.qpp", wires=1))
        def circuit(a):
            qp.RX(a / 2, wires=[0])
            return qp.state()

        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit_lightning(a):
            qp.RX(a / 2, wires=[0])
            return qp.state()

        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
        catalyst_compiled = qjit(circuit_lightning)
        expected = catalyst_compiled(3.14)
        observed = cuda_compiled(3.14)
        assert_allclose(expected, observed)

    def test_multiple_values(self):
        """Test multiple_values."""

        @qp.qnode(qp.device("softwareq.qpp", wires=1))
        def circuit(params):
            x, y = jax.numpy.array_split(params, 2)
            qp.RX(x[0], wires=[0])
            qp.RX(y[0], wires=[0])
            return qp.state()

        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit_lightning(params):
            x, y = jax.numpy.array_split(params, 2)
            qp.RX(x[0], wires=[0])
            qp.RX(y[0], wires=[0])
            return qp.state()

        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
        catalyst_compiled = qjit(circuit_lightning)
        expected = catalyst_compiled(jax.numpy.array([3.14, 0.0]))
        observed = cuda_compiled(jax.numpy.array([3.14, 0.0]))
        assert_allclose(expected, observed)

    @pytest.mark.skipif("0.35" not in qp.version(), reason="Unsupported in pennylane version")
    def test_cuda_device_entry_point(self):
        """Test the entry point for SoftwareQQPP"""

        @qp.qnode(qp.device("softwareq.qpp", wires=1))
        def circuit(a):
            qp.RX(a, wires=[0])
            return {"a": qp.state()}

        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit_lightning(a):
            qp.RX(a, wires=[0])
            return {"a": qp.state()}

        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
        catalyst_compiled = qjit(circuit_lightning)
        expected = catalyst_compiled(3.14)
        observed = cuda_compiled(3.14)
        assert_allclose(expected["a"], observed["a"])

    @pytest.mark.skipif("0.35" not in qp.version(), reason="Unsupported in pennylane version")
    def test_cuda_device_entry_point_compiler(self):
        """Test the entry point for cudaq.qjit"""

        @qjit(compiler="cuda_quantum")
        @qp.qnode(qp.device("softwareq.qpp", wires=1))
        def circuit(a):
            qp.RX(a, wires=[0])
            return {"a": qp.state()}

        circuit(3.14)

    def test_expval(self):
        """Test multiple_values."""

        @qp.qnode(qp.device("softwareq.qpp", wires=1))
        def circuit():
            qp.RX(jnp.pi / 2, wires=[0])
            return qp.expval(qp.PauliZ(0))

        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit_catalyst():
            qp.RX(jnp.pi / 2, wires=[0])
            return qp.expval(qp.PauliZ(0))

        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
        observed = cuda_compiled()
        catalyst_compiled = qjit(circuit_catalyst)
        expected = catalyst_compiled()
        assert_allclose(expected, observed)

    def test_expval_2(self):
        """Test multiple_values."""

        @qp.qnode(qp.device("softwareq.qpp", wires=2))
        def circuit():
            qp.RY(jnp.pi / 4, wires=[1])
            return qp.expval(qp.PauliZ(1) + qp.PauliX(1))

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit_catalyst():
            qp.RY(jnp.pi / 4, wires=[1])
            return qp.expval(qp.PauliZ(1) + qp.PauliX(1))

        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
        observed = cuda_compiled()
        catalyst_compiled = qjit(circuit_catalyst)
        expected = catalyst_compiled()
        assert_allclose(expected, observed)

    def test_adjoint(self):
        """Test adjoint."""

        @qp.qnode(qp.device("softwareq.qpp", wires=2))
        def circuit():
            def f(theta):
                qp.RX(theta / 23, wires=[0])
                qp.RX(theta / 17, wires=[1])
                qp.Hadamard(wires=[0])
                qp.Hadamard(wires=[1])
                qp.PauliX(wires=0)
                qp.PauliY(wires=1)

            qp.adjoint(f)(jnp.pi)
            return qp.state()

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit_catalyst():
            def f(theta):
                qp.RX(theta / 23, wires=[0])
                qp.RX(theta / 17, wires=[1])
                qp.Hadamard(wires=[0])
                qp.Hadamard(wires=[1])
                qp.PauliX(wires=0)
                qp.PauliY(wires=1)

            qp.adjoint(f)(jnp.pi)
            return qp.state()

        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
        observed = cuda_compiled()
        catalyst_compiled = qjit(circuit_catalyst)
        expected = catalyst_compiled()
        assert_allclose(expected, observed)

    def test_control_ry(self):
        """Test control ry."""

        @qp.qnode(qp.device("softwareq.qpp", wires=2))
        def circuit():
            qp.Hadamard(wires=[0])
            qp.CRY(jnp.pi / 2, wires=[0, 1])
            return qp.state()

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit_catalyst():
            qp.Hadamard(wires=[0])
            qp.CRY(jnp.pi / 2, wires=[0, 1])
            return qp.state()

        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
        observed = cuda_compiled()
        catalyst_compiled = qjit(circuit_catalyst)
        expected = catalyst_compiled()
        assert_allclose(expected, observed)

    def test_swap(self):
        """Test swap."""

        @qp.qnode(qp.device("softwareq.qpp", wires=2))
        def circuit():
            qp.RX(jnp.pi / 3, wires=[0])
            qp.SWAP(wires=[0, 1])
            return qp.state()

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit_catalyst():
            qp.RX(jnp.pi / 3, wires=[0])
            qp.SWAP(wires=[0, 1])
            return qp.state()

        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
        observed = cuda_compiled()
        catalyst_compiled = qjit(circuit_catalyst)
        expected = catalyst_compiled()
        assert_allclose(expected, observed)

    def test_entanglement(self):
        """Test swap."""

        @qp.qnode(qp.device("softwareq.qpp", wires=2))
        def circuit():
            qp.Hadamard(wires=[0])
            qp.CNOT(wires=[0, 1])
            return qp.state()

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit_catalyst():
            qp.Hadamard(wires=[0])
            qp.CNOT(wires=[0, 1])
            return qp.state()

        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
        observed = cuda_compiled()
        catalyst_compiled = qjit(circuit_catalyst)
        expected = catalyst_compiled()
        assert_allclose(expected, observed)

    def test_cswap(self):
        """Test cswap."""

        @qp.qnode(qp.device("softwareq.qpp", wires=3))
        def circuit():
            qp.Hadamard(wires=[0])
            qp.RX(jnp.pi / 7, wires=[1])
            qp.CSWAP(wires=[0, 1, 2])
            return qp.state()

        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def circuit_catalyst():
            qp.Hadamard(wires=[0])
            qp.RX(jnp.pi / 7, wires=[1])
            qp.CSWAP(wires=[0, 1, 2])
            return qp.state()

        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
        observed = cuda_compiled()
        catalyst_compiled = qjit(circuit_catalyst)
        expected = catalyst_compiled()
        assert_allclose(expected, observed)

    def test_state_is_jax_array(self):
        """Test return type for state."""

        @qp.qnode(qp.device("softwareq.qpp", wires=3))
        def circuit():
            qp.Hadamard(wires=[0])
            qp.RX(jnp.pi / 7, wires=[1])
            qp.CSWAP(wires=[0, 1, 2])
            return qp.state()

        cuda_compiled = catalyst.third_party.cuda.cudaqjit(circuit)
        observed = cuda_compiled()
        assert isinstance(observed, jax.Array)

    def test_error_message_using_host_context(self):
        """Test error message"""

        @qp.qnode(qp.device("softwareq.qpp", wires=2))
        def circuit(x):
            qp.Hadamard(wires=[0])
            qp.CNOT(wires=[0, 1])
            qp.RX(x, wires=[0])
            return qp.state()

        def wrapper(y):
            x = y + 1
            return circuit(x)

        with pytest.raises(CompileError, match="Cannot translate tapes with context"):
            catalyst.third_party.cuda.cudaqjit(wrapper)(1.0)

    def test_samples_multiple_wires(self):
        """Samples with more than one wire."""

        from catalyst.third_party.cuda import cudaqjit as cjit

        @qjit
        @qp.set_shots(10)
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit1(a):
            qp.RX(a, wires=0)
            return qp.sample()

        expected = circuit1(3.14)

        @cjit
        @qp.set_shots(10)
        @qp.qnode(qp.device("softwareq.qpp", wires=2))
        def circuit2(a):
            qp.RX(a, wires=0)
            return qp.sample()

        observed = circuit2(3.14)
        assert_allclose(expected, observed)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
