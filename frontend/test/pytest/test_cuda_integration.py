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
"""CUDA Integration testing."""

import jax
import pennylane as qml
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

from catalyst import measure, qjit
from catalyst.compilation_pipelines import QJIT
from catalyst.compiler import CompileOptions
from catalyst.utils.jax_extras import remove_host_context

# This import is here on purpose. We shouldn't ever import CUDA
# when we are running kokkos. Importing CUDA before running any kokkos
# kernel polutes the environment and will create a segfault.
# pylint: disable=import-outside-toplevel


@pytest.mark.cuda
class TestCuda:
    """CUDA integration tests. Skip if kokkos."""

    def test_argument(self):
        """Test that we can pass cuda_quantum as a compiler to @qjit decorator."""

        @qjit(compiler="cuda_quantum")
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit_foo():
            return qml.state()

    def test_qjit_cuda_generate_jaxpr(self):
        """Test the JAXPR generation."""

        from catalyst.cuda.catalyst_to_cuda_interpreter import QJIT_CUDA

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit_foo():
            return qml.state()

        opts = CompileOptions()
        expected_jaxpr = QJIT(circuit_foo, opts).jaxpr
        observed_jaxpr, _ = QJIT_CUDA(circuit_foo).get_jaxpr()
        assert str(expected_jaxpr) == str(observed_jaxpr)

    def test_qjit_cuda_remove_host_context(self):
        """Test removing the host context."""

        from catalyst.cuda.catalyst_to_cuda_interpreter import QJIT_CUDA

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit_foo():
            return qml.state()

        observed_jaxpr, _ = QJIT_CUDA(circuit_foo).get_jaxpr()
        jaxpr = remove_host_context(observed_jaxpr)
        assert jaxpr

    def test_qjit_catalyst_to_cuda_jaxpr(self):
        """Assert that catalyst_to_cuda returns something."""
        from catalyst.cuda.catalyst_to_cuda_interpreter import interpret

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit_foo():
            return qml.state()

        cuda_jaxpr = jax.make_jaxpr(interpret(circuit_foo))
        assert cuda_jaxpr

    def test_qjit_catalyst_to_cuda_jaxpr_actually_call_ry(self):
        """Assert that catalyst_to_cuda matches the expected results."""
        import cudaq

        from catalyst.cuda.catalyst_to_cuda_interpreter import interpret

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit_foo():
            qml.RY(jnp.pi, wires=[0])
            qml.RX(jnp.pi, wires=[0])
            return qml.state()

        def cuda_equivalent():
            kernel = cudaq.make_kernel()
            qubits = kernel.qalloc(1)
            qubit = qubits[0]
            kernel.ry(jnp.pi, qubit)
            kernel.rx(jnp.pi, qubit)
            return cudaq.get_state(kernel)

        cuda_jaxpr = jax.make_jaxpr(interpret(circuit_foo))()
        assert "ry" in str(cuda_jaxpr)
        assert "rx" in str(cuda_jaxpr)
        expected = jnp.array(cuda_equivalent())
        observed = jnp.array(jax.core.eval_jaxpr(cuda_jaxpr.jaxpr, cuda_jaxpr.consts)[0])
        assert_allclose(expected, observed)

    def test_sample_with_shots(self):
        """Assert that catalyst_to_cuda can handle shots."""
        from catalyst.cuda.catalyst_to_cuda_interpreter import interpret

        @qml.qnode(qml.device("lightning.qubit", wires=1, shots=30))
        def circuit_foo():
            qml.RX(jnp.pi, wires=[0])
            return qml.sample()

        cuda_jaxpr = jax.make_jaxpr(interpret(circuit_foo))()
        assert "sample" in str(cuda_jaxpr)
        assert "shots_count=30" in str(cuda_jaxpr)
        # Due to non-determinism, let's just check the length of the samples

        circuit_foo()
        observed = jax.core.eval_jaxpr(cuda_jaxpr.jaxpr, cuda_jaxpr.consts)[0]
        assert len(observed) == 30

    def test_counts_with_shots(self):
        """Assert that catalyst_to_cuda can handle counts with shots."""
        from catalyst.cuda.catalyst_to_cuda_interpreter import interpret

        @qml.qnode(qml.device("lightning.qubit", wires=2, shots=30))
        def circuit_foo():
            qml.RX(jnp.pi, wires=[0])
            return qml.counts()

        cuda_jaxpr = jax.make_jaxpr(interpret(circuit_foo))()
        assert "counts" in str(cuda_jaxpr)
        assert "shots_count=30" in str(cuda_jaxpr)
        expected = qjit(circuit_foo)()
        observed = jax.core.eval_jaxpr(cuda_jaxpr.jaxpr, cuda_jaxpr.consts)
        assert_allclose(expected, observed)

    def test_measurement_side_effect(self):
        """Test the measurement code is added."""

        from catalyst.cuda.catalyst_to_cuda_interpreter import interpret

        @qml.qnode(qml.device("lightning.qubit", wires=1, shots=30))
        def circuit():
            qml.RX(jnp.pi / 4, wires=[0])
            measure(0)
            return qml.state()

        cuda_jaxpr = jax.make_jaxpr(interpret(circuit))()
        assert "mz" in str(cuda_jaxpr)
        observed = jax.core.eval_jaxpr(cuda_jaxpr.jaxpr, cuda_jaxpr.consts)[0]
        assert abs(observed[0] ** 2) == 1 or abs(observed[1] ** 2) == 1

    def test_measurement_side_return(self):
        """Test the measurement code is added."""

        from catalyst.cuda.catalyst_to_cuda_interpreter import interpret

        with pytest.raises(NotImplementedError, match="cannot return measurements directly"):

            @qml.qnode(qml.device("lightning.qubit", wires=1, shots=30))
            def circuit():
                qml.RX(jnp.pi / 4, wires=[0])
                return measure(0)

            jax.make_jaxpr(interpret(circuit))()

    def test_pytrees(self):
        """Test that we can return a dictionary."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(a):
            qml.RX(a, wires=[0])
            return {"a": qml.state()}

        cuda_compiled = qjit(compiler="cuda_quantum")(circuit)
        observed = cuda_compiled(3.14)
        catalyst_compiled = qjit(circuit)
        expected = catalyst_compiled(3.14)
        assert_allclose(expected["a"], observed["a"])

    def test_cuda_device(self):
        """Test CudaQDevice."""
        from catalyst.cuda import CudaQDevice

        @qml.qnode(CudaQDevice(wires=1))
        def circuit(a):
            qml.RX(a, wires=[0])
            return qml.state()

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit_lightning(a):
            qml.RX(a, wires=[0])
            return qml.state()

        cuda_compiled = qjit(compiler="cuda_quantum")(circuit)
        catalyst_compiled = qjit(circuit_lightning)
        expected = catalyst_compiled(3.14)
        observed = cuda_compiled(3.14)
        assert_allclose(expected, observed)

    def test_qjit_cuda_device(self):
        """Test CudaQDevice."""
        from catalyst.cuda import CudaQDevice, qjit_cuda

        @qml.qnode(CudaQDevice(wires=1))
        def circuit(a):
            qml.RX(a, wires=[0])
            return qml.state()

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit_lightning(a):
            qml.RX(a, wires=[0])
            return qml.state()

        cuda_compiled = qjit_cuda(fn=circuit, compiler="cuda_quantum")
        catalyst_compiled = qjit(circuit_lightning)
        expected = catalyst_compiled(3.14)
        observed = cuda_compiled(3.14)
        assert_allclose(expected, observed)

    def test_abstract_variable(self):
        """Test abstract variable."""
        from catalyst.cuda import CudaQDevice

        @qml.qnode(CudaQDevice(wires=1))
        def circuit(a: float):
            qml.RX(a, wires=[0])
            return qml.state()

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit_lightning(a):
            qml.RX(a, wires=[0])
            return qml.state()

        cuda_compiled = qjit(compiler="cuda_quantum")(circuit)
        catalyst_compiled = qjit(circuit_lightning)
        expected = catalyst_compiled(3.14)
        observed = cuda_compiled(3.14)
        assert_allclose(expected, observed)

    def test_arithmetic(self):
        """Test arithmetic."""
        from catalyst.cuda import CudaQDevice

        @qml.qnode(CudaQDevice(wires=1))
        def circuit(a):
            qml.RX(a / 2, wires=[0])
            return qml.state()

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit_lightning(a):
            qml.RX(a / 2, wires=[0])
            return qml.state()

        cuda_compiled = qjit(compiler="cuda_quantum")(circuit)
        catalyst_compiled = qjit(circuit_lightning)
        expected = catalyst_compiled(3.14)
        observed = cuda_compiled(3.14)
        assert_allclose(expected, observed)

    def test_multiple_values(self):
        """Test multiple_values."""
        from catalyst.cuda import CudaQDevice

        @qml.qnode(CudaQDevice(wires=1))
        def circuit(params):
            x, y = jax.numpy.array_split(params, 2)
            qml.RX(x[0], wires=[0])
            qml.RX(y[0], wires=[0])
            return qml.state()

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit_lightning(params):
            x, y = jax.numpy.array_split(params, 2)
            qml.RX(x[0], wires=[0])
            qml.RX(y[0], wires=[0])
            return qml.state()

        cuda_compiled = qjit(compiler="cuda_quantum")(circuit)
        catalyst_compiled = qjit(circuit_lightning)
        expected = catalyst_compiled(jax.numpy.array([3.14, 0.0]))
        observed = cuda_compiled(jax.numpy.array([3.14, 0.0]))
        assert_allclose(expected, observed)

    @pytest.mark.skipif("0.35" not in qml.version(), reason="Unsupported in pennylane version")
    def test_cuda_device_entry_point(self):
        """Test the entry point for CudaQDevice"""

        @qml.qnode(qml.device("cudaq", wires=1))
        def circuit(a):
            qml.RX(a, wires=[0])
            return {"a": qml.state()}

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit_lightning(a):
            qml.RX(a, wires=[0])
            return {"a": qml.state()}

        cuda_compiled = qjit(compiler="cuda_quantum")(circuit)
        catalyst_compiled = qjit(circuit_lightning)
        expected = catalyst_compiled(3.14)
        observed = cuda_compiled(3.14)
        assert_allclose(expected["a"], observed["a"])

    @pytest.mark.skipif("0.35" not in qml.version(), reason="Unsupported in pennylane version")
    def test_cuda_device_entry_point_compiler(self):
        """Test the entry point for cudaq.qjit"""

        @qml.qjit(compiler="cuda_quantum")
        @qml.qnode(qml.device("cudaq", wires=1))
        def circuit(a):
            qml.RX(a, wires=[0])
            return {"a": qml.state()}

        circuit(3.14)

    def test_cuda_invalid_target(self):
        """Test invalid target."""
        from catalyst.cuda import qjit_cuda

        with pytest.raises(ValueError, match="Unsupported target foo."):
            qjit_cuda(fn=None, target="foo")

    def test_cuda_unimplemented_target(self):
        """Test unimplemented target."""
        from catalyst.cuda import qjit_cuda

        with pytest.raises(NotImplementedError, match="Unimplemented target nvidia."):
            qjit_cuda(fn=None, target="nvidia")
