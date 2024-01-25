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
import pennylane as qml

from catalyst import qjit, measure
from catalyst.compilation_pipelines import QJIT_CUDA, QJIT
from catalyst.compiler import CompileOptions
from catalyst.utils.jax_extras import remove_host_context
from catalyst.cuda_quantum_integration import catalyst_to_cuda
import pytest
import cudaq
from numpy.testing import assert_allclose

import jax

from jax import numpy as jnp


def test_argument():
    """Test that we can pass cuda-quantum as a compiler to @qjit decorator."""

    @qjit(compiler="cuda-quantum")
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def foo():
        return qml.state()


def test_qjit_cuda_generate_jaxpr():
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def foo():
        return qml.state()

    opts = CompileOptions()
    expected_jaxpr = QJIT(foo, opts).jaxpr
    observed_jaxpr = QJIT_CUDA(foo, opts).get_jaxpr()
    assert str(expected_jaxpr) == str(observed_jaxpr)


def test_qjit_cuda_remove_host_context():
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def foo():
        return qml.state()

    opts = CompileOptions()
    expected_jaxpr = QJIT(foo, opts).jaxpr
    observed_jaxpr = QJIT_CUDA(foo, opts).get_jaxpr()
    jaxpr = remove_host_context(observed_jaxpr)
    assert jaxpr


def test_qjit_catalyst_to_cuda_jaxpr():
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def foo():
        return qml.state()

    cuda_jaxpr = jax.make_jaxpr(catalyst_to_cuda(foo))


def test_qjit_catalyst_to_cuda_jaxpr_actually_call_ry():
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def foo():
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

    cuda_jaxpr = jax.make_jaxpr(catalyst_to_cuda(foo))()
    assert "ry" in str(cuda_jaxpr)
    assert "rx" in str(cuda_jaxpr)
    expected = jnp.array(cuda_equivalent())
    observed = jnp.array(jax.core.eval_jaxpr(cuda_jaxpr.jaxpr, cuda_jaxpr.consts)[0])
    assert_allclose(expected, observed)


def test_sample_with_shots():
    @qml.qnode(qml.device("lightning.qubit", wires=1, shots=30))
    def foo():
        qml.RX(jnp.pi, wires=[0])
        return qml.sample()

    cuda_jaxpr = jax.make_jaxpr(catalyst_to_cuda(foo))()
    assert "sample" in str(cuda_jaxpr)
    assert "shots_count=30" in str(cuda_jaxpr)
    # Due to non-determinism, let's just check the length of the samples

    expected_shape = foo()
    observed = jax.core.eval_jaxpr(cuda_jaxpr.jaxpr, cuda_jaxpr.consts)[0]
    assert len(observed) == 30


def test_counts_with_shots():
    @qml.qnode(qml.device("lightning.qubit", wires=2, shots=30))
    def foo():
        qml.RX(jnp.pi, wires=[0])
        return qml.counts()

    cuda_jaxpr = jax.make_jaxpr(catalyst_to_cuda(foo))()
    assert "counts" in str(cuda_jaxpr)
    assert "shots_count=30" in str(cuda_jaxpr)
    expected = qjit(foo)()
    observed = jax.core.eval_jaxpr(cuda_jaxpr.jaxpr, cuda_jaxpr.consts)
    assert_allclose(expected, observed)


def test_measurement_side_effect():
    """Test the measurement code is added."""

    @qml.qnode(qml.device("lightning.qubit", wires=1, shots=30))
    def baz():
        qml.RX(jnp.pi / 4, wires=[0])
        measure(0)
        return qml.state()

    cuda_jaxpr = jax.make_jaxpr(catalyst_to_cuda(baz))()
    assert "mz" in str(cuda_jaxpr)
    observed = jax.core.eval_jaxpr(cuda_jaxpr.jaxpr, cuda_jaxpr.consts)[0]
    assert abs(observed[0] ** 2) == 1 or abs(observed[1] ** 2) == 1


def test_measurement_side_return():
    """Test the measurement code is added."""

    with pytest.raises(NotImplementedError, match="cannot return measurements directly"):

        @qml.qnode(qml.device("lightning.qubit", wires=1, shots=30))
        def baz():
            qml.RX(jnp.pi / 4, wires=[0])
            return measure(0)

        cuda_jaxpr = jax.make_jaxpr(catalyst_to_cuda(baz))()
