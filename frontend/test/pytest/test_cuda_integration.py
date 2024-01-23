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

from catalyst import qjit
from catalyst.compilation_pipelines import QJIT_CUDA, QJIT
from catalyst.compiler import CompileOptions
from catalyst.utils.jax_extras import remove_host_context
from catalyst.cuda_quantum_integration import catalyst_to_cuda
import pytest

import jax

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

def test_qjit_catalyst_to_cuda_jaxpr_actually_call():
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def foo():
        return qml.state()

    cuda_jaxpr = jax.make_jaxpr(catalyst_to_cuda(foo))
    with pytest.raises(NotImplementedError, match="TODO"):
        cuda_jaxpr()

