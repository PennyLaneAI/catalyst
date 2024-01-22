# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for the async execution of QNodes features."""
import numpy as np
import pennylane as qml
import pytest
from jax import numpy as jnp

from catalyst import grad, qjit


@pytest.mark.skip()
def test_qnode_execution(backend):
    """The two first QNodes are executed in parrallel."""
    dev = qml.device(backend, wires=2)

    def multiple_qnodes(params):
        @qml.qnode(device=dev)
        def circuit1(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0))

        @qml.qnode(device=dev)
        def circuit2(params):
            qml.RY(params[0], wires=0)
            qml.RZ(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(wires=0))

        @qml.qnode(device=dev)
        def circuit3(params):
            qml.RZ(params[0], wires=0)
            qml.RX(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(wires=0))

        new_params = jnp.array([circuit1(params), circuit2(params)])
        return circuit3(new_params)

    params = jnp.array([1.0, 2.0])
    compiled = qjit(async_qnodes=True)(multiple_qnodes)
    observed = compiled(params)
    expected = qjit()(multiple_qnodes)(params)
    assert "async_execute_fn" in compiled.qir
    assert np.allclose(expected, observed)


# TODO: add the following diff_methods once issue #419 is fixed:
# ("parameter-shift", "auto"), ("adjoint", "auto")]
@pytest.mark.skip()
@pytest.mark.parametrize("diff_methods", [("finite-diff", "fd")])
@pytest.mark.parametrize("inp", [(1.0), (2.0), (3.0), (4.0)])
def test_gradient(inp, diff_methods, backend):
    """Parameter shift and finite diff generate multiple QNode that are run async."""

    def f(x):
        qml.RX(x * 2, wires=0)
        return qml.expval(qml.PauliY(0))

    @qjit(async_qnodes=True)
    def compiled(x: float):
        g = qml.qnode(qml.device(backend, wires=1), diff_method=diff_methods[0])(f)
        h = grad(g, method=diff_methods[1])
        return h(x)

    def interpreted(x):
        device = qml.device("default.qubit", wires=1)
        g = qml.QNode(f, device, diff_method="backprop")
        h = qml.grad(g, argnum=0)
        return h(x)

    assert "async_execute_fn" in compiled.qir
    assert np.allclose(compiled(inp), interpreted(inp))


if __name__ == "__main__":
    pytest.main(["-x", __file__])
