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

"""Test cases relating to quantum functions represented via :class:`qml.QNode.`"""

import jax.numpy as jnp
import pennylane as qml
import pytest

from catalyst import CompileError, measure, qjit


@pytest.mark.parametrize("_in,_out", [(0, False), (1, True)])
def test_variable_capture(_in, _out):
    """Test closures (outer-scope variable capture) for quantum functions."""

    @qjit()
    def workflow(n: int):
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f(x: float):
            qml.RX(n * x, wires=n)
            return measure(wires=n)

        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def g(x: float):
            qml.RX(x, wires=1)
            return measure(wires=1)

        return jnp.array_equal(f(jnp.pi), g(jnp.pi))

    assert workflow(_in) == _out


@pytest.mark.parametrize(
    "_in,_out",
    [
        (0, False),
        (1, True),
    ],
)
def test_variable_capture_multiple_devices(_in, _out, backend):
    """Test variable capture using multiple backend devices."""

    @qjit()
    def workflow(n: int):
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f(x: float):
            qml.RX(n * x, wires=n)
            return measure(wires=n)

        @qml.qnode(qml.device(backend, wires=2))
        def g(x: float):
            qml.RX(x, wires=1)
            return measure(wires=1)

        return jnp.array_equal(f(jnp.pi), g(jnp.pi))

    assert workflow(_in) == _out


def test_unsupported_device():
    """Test unsupported device."""

    @qml.qnode(qml.device("default.qubit", wires=2))
    def func():
        return qml.probs()

    with pytest.raises(
        CompileError,
        match="is not supported for compilation at the moment.",
    ):
        qjit(func)


def test_qfunc_output_shape_scalar():
    """Check that scalar outputs of QNodes are not wrapped in a list/tuple.
    Note that qjit separately unwraps length-1 return values from the jitted function."""

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    @qjit
    def cost_fn(x: float):
        res = circuit(x)

        assert not isinstance(res, (list, tuple))

        return res * 1j


@pytest.mark.xfail(reason="Preserving scalars is preferred over preserving length-1 containers.")
def test_qfunc_output_shape_list():
    """Check that length-1 list outputs of QNodes are preserved."""

    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit(x):
        qml.RX(x, wires=0)
        return [qml.expval(qml.PauliZ(0))]

    @qjit
    def cost_fn(x: float):
        res = circuit(x)

        assert isinstance(res, list)
        assert len(res) == 1

        return res[0] * 1j


if __name__ == "__main__":
    pytest.main(["-x", __file__])
