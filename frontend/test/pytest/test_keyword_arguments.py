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

"""Test features related to keyword arguments."""

import functools

import jax.numpy as jnp
import pennylane as qml
import pytest

from catalyst import qjit


class TestKeywordArguments:
    """Test QJIT with keyword arguments."""

    def test_function_with_kwargs(self):
        """Test that a function works with keyword argeument."""

        @qjit()
        def f(x, y):
            return x * y

        result = f(3, y=2)
        assert result == f(2, 3)

    def test_function_with_kwargs_partial(self):
        """Test that a function works with keyword argeument."""

        @qjit()
        def f(x, y):
            return x * y

        result = functools.partial(f, y=2)(3)
        assert result == f(2, 3)

    def test_qnode_with_kwargs(self, backend):
        """Test that a qnode works with keyword argeument."""
        dev = qml.device(backend, wires=1)

        @qjit()
        @qml.qnode(dev)
        def circuit(x, c):
            qml.RY(c, 0)
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        assert jnp.allclose(circuit(0.5, c=0.5), circuit(0.5, 0.5))


if __name__ == "__main__":
    pytest.main(["-x", __file__])
