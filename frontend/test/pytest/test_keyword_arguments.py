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

    def test_function_with_no_args(self):
        """Test qjit on a function without arguments."""

        @qjit
        def f():
            return 19

        assert f() == 19

    def test_function_with_positional_args(self):
        """Test qjit on a function with positional only arguments."""

        @qjit
        def f(x, y, z, t, /):
            return sum([x, y, z]) * t

        assert f(4, 5, 2, 3) == 33
        # with pytest.raises(TypeError):
        #     f(1, 2, 3, t=4)

    def test_function_with_positional_or_kwargs(self):
        """Test qjit on a function with positional or keyword arguments."""

        @qjit
        def f(x, y):
            return x * y

        assert f(4, 7) == 28
        assert f(x=3, y=-2) == -6
        assert f(2, y=3) == f(2, 3)

    def test_function_with_var_positional(self):
        """Test qjit on a function with variable number of positional arguments."""

        @qjit
        def f(*args):
            return sum(args)

        assert f(1) == 1
        assert f(3, 6) == 9
        assert f(4, 6, 8, 10) == 28

    def test_function_with_keyword_only(self):
        """Test qjit on a function with keyword only arguments."""

        @qjit
        def f(*, x=1, y=3):
            return x - y

        assert f() == -2
        assert f(x=4) == 1
        assert f(y=2) == -1

    def test_function_with_required_keyword_only(self):
        """Test qjit on a function with required keyword only arguments."""

        @qjit
        def f(*, x, y):
            return x / y

        assert f(x=1, y=2) == 1 / 2

        # with pytest.raises(TypeError):
        #     f(1, y=2)

    def test_function_with_variable_kwargs(self):
        """Test qjit on a function with a variable number of keyword arguments."""

        @qjit
        def f(**kwargs):
            return sum(kwargs.values())

        print(f"test output: {f(x=1)}")
        assert f(x=1, y=4, eighteen=18, nineteen=19) == 42

    def test_function_with_kwargs_partial(self):
        """Test that a function works with keyword argument."""

        @qjit
        def f(x, y):
            return x * y

        result = functools.partial(f, y=2)(3)
        assert result == f(2, 3)

    def test_qnode_with_kwargs(self, backend):
        """Test that a qnode works with keyword argument."""
        dev = qml.device(backend, wires=1)

        @qjit
        @qml.qnode(dev)
        def circuit(x, c):
            qml.RY(c, 0)
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        assert jnp.allclose(circuit(0.5, c=0.5), circuit(0.5, 0.5))

    def test_qnode_with_kwargs_switch_order(self, backend):
        """Test that a qnode works with keyword argument."""
        dev = qml.device(backend, wires=1)

        @qjit
        @qml.qnode(dev)
        def circuit(x, c):
            qml.RX(x, wires=0)
            qml.RY(c, wires=0)
            return qml.probs()

        same_order = circuit(c=0.8, x=0.2)
        switched_order = circuit(x=0.2, c=0.8)
        expected = circuit(0.2, 0.8)
        assert jnp.allclose(same_order, expected)
        assert jnp.allclose(switched_order, expected)

    def test_keyword_recompilation(self):
        """
        Test that functions are correctly recompiled when keywords change.
        """

        @qjit
        def circuit(x, y):
            return x + y

        assert circuit(1, y=2) == 3
        assert jnp.allclose(circuit(1, y=jnp.array([2, 2])), jnp.array([3, 3]))

    def test_keyword_ordering(self):
        """
        Test that kwargs in potentially positional positions are recognized as kwargs.
        """

        @qjit
        def foo(x, *args, y=1):
            return x - sum(args) / y

        assert foo(1, 2, 3) == -4
        assert jnp.allclose(foo(1, 2, y=3), 1 / 3)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
