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
"""Test callbacks"""


import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from catalyst.pennylane_extensions import callback, io_callback, pure_callback


@pytest.mark.parametrize("arg", [1, 2, 3])
def test_callback_no_tracing(arg):
    """Test that when there's no tracing the behaviour of identity
    stays the same."""

    @callback
    def identity(x):
        return x

    assert identity(arg) == arg


def test_callback_no_returns_no_params(capsys):
    """Test callback no parameters no returns"""

    @callback
    def my_callback() -> None:
        print("Hello erick")

    @qml.qjit
    def cir():
        my_callback()
        return None

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

    cir()
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello erick"


def test_callback_twice(capsys):
    """Test callback no parameters no returns"""

    @callback
    def my_callback():
        print("Hello erick")

    @qml.qjit
    def cir():
        my_callback()
        return None

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

    cir()
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello erick"

    @qml.qjit
    def cir2():
        my_callback()
        return None

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

    cir2()
    captured = capsys.readouterr()
    assert captured.out.strip() == "Hello erick"


def test_callback_send_param(capsys):
    """Test callback with parameters no returns"""

    @callback
    def my_callback(n) -> None:
        print(n)

    @qml.qjit
    def cir(n):
        my_callback(n)
        return None

    cir(0)
    captured = capsys.readouterr()
    assert captured.out.strip() == "0"


def test_kwargs(capsys):
    """Test kwargs returns"""

    @callback
    def my_callback(**kwargs) -> None:
        for k, v in kwargs.items():
            print(k, v)

    @qml.qjit
    def cir(a, b, c):
        my_callback(a=a, b=b, c=c, d=3, e=4)
        return None

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

    cir(0, 1, 2)
    captured = capsys.readouterr()
    for string in ["a 0", "b 1", "c 2", "d 3", "e 4"]:
        assert string in captured.out


@pytest.mark.parametrize(
    "arg",
    [0, 3.14, complex(0.0, 1.0), jnp.array(0), jnp.array([1, 2, 3]), jnp.array([[1, 2], [2, 3]])],
)
def test_identity_types(arg):
    """Test callback with return values"""

    @callback
    def identity(arg) -> arg:
        """Weird trick, if it is the identity function, we can just pass arg
        as the return type. arg will be abstracted to find the type. This
        just avoids writing out the explicit type once you have a value
        that you know will be the same type as the return type."""
        return arg

    @qml.qjit
    def cir(x):
        return identity(x)

    assert np.allclose(cir(arg), arg)


@pytest.mark.parametrize(
    "arg",
    [jnp.array(0), jnp.array(1)],
)
def test_identity_types_shaped_array(arg):
    """Test callback with return values. Use ShapedArray to denote the type"""

    @callback
    def identity(arg) -> jax.core.ShapedArray([], int):
        return arg

    @qml.qjit
    def cir(x):
        return identity(x)

    assert np.allclose(cir(arg), arg)


@pytest.mark.parametrize(
    "arg",
    [0],
)
def test_multiple_returns(arg):
    """Test callback with multiple return values."""

    @callback
    def identity(arg) -> (int, int):
        return arg, arg

    @qml.qjit
    def cir(x):
        return identity(x)

    assert np.allclose(cir(arg), (arg, arg))


@pytest.mark.parametrize(
    "arg",
    [jnp.array([0.0, 1.0, 2.0])],
)
def test_incorrect_return(arg):
    """Test callback with incorrect return types."""

    @callback
    def identity(arg) -> int:
        return arg

    @qml.qjit
    def cir(x):
        return identity(x)

    with pytest.raises(TypeError, match="Callback identity expected type"):
        cir(arg)


def test_pure_callback():
    """Test identity pure callback."""

    def identity(a):
        return a

    @qml.qjit
    def cir(x):
        return pure_callback(identity, float)(x)

    assert np.allclose(cir(0.0), 0.0)


def test_pure_callback_no_return_value():
    """Test identity pure callback no return."""

    def identity(a):
        return a

    @qml.qjit
    def cir(x):
        return pure_callback(identity)(x)

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        cir(0.0)


def test_io_callback(capsys):
    """Test io callback"""

    def my_own_print(a):
        print(a)

    @qml.qjit
    def cir(x):
        io_callback(my_own_print)(x)
        return None

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

    cir(0)
    captured = capsys.readouterr()
    assert captured.out.strip() == "0"


def test_io_callback_returns_something(capsys):
    """Test io callback returns something"""

    def my_own_print(a):
        print(a)
        return 1

    @qml.qjit
    def cir(x):
        io_callback(my_own_print)(x)
        return None

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

    with pytest.raises(ValueError, match="io_callback is expected to return None"):
        cir(0)
