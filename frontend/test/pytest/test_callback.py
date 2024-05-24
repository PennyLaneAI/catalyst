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


from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from catalyst import debug, pure_callback
from catalyst.api_extensions.callbacks import ActiveCallback, base_callback
from catalyst.utils.exceptions import DifferentiableCompileError


class TestActiveCallback:

    def test_allow_no_custom_gradient(self):
        f = lambda x: x
        active_callback = ActiveCallback(f, float)
        observed_jaxpr = jax.make_jaxpr(active_callback)(1.0)
        expected_jaxpr = jax.make_jaxpr(f)(1.0)
        assert str(observed_jaxpr) == str(expected_jaxpr)

    def test_error_message_only_fwd_defined(self):
        f = lambda x: x
        fwd = lambda x: 1.0
        active_callback = ActiveCallback(f, float)
        active_callback.fwd(fwd)
        msg = ".*differentiated but missing reverse pass"
        with pytest.raises(DifferentiableCompileError, match=msg):
            jax.make_jaxpr(active_callback)(1.0)

    def test_error_message_only_bwd_defined(self):
        f = lambda x: x
        bwd = lambda x: x
        active_callback = ActiveCallback(f, float)
        active_callback.bwd(bwd)
        msg = ".*differentiated but missing forward pass"
        with pytest.raises(DifferentiableCompileError, match=msg):
            jax.make_jaxpr(active_callback)(1.0)

    def test_fwd_bwd_jaxpr_fields_are_set(self):
        f = lambda x: x
        fwd = lambda x: (x, 1.0)
        bwd = lambda res, cotangents: res * cotangents
        active_callback = ActiveCallback(f, float)
        active_callback.fwd(fwd)
        active_callback.bwd(bwd)
        jax.make_jaxpr(active_callback)(1.0)
        assert active_callback.forward_jaxpr
        assert active_callback.reverse_jaxpr


@pytest.mark.parametrize("arg", [1, 2, 3])
def test_callback_no_tracing(arg):
    """Test that when there's no tracing the behaviour of identity
    stays the same."""

    @base_callback
    def identity(x):
        return x

    assert identity(arg) == arg


def test_callback_no_returns_no_params(capsys):
    """Test callback no parameters no returns"""

    @base_callback
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

    @base_callback
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

    @base_callback
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

    @base_callback
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


def test_simple_increment():
    """Test increment function"""

    @base_callback
    def inc(arg) -> int:
        return arg + 1

    @qml.qjit
    def cir(arg):
        return inc(arg)

    assert np.allclose(cir(0), 1)


@pytest.mark.parametrize(
    "arg",
    [0, 3.14, complex(0.0, 1.0), jnp.array(0), jnp.array([1, 2, 3]), jnp.array([[1, 2], [2, 3]])],
)
def test_identity_types(arg):
    """Test callback with return values"""

    @base_callback
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

    @base_callback
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

    @base_callback
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

    @base_callback
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


def test_pure_callback_decorator():
    """Test identity pure callback."""

    @pure_callback
    def identity(a) -> float:
        return a

    @qml.qjit
    def cir(x):
        return identity(x)

    assert np.allclose(cir(0.0), 0.0)


def test_pure_callback_no_return_value():
    """Test identity pure callback no return."""

    def identity(a):
        return a

    @qml.qjit
    def cir(x):
        return pure_callback(identity)(x)

    msg = "A function using pure_callback requires return types to be "
    msg += "passed in as a parameter or type annotation."
    with pytest.raises(TypeError, match=msg):
        cir(0.0)


def test_debug_callback(capsys):
    """Test debug callback"""

    def my_own_print(a):
        print(a)

    @qml.qjit
    def cir(x):
        debug.callback(my_own_print)(x)
        return None

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

    cir(0)
    captured = capsys.readouterr()
    assert captured.out.strip() == "0"


def test_debug_callback_decorator(capsys):
    """Test debug callback"""

    @debug.callback
    def my_own_print(a):
        print(a)

    @qml.qjit
    def cir(x):
        my_own_print(x)
        return None

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

    cir(0)
    captured = capsys.readouterr()
    assert captured.out.strip() == "0"


def test_debug_callback_returns_something(capsys):
    """Test io callback returns something"""

    def my_own_print(a):
        print(a)
        return 1

    @qml.qjit
    def cir(x):
        debug.callback(my_own_print)(x)
        return None

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

    with pytest.raises(ValueError, match="debug.callback is expected to return None"):
        cir(0)


def test_io_callback_modify_global(capsys):
    """Test mutation"""

    x = 0

    @debug.callback
    def set_x_to(y):
        nonlocal x
        x = y

    @debug.callback
    def print_x():
        nonlocal x
        print(x)

    @qml.qjit
    def cir():
        print_x()
        set_x_to(1)
        print_x()

    cir()

    captured = capsys.readouterr()
    assert captured.out.strip() == "0\n1"


@pytest.mark.parametrize(
    "arg",
    [0.1, jnp.array(0.1)],
)
def test_no_return_list(arg):
    """Test that the callback returns a scalar and not a list."""

    @pure_callback
    def callback_fn(x) -> float:
        return np.sin(x)

    @qml.qjit
    def f(x):
        res = callback_fn(x**2)
        assert not isinstance(res, Sequence)
        return jnp.cos(res)

    f(arg)


def test_tuple_out():
    """Test with multiple tuples."""

    @pure_callback
    def callback_fn(x) -> (bool, bool):
        return x > 1.0, x > 2.0

    @qml.qjit
    def f(x):
        res = callback_fn(x**2)
        assert isinstance(res, tuple) and len(res) == 2
        return jnp.cos(res[0])

    f(0.1)


def test_numpy_ufuncs():
    """Test with numpy ufuncs."""

    @qml.qjit
    def f(x):
        y = pure_callback(np.sin, float)(x)
        return y

    assert np.allclose(np.sin(1.0 / 2.0), f(1.0 / 2.0))


if __name__ == "__main__":
    pytest.main(["-x", __file__])
