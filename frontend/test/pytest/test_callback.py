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

from catalyst import accelerate, debug, grad, jacobian, pure_callback
from catalyst.api_extensions.callbacks import base_callback
from catalyst.utils.patching import Patcher


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


@pytest.mark.parametrize(
    "arg",
    [0.1, jnp.array(0.1)],
)
def test_dictionary(arg):
    """Test pytrees. Specifying the type is easier for accelerate since it is
    not needed. But here, we just use the same trick as above where
    we can use any value with the same type as the return to specify
    the return type in a callback.
    """

    @pure_callback
    def callback_fn(x) -> {"helloworld": arg}:
        return {"helloworld": x}

    @qml.qjit
    def f(x):
        return callback_fn(x)["helloworld"]

    assert np.allclose(f(arg), arg)


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


@pytest.mark.parametrize(
    "arg",
    [0.1, jnp.array(0.1), jnp.array([0.1]), jnp.array([0.1, 0.2]), jnp.array([[1, 2], [3, 4]])],
)
def test_accelerate_device(arg):
    """Test with device parameter"""

    @accelerate(dev=jax.devices()[0])
    def identity(x):
        return x

    @qml.qjit
    def qjitted_fn(x):
        return identity(x)

    assert np.allclose(qjitted_fn(arg), arg)


@pytest.mark.parametrize(
    "arg",
    [0.1, jnp.array(0.1), jnp.array([0.1]), jnp.array([0.1, 0.2]), jnp.array([[1, 2], [3, 4]])],
)
def test_accelerate_no_device(arg):
    """Test with no device parameter"""

    @accelerate
    def identity(x):
        return x

    @qml.qjit
    def qjitted_fn(x):
        return identity(x)

    assert np.allclose(qjitted_fn(arg), arg)


@pytest.mark.parametrize(
    "arg",
    [0.1, jnp.array(0.1), jnp.array([0.1]), jnp.array([0.1, 0.2]), jnp.array([[1, 2], [3, 4]])],
)
def test_accelerate_no_device_inside(arg):
    """Test with no device parameter accelerate is inside qjit"""

    @qml.qjit
    def qjitted_fn(x):
        @accelerate
        def identity(x):
            return x

        return identity(x)

    assert np.allclose(qjitted_fn(arg), arg)


@pytest.mark.parametrize(
    "arg",
    [0.1, jnp.array(0.1), jnp.array([0.1]), jnp.array([0.1, 0.2]), jnp.array([[1, 2], [3, 4]])],
)
def test_accelerate_no_device_autograph(arg):
    """Test with no device parameter"""

    @accelerate
    def identity(x):
        return x

    @qml.qjit(autograph=True)
    def qjitted_fn(x):
        return identity(x)

    assert np.allclose(qjitted_fn(arg), arg)


@pytest.mark.parametrize(
    "arg",
    [0.1, jnp.array(0.1), jnp.array([0.1]), jnp.array([0.1, 0.2]), jnp.array([[1, 2], [3, 4]])],
)
def test_accelerate_manual_jax_jit(arg):
    """Test with no device parameter"""

    @accelerate
    @jax.jit
    def identity(x):
        return x

    @qml.qjit
    def qjitted_fn(x):
        return identity(x)

    assert np.allclose(qjitted_fn(arg), arg)


def test_jax_jit_returns_nothing(capsys):
    """This is more a question for reviewer"""

    @accelerate
    def noop(x):
        jax.debug.print("x={x}", x=x)

    @qml.qjit
    def func(x: float):
        noop(x)
        return x

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

    func(1.0)
    captured = capsys.readouterr()
    assert captured.out.strip() == "x=1.0"


def test_non_jax_jittable():
    """Test that error is raised when jax-jit fails"""

    @accelerate
    def impossible(x):
        if x:
            return 0
        return 1

    msg = "Function impossible must be jax.jit-able"
    with pytest.raises(ValueError, match=msg):

        @qml.qjit
        def func(x: bool):
            return impossible(x)


def test_that_jax_jit_is_called():
    """Test that jax.jit is called"""

    called_jax_jit = False

    builtin_jax_jit = jax.jit

    def mock_jax_jit(func):
        nonlocal called_jax_jit
        called_jax_jit = True
        return builtin_jax_jit(func)

    with Patcher((jax, "jit", mock_jax_jit)):

        @accelerate
        def identity(x):
            return x

        @qml.qjit
        def wrapper(x):
            return identity(x)

        wrapper(1.0)

    assert called_jax_jit


def test_callback_cache():
    """Test callback cache. This test is for coverage."""

    @debug.callback
    def hello_world():
        print("hello world")

    @qml.qjit
    def wrapper():
        hello_world()
        hello_world()


@pytest.mark.parametrize("arg", [(0.1), (0.2), (0.3)])
def test_inactive_debug_grad(capsys, arg):
    """Test that debug callback can be differentiated
    and not affects the output"""

    @qml.qjit
    @grad
    def identity(x: float):
        debug.print(x)
        return x

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

    assert np.allclose(identity(arg), 1.0)

    captured = capsys.readouterr()
    assert captured.out.strip() == str(arg)


@pytest.mark.parametrize("arg", [jax.numpy.identity(2, dtype=float)])
def test_inactive_debug_jacobian(capsys, arg):
    """Test that debug callback can be differentiated
    and not affects the output"""

    @qml.qjit
    @jacobian
    def identity(x):
        debug.print(x)
        return x

    @jax.jit
    @jax.jacobian
    def identity_jax(x):
        return x

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

    assert np.allclose(identity(arg), identity_jax(arg))

    captured = capsys.readouterr()
    assert str(arg) in captured.out.strip()


@pytest.mark.parametrize("scale", [(0.1), (0.2), (0.3)])
def test_active_grad_no_tape(scale):
    """Test that pure callback can be differentiated no tape"""

    @pure_callback
    def identity(x) -> float:
        return x

    @identity.fwd
    def fwd(x):
        # Still needs to return a tuple.
        return identity(x), None

    @identity.bwd
    def bwd(_res, cot):
        return cot

    @qml.qjit
    @grad
    def wrapper(x):
        return scale * identity(x)

    assert np.allclose(wrapper(42.0), scale)


@pytest.mark.parametrize("scale", [(0.1), (0.2), (0.3)])
def test_active_grad_tape(scale):
    """Test that pure callback can be differentiated with a tape"""

    @pure_callback
    def identity(x) -> float:
        return x

    @identity.fwd
    def fwd(x):
        return identity(x), 1.0

    @identity.bwd
    def bwd(res, cot):
        return cot * res

    @qml.qjit
    @grad
    def wrapper(x):
        return scale * identity(x)

    assert np.allclose(wrapper(42.0), scale)


@pytest.mark.parametrize("scale", [(0.1), (0.2), (0.3)])
@pytest.mark.parametrize("space", [(2), (3), (4)])
def test_active_grad_many_residuals(scale, space):
    """Test that pure callback can be differentiated with many residuals"""

    @pure_callback
    def identity(x) -> float:
        return x

    @identity.fwd
    def fwd(x):
        tape = [1 / space] * space
        return identity(x), tuple(tape)

    @identity.bwd
    def bwd(res, cot):
        return cot * sum(res)

    @qml.qjit
    @grad
    def wrapper(x):
        return scale * identity(x)

    assert np.allclose(wrapper(42.0), scale)


@pytest.mark.parametrize("scale", [(0.1), (0.2), (0.3)])
@pytest.mark.parametrize("space", [(2), (3), (4)])
def test_active_jacobian_many_residuals(scale, space):
    """Test that pure callback can be differentiated with many residuals"""

    # This is a hack, just for the type
    arg = jax.numpy.identity(2, dtype=float)

    @pure_callback
    def identity(x) -> arg:
        return x

    @identity.fwd
    def fwd(x):
        tape = [1 / space] * space
        return identity(x), tuple(tape)

    @identity.bwd
    def bwd(res, cot):
        return cot * sum(res)

    @qml.qjit
    @jacobian
    def wrapper(x):
        return scale * identity(x)

    @jax.jit
    @jax.jacobian
    def wrapper_jax(x):
        return scale * x

    assert np.allclose(wrapper(arg), wrapper_jax(arg))


@pytest.mark.parametrize("arg0", [(0.1), (0.2), (0.3)])
@pytest.mark.parametrize("arg1", [(2.0), (3.0), (4.0)])
def test_example_from_story(arg0, arg1):
    """Just exactly the same function on the spec
    modulo errors in the example
    """

    @pure_callback
    def some_func(x, y) -> float:
        return jnp.sin(x) * y

    @some_func.fwd
    def some_func_fwd(x, y):
        return some_func(x, y), (jnp.cos(x), jnp.sin(x), y)

    @some_func.bwd
    def some_func_bws(res, dy):
        cos_x, sin_x, y = res  # Gets residuals computed in f_fwd
        return (cos_x * dy * y, sin_x * dy)

    @qml.qjit
    @grad
    def cost(x, y):
        return jnp.sin(some_func(jnp.cos(x), y))

    @jax.jit
    @jax.grad
    def jax_jit_cost(x, y):
        return jnp.sin(jnp.sin(jnp.cos(x)) * y)

    assert np.allclose(jax_jit_cost(arg0, arg1), cost(arg0, arg1))


@pytest.mark.parametrize("scale", [(0.1), (0.2), (0.3)])
def test_active_grad_inside_qjit(backend, scale):
    """Test that pure callback can be differentiated no tape"""

    @pure_callback
    def identity(x) -> float:
        return x

    @identity.fwd
    def fwd(x):
        # Still needs to return a tuple.
        return identity(x), None

    @identity.bwd
    def bwd(_res, cot):
        return cot

    @qml.qjit
    @grad
    @qml.qnode(qml.device(backend, wires=1))
    def wrapper(x):
        param = scale * identity(x)
        qml.RX(param, wires=0)
        return qml.expval(qml.PauliZ(0))

    @jax.jit
    @qml.grad
    @qml.qnode(qml.device(backend, wires=1))
    def wrapper_jit(x):
        param = scale * identity(x)
        qml.RX(param, wires=0)
        return qml.expval(qml.PauliZ(0))

    assert np.allclose(wrapper_jit(42.0), wrapper(42.0))


if __name__ == "__main__":
    pytest.main(["-x", __file__])
