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
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from catalyst import accelerate, debug, grad, jacobian, pure_callback, qjit
from catalyst.api_extensions.callbacks import base_callback
from catalyst.utils.exceptions import DifferentiableCompileError
from catalyst.utils.patching import Patcher

# pylint: disable=protected-access,too-many-lines


@pytest.mark.parametrize("arg", [1, 2, 3])
def test_callback_no_tracing(arg):
    """Test that when there's no tracing the behaviour of identity
    stays the same."""

    @base_callback
    def identity(x):
        return x

    assert identity(arg) == arg


@pytest.mark.parametrize("arg", [1, 2, 3])
def test_purecallback_no_tracing(arg):
    """Test that when there's no tracing the behaviour of identity
    stays the same."""

    @pure_callback
    def identity(x) -> int:
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

    # NOTE: Currently, this will raise a TypeError exception on Linux and the same exception wrapped
    # as a RuntimeError on macOS. This appears to be related to using nanobind for the Python/C++
    # bindings. To avoid separate test cases for Linux and macOS, we accept either exception type
    # here and match on the string below, which should be contained in the messages of both.
    # TODO: Why does this happen?
    with pytest.raises((TypeError, RuntimeError), match="Callback identity expected type"):
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

    # NOTE: Currently, this will raise a ValueError exception on Linux and the same exception wrapped
    # as a RuntimeError on macOS. This appears to be related to using nanobind for the Python/C++
    # bindings. To avoid separate test cases for Linux and macOS, we accept either exception type
    # here and match on the string below, which should be contained in the messages of both.
    # TODO: Why does this happen?
    with pytest.raises(
        (ValueError, RuntimeError), match="debug.callback is expected to return None"
    ):
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
        return np.sin(x) * y

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
        # This one cannot have np.sin so we just inline it and change it to jnp.sin
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


@pytest.mark.parametrize(
    "arg", [jnp.array([0.1, 0.2]), jnp.array([0.2, 0.3]), jnp.array([0.3, 0.4])]
)
def test_array_input(arg):
    """Test array input"""

    @pure_callback
    def some_func(x) -> float:
        return np.sin(x[0]) * x[1]

    @some_func.fwd
    def some_func_fwd(x):
        return some_func(x), (jnp.cos(x[0]), jnp.sin(x[0]), x[1])

    @some_func.bwd
    def some_func_bwd(res, dy):
        cos_x0, sin_x0, x1 = res  # Gets residuals computed in f_fwd
        # since there is a single array parameter, we return
        # a VJP which is a tuple of length 1 containing an array
        # parameter of the same shape
        return (jnp.array([cos_x0 * dy * x1, sin_x0 * dy]),)

    @qml.qjit
    @grad
    def cost(x):
        y = jnp.array([jnp.cos(x[0]), x[1]])
        return jnp.sin(some_func(y))

    @jax.jit
    @jax.grad
    def cost_jax(x):
        y = jnp.array([jnp.cos(x[0]), x[1]])
        return jnp.sin(jnp.sin(y[0]) * y[1])

    assert np.allclose(cost(arg), cost_jax(arg))


def test_array_in_scalar_out():
    """Test array in scalar out"""

    @pure_callback
    def some_func(x) -> float:
        return np.sin(x[0]) * x[1]

    @some_func.fwd
    def some_func_fwd(x):
        return some_func(x), (jnp.cos(x[0]), jnp.sin(x[0]), x[1])

    @some_func.bwd
    def some_func_bws(res, dy):
        cos_x0, sin_x0, x1 = res
        return (jnp.array([cos_x0 * dy * x1, sin_x0 * dy]),)

    @qml.qjit
    @grad
    def result(x):
        y = jnp.array([jnp.cos(x[0]), x[1]])
        return jnp.sin(some_func(y))

    @jax.jit
    @jax.grad
    def expected(x):
        y = jnp.array([jnp.cos(x[0]), x[1]])
        return jnp.sin(jnp.sin(y[0]) * y[1])

    x = jnp.array([1.0, 0.5])
    assert np.allclose(result(x), expected(x))

    # Array([-0.34893507,  0.49747506], dtype=float64)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_scalar_in_array_out(dtype):
    """Test scalar in array out"""

    @pure_callback
    def some_func(x) -> jax.ShapeDtypeStruct((2,), dtype):
        return np.array([np.sin(x), np.cos(x)], dtype=dtype)

    @some_func.fwd
    def some_func_fwd(x):
        return some_func(x), x

    @some_func.bwd
    def some_func_bws(res, dy):
        x = res
        return (jnp.array([jnp.cos(x), -jnp.sin(x)]) @ dy,)

    @qml.qjit
    @grad
    def result(x):
        return jnp.sum(some_func(jnp.sin(x)))

    @jax.jit
    @jax.grad
    def expected(x):
        x = jnp.sin(x)
        return jnp.sin(x) + jnp.cos(x)

    x = 0.435
    assert np.allclose(result(x), expected(x))

    # Array(0.4565774, dtype=float64)


def test_scalar_in_array_out_float32_wrong():
    """Test float32 support in pure callbacks, result in type mismatch"""

    @pure_callback
    def some_func(x) -> jax.ShapeDtypeStruct((2,), jnp.float32):
        return np.array([np.sin(x), np.cos(x)])

    @some_func.fwd
    def some_func_fwd(x):
        return some_func(x), x

    @some_func.bwd
    def some_func_bws(res, dy):
        x = res
        return (jnp.array([jnp.cos(x), -jnp.sin(x)]) @ dy,)

    @qml.qjit
    @grad
    def result(x):
        return jnp.sum(some_func(jnp.sin(x)))

    x = 0.435
    # NOTE: Currently, this will raise a TypeError exception on Linux and the same exception wrapped
    # as a RuntimeError on macOS. This appears to be related to using nanobind for the Python/C++
    # bindings. To avoid separate test cases for Linux and macOS, we accept either exception type
    # here and match on the string below, which should be contained in the messages of both.
    # TODO: Why does this happen?
    with pytest.raises((TypeError, RuntimeError), match="Callback some_func expected type"):
        result(x)


def test_scalar_in_tuple_scalar_array_out():
    """Test scalar in tuple scalar array out"""

    @pure_callback
    def some_func(
        x,
    ) -> (jax.ShapeDtypeStruct(tuple(), jnp.float64), jax.ShapeDtypeStruct((2,), jnp.float64)):
        y = x**2, np.array([np.sin(x), np.cos(x)])
        return y

    @some_func.fwd
    def some_func_fwd(x):
        return some_func(x), x

    @some_func.bwd
    def some_func_bwd(res, dy):
        x = res
        vjp0 = 2 * x * dy[0]
        vjp1 = jnp.array([jnp.cos(x), -jnp.sin(x)]) @ dy[1]
        return (vjp0 + vjp1,)

    @qml.qjit
    @grad
    def result(x):
        a, b = some_func(jnp.sin(x))
        return a + jnp.sum(b)

    @jax.jit
    @jax.grad
    def expected(x):
        x = jnp.sin(x)
        return jnp.sin(x) + jnp.cos(x) + x**2

    x = 0.435
    assert np.allclose(result(x), expected(x))

    # Array(1.2209063, dtype=float64)


def test_array_in_tuple_array_out():
    """Test array in tuple array out"""

    def _some_func(x):
        return np.sin(x), x**2

    @pure_callback
    def some_func(
        x,
    ) -> (jax.ShapeDtypeStruct((2,), jnp.float64), jax.ShapeDtypeStruct((2,), jnp.float64)):
        return np.sin(x), x**2

    @some_func.fwd
    def some_func_fwd(x):
        return some_func(x), x

    @some_func.bwd
    def some_func_bws(res, dy):
        x = res
        vjp0 = jnp.cos(x) * dy[0]
        vjp1 = 2 * x * dy[1]
        return (vjp0 + vjp1,)

    @qml.qjit
    @grad
    def result(x):
        return jnp.dot(*some_func(jnp.sin(x)))

    @jax.jit
    @jax.grad
    def expected(x):
        x = jnp.sin(x)
        return jnp.dot(jnp.sin(x), x**2)

    x = jnp.array([1.0, 0.5])
    assert np.allclose(result(x), expected(x))

    # Array([0.9329284 , 0.56711537], dtype=float64)


def test_tuple_array_in_tuple_array_out():
    """Test tuple array in tuple array out"""

    @pure_callback
    def some_func(
        x, y
    ) -> (jax.ShapeDtypeStruct((2,), jnp.float64), jax.ShapeDtypeStruct((2,), jnp.float64)):
        return np.sin(x) @ y, y**2

    @some_func.fwd
    def some_func_fwd(x, y):
        return some_func(x, y), (x, y)

    @some_func.bwd
    def some_func_bwd(res, dy):
        x, y = res
        vjp0 = y * jnp.cos(x) * jnp.reshape(dy[0], (-1, 1))
        vjp1 = dy[0] @ jnp.sin(x) + 2 * y * dy[1]
        return (vjp0, vjp1)

    @qml.qjit
    @partial(grad, argnums=[0, 1])
    def result(x, y):
        return jnp.dot(*some_func(x, y**2))

    @jax.jit
    @partial(jax.grad, argnums=[0, 1])
    def expected(x, y):
        return jnp.dot(jnp.sin(x) @ y**2, (y**2) ** 2)

    x = jnp.array([[1.0, 0.5], [0.12, -1.2]])
    y = jnp.array([-0.6, 0.2])
    flat_results_obs, obs_shape = jax._src.tree_util.tree_flatten(result(x, y))
    flat_results_exp, exp_shape = jax._src.tree_util.tree_flatten(expected(x, y))
    assert obs_shape == exp_shape
    for obs, exp in zip(flat_results_obs, flat_results_exp):
        assert np.allclose(obs, exp)

    # (Array([[2.5208345e-02, 4.5493888e-03],
    #         [5.7185785e-04, 2.3190898e-05]], dtype=float64),
    #  Array([-0.40939555,  0.02444299], dtype=float64))


def test_pytree_in_pytree_out():
    """Test pytree in pytree out"""

    shape = {
        "one": jax.ShapeDtypeStruct((2,), jnp.float64),
        "two": jax.ShapeDtypeStruct((2,), jnp.float64),
    }

    @pure_callback
    def some_func(weights) -> shape:
        return {"one": np.sin(weights["x"]) @ weights["y"], "two": weights["y"] ** 2}

    @some_func.fwd
    def some_func_fwd(weights):
        return some_func(weights), weights

    @some_func.bwd
    def some_func_bwd(res, dy):
        vjp0 = res["y"] * jnp.cos(res["x"]) * jnp.reshape(dy["one"], (-1, 1))
        vjp1 = dy["one"] @ jnp.sin(res["x"]) + 2 * res["y"] * dy["two"]
        return ({"x": vjp0, "y": vjp1},)

    @qml.qjit
    @grad
    def result(weights):
        weights["y"] = weights["y"] ** 2
        res = some_func(weights)
        return jnp.dot(res["one"], res["two"])

    @jax.jit
    @jax.grad
    def expected(weights):
        x = weights["x"]
        y = weights["y"]
        return jnp.dot(jnp.sin(x) @ y**2, y**4)

    weights = {"x": jnp.array([[1.0, 0.5], [0.12, -1.2]]), "y": jnp.array([-0.6, 0.2])}
    flat_results_obs, obs_shape = jax._src.tree_util.tree_flatten(result(weights))
    flat_results_exp, exp_shape = jax._src.tree_util.tree_flatten(expected(weights))
    assert obs_shape == exp_shape
    for obs, exp in zip(flat_results_obs, flat_results_exp):
        assert np.allclose(obs, exp)

    # {'x': Array([[2.5208345e-02, 4.5493888e-03],
    #        [5.7185785e-04, 2.3190898e-05]], dtype=float64),
    # 'y': Array([-0.40939555,  0.02444299], dtype=float64)}


def test_callback_backwards_function():
    """Test a workflow where the bwd function is also using
    a callback to compute the VJP"""

    shape = {
        "one": jax.ShapeDtypeStruct((2,), jnp.float64),
        "two": jax.ShapeDtypeStruct((2,), jnp.float64),
    }
    vjp_shape = {
        "x": jax.ShapeDtypeStruct((2, 2), jnp.float64),
        "y": jax.ShapeDtypeStruct((2,), jnp.float64),
    }

    @pure_callback
    def some_func(weights) -> shape:
        return {"one": np.sin(weights["x"]) @ weights["y"], "two": weights["y"] ** 2}

    @some_func.fwd
    def some_func_fwd(weights):
        return some_func(weights), weights

    @pure_callback
    def some_func_bwd_vjp(res, dy) -> vjp_shape:
        vjp0 = res["y"] * np.cos(res["x"]) * np.reshape(dy["one"], (-1, 1))
        vjp1 = dy["one"] @ np.sin(res["x"]) + 2 * res["y"] * dy["two"]
        return ({"x": vjp0, "y": vjp1},)

    @some_func.bwd
    def some_func_bwd(res, dy):
        return some_func_bwd_vjp(res, dy)

    @qml.qjit
    @grad
    def result(weights):
        weights["y"] = weights["y"] ** 2
        res = some_func(weights)
        return jnp.dot(res["one"], res["two"])

    @jax.jit
    @jax.grad
    def expected(weights):
        x = weights["x"]
        y = weights["y"]
        return jnp.dot(jnp.sin(x) @ y**2, y**4)

    weights = {"x": jnp.array([[1.0, 0.5], [0.12, -1.2]]), "y": jnp.array([-0.6, 0.2])}
    flat_results_obs, obs_shape = jax._src.tree_util.tree_flatten(result(weights))
    flat_results_exp, exp_shape = jax._src.tree_util.tree_flatten(expected(weights))
    assert obs_shape == exp_shape
    for obs, exp in zip(flat_results_obs, flat_results_exp):
        assert np.allclose(obs, exp)


def test_different_shapes():
    """Different input output shape"""

    def fun(x):
        y = jnp.array([3.0, 4.0, 5.0])
        return y * x[0] + x[1]

    # hack for the type
    ty = jnp.array([2.0, 2.0, 2.0])

    @pure_callback
    def fun_callback(x) -> ty:
        return fun(x)

    f_vjp = None

    @pure_callback
    def fun_fwd_callback(x) -> ty:
        nonlocal f_vjp
        primals, f_vjp = jax.vjp(fun, x)
        return primals

    @fun_callback.fwd
    def fun_fwd(x):
        return fun_fwd_callback(x), None

    @pure_callback
    def fun_bwd_callback(cot) -> jnp.array([1.0, 1.0]):
        nonlocal f_vjp
        return f_vjp(cot)

    @fun_callback.bwd
    def fun_bwd(_res, cot):
        return fun_bwd_callback(cot)

    @qml.qjit
    @jacobian
    def wrapper(x):
        return fun_callback(x)

    @jax.jit
    @jacobian
    def wrapper_jax(x):
        return fun(x)

    arg = jax.numpy.array([3.14, 0.001519])
    assert np.allclose(wrapper(arg), wrapper_jax(arg))


def test_multiply_two_matrices_to_get_something_with_different_dimensions():
    """matrix multiplication with constant"""

    A = jax.numpy.array([[1.0, 2.0], [1.0, 3.0], [3.0, 2.0]])

    B = jax.numpy.array([[1.0], [2.0]])

    # Just for the type
    C = A @ B

    def matrix_multiply(X):
        return X @ B

    f_vjp = None

    def matrix_multiply_keep_state(X):
        nonlocal f_vjp
        primals, f_vjp = jax.vjp(matrix_multiply, X)
        return primals

    @pure_callback
    def matrix_multiply_vjp(cotangents) -> A:
        nonlocal f_vjp
        retval = f_vjp(cotangents)
        return retval

    @pure_callback
    def matrix_multiply_callback(X) -> C:
        return matrix_multiply_keep_state(X)

    @matrix_multiply_callback.fwd
    def matrix_multiply_fwd(X):
        return matrix_multiply_callback(X), None

    @matrix_multiply_callback.bwd
    def matrix_multiply_bwd(_residuals, cotangents):
        return matrix_multiply_vjp(cotangents)

    @qml.qjit
    @jacobian
    def mul(X):
        return matrix_multiply_callback(X)

    @jax.jit
    @jacobian
    def mul_jax(A):
        return A @ B

    assert np.allclose(mul(A), mul_jax(A))


def test_multiply_two_matrices_to_get_something_with_different_dimensions2():
    """Matrix multiply argnums=0"""

    A = jax.numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    B = jax.numpy.array([[7.0], [8.0]])

    # Just for the type
    C = A @ B

    def matrix_multiply(X, Y):
        return X @ Y

    f_vjp = None

    def matrix_multiply_keep_state(X, Y):
        nonlocal f_vjp
        primals, f_vjp = jax.vjp(matrix_multiply, X, Y)
        return primals

    @pure_callback
    def matrix_multiply_vjp(cotangents) -> A:
        nonlocal f_vjp
        nonlocal A, B
        retval = f_vjp(cotangents)
        return retval[0]

    @pure_callback
    def matrix_multiply_callback(X, Y) -> C:
        return matrix_multiply_keep_state(X, Y)

    @matrix_multiply_callback.fwd
    def matrix_multiply_fwd(X, Y):
        return matrix_multiply_callback(X, Y), None

    @matrix_multiply_callback.bwd
    def matrix_multiply_bwd(_residuals, cotangents):
        return matrix_multiply_vjp(cotangents)

    @qml.qjit
    @jacobian
    def mul(X, Y):
        return matrix_multiply_callback(X, Y)

    @jax.jit
    @jacobian
    def mul_jax(A, B):
        return A @ B

    assert np.allclose(mul(A, B), mul_jax(A, B))


@pytest.mark.xfail(reason="error in gradient frontend verification")
def test_multiply_two_matrices_to_get_something_with_different_dimensions3():
    """I want to run this test but I can't. I get an error saying that grad
    is for scalars only but I am using jacobian"""

    A = jax.numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    B = jax.numpy.array([[7.0], [8.0]])

    # Just for the type
    C = A @ B

    def matrix_multiply(X, Y):
        return X @ Y

    f_vjp = None

    def matrix_multiply_keep_state(X, Y):
        nonlocal f_vjp
        primals, f_vjp = jax.vjp(matrix_multiply, X, Y)
        return primals

    @pure_callback
    def matrix_multiply_vjp(cotangents) -> (A, B):
        nonlocal f_vjp
        nonlocal A, B
        retval = f_vjp(cotangents)
        return retval

    @pure_callback
    def matrix_multiply_callback(X, Y) -> C:
        return matrix_multiply_keep_state(X, Y)

    @matrix_multiply_callback.fwd
    def matrix_multiply_fwd(X, Y):
        return matrix_multiply_callback(X, Y), None

    @matrix_multiply_callback.bwd
    def matrix_multiply_bwd(_residuals, cotangents):
        return matrix_multiply_vjp(cotangents)

    @qml.qjit
    @jacobian(argnums=[0, 1])
    def mul(X, Y):
        return matrix_multiply_callback(X, Y)

    @jax.jit
    def mul_jax(A, B):
        return jax.jacobian(A @ B, argnums=[0, 1])(A, B)

    assert np.allclose(mul(A, B), mul_jax(A, B))


@pytest.mark.parametrize("arg", [jnp.array([[0.1, 0.2], [0.3, 0.4]])])
@pytest.mark.parametrize("order", ["truth_hypo", "hypo_truth"])
def test_vjp_as_residual(arg, order):
    """See https://github.com/PennyLaneAI/catalyst/issues/852"""

    def jax_callback(fn, result_type):

        @pure_callback
        def callback_fn(*args) -> result_type:
            return fn(*args)

        @callback_fn.fwd
        def callback_fn_fwd(*args):
            ans, vjp_func = accelerate(lambda *x: jax.vjp(fn, *x))(*args)
            return ans, vjp_func

        @callback_fn.bwd
        def callback_fn_bwd(vjp_func, dy):
            return accelerate(vjp_func)(dy)

        return callback_fn

    @qml.qjit
    @jacobian
    def hypothesis(x):
        expm = jax_callback(jax.scipy.linalg.expm, jax.ShapeDtypeStruct((2, 2), jnp.float64))
        return expm(x)

    @jax.jacobian
    def ground_truth(x):
        return jax.scipy.linalg.expm(x)

    if order == "hypo_truth":
        obs = hypothesis(arg)
        exp = ground_truth(arg)
    else:
        exp = ground_truth(arg)
        obs = hypothesis(arg)
    assert np.allclose(obs, exp)


@pytest.mark.parametrize("arg", [jnp.array([[0.1, 0.2], [0.3, 0.4]])])
@pytest.mark.parametrize("order", ["truth_hypo", "hypo_truth"])
def test_vjp_as_residual_automatic(arg, order):
    """Test automatic differentiation of accelerated function"""

    @qml.qjit
    @jacobian
    def hypothesis(x):
        return accelerate(jax.scipy.linalg.expm)(x)

    @jax.jacobian
    def ground_truth(x):
        return jax.scipy.linalg.expm(x)

    if order == "hypo_truth":
        obs = hypothesis(arg)
        exp = ground_truth(arg)
    else:
        exp = ground_truth(arg)
        obs = hypothesis(arg)
    assert np.allclose(obs, exp)


@pytest.mark.parametrize("arg", [jnp.array([[0.1, 0.2], [0.3, 0.4]])])
def test_example_from_epic(arg):
    """Test example from epic"""

    @qml.qjit
    @grad
    def hypothesis(x):
        expm = accelerate(jax.scipy.linalg.expm)
        return jnp.sum(expm(jnp.sin(x) ** 2))

    @jax.jit
    @jax.grad
    def ground_truth(x):
        expm = jax.scipy.linalg.expm
        return jnp.sum(expm(jnp.sin(x) ** 2))

    obs = hypothesis(arg)
    exp = ground_truth(arg)
    assert np.allclose(obs, exp)


def test_automatic_differentiation_of_accelerate():
    """Same but easier"""

    @qml.qjit
    @grad
    @accelerate
    def identity(x: float):
        return x

    assert identity(4.0) == 1.0


def test_error_incomplete_grad_only_forward():
    """Test error about missing reverse pass"""

    @pure_callback
    def identity(x) -> float:
        return x

    @identity.fwd
    def fwd(x):
        return identity(x), None

    @grad
    def wrapper(x: float):
        return identity(x)

    with pytest.raises(DifferentiableCompileError, match="missing reverse pass"):
        qjit(wrapper)


def test_error_incomplete_grad_only_reverse():
    """Test error about missing forward pass"""

    @pure_callback
    def identity(x) -> float:
        return x

    @identity.bwd
    def bwd(_res, cot):
        return cot

    @grad
    def wrapper(x: float):
        return identity(x)

    with pytest.raises(DifferentiableCompileError, match="missing forward pass"):
        qjit(wrapper)


def test_nested_accelerate_grad():
    """https://github.com/PennyLaneAI/catalyst/issues/1086"""

    @qml.qjit
    @grad
    def hypothesis(x):
        return accelerate(accelerate(jnp.sin))(x)

    @jax.jit
    @jax.grad
    def ground_truth(x):
        return jax.jit(jax.jit(jnp.sin))(x)

    assert np.allclose(hypothesis(0.43), ground_truth(0.43))


if __name__ == "__main__":
    pytest.main(["-x", __file__])
