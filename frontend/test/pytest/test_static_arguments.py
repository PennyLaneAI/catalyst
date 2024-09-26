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

"""Test features related to static arguments."""

from dataclasses import dataclass

import pennylane as qml
import pytest
from jax.errors import TracerBoolConversionError

from catalyst import qjit
from catalyst.utils.exceptions import CompileError


class TestStaticArguments:
    """Test QJIT with static arguments."""

    def test_function_without_hints(self):
        """Test that a function without type hints works with static argnums (bug fix)."""

        @qjit(static_argnums=1)
        def f(x, y):
            return x + len(y)

        assert f(5, "hi") == 7

    def test_function_with_varargs(self):
        """Test that a function without a fixed number of arguments works with static argnums."""

        @qjit(static_argnums=1)
        def f(*args):
            return args[0] + len(args[1]) + args[2]

        assert f(1, "hi", 4) == 7

    @pytest.mark.parametrize("argnums", [(), None])
    def test_zero_static_argument(self, argnums):
        """Test QJIT with no static arguments."""

        @qjit(static_argnums=argnums)
        def f(x: int):
            return x

        assert f(1) == 1

    @pytest.mark.parametrize("argnums", [-2, 100])
    def test_out_of_bounds_static_argument(self, argnums):
        """Test QJIT with invalid static argument index with respect to provided arguments."""

        @qjit(static_argnums=argnums)
        def f(x):
            return x

        with pytest.raises(CompileError, match="is beyond the valid range"):
            f(5)

    @pytest.mark.parametrize("argnums", [1.0, [1.0], ["x"]])
    def test_unsopported_type_static_argument(self, argnums):
        """Test QJIT with invalid static argument type."""

        @qjit(static_argnums=argnums)
        def f(x, y):
            return x + y

        with pytest.raises(TypeError, match="The `static_argnums` argument to"):
            f(5, 6)

    def test_one_static_argument(self):
        """Test QJIT with one static argument."""

        @dataclass
        class MyClass:
            """Test class"""

            val: int

            def __hash__(self):
                return hash(str(self))

        @qjit(static_argnums=1)
        def f(x: int, y: MyClass):
            return x + y.val

        assert f(1, MyClass(5)) == 6
        function = f.compiled_function
        assert f(1, MyClass(7)) == 8
        assert function != f.compiled_function
        # Same static argument should not trigger re-compilation.
        assert f(2, MyClass(5)) == 7
        assert function == f.compiled_function

    def test_multiple_static_arguments(self):
        """Test QJIT with more than one static argument."""

        @dataclass
        class MyClass:
            """Test class"""

            val: int

            def __hash__(self):
                return hash(str(self))

        @qjit(static_argnums=(2, 0))
        def f(x: MyClass, y: int, z: MyClass):
            return x.val + y + z.val

        assert f(MyClass(5), 1, MyClass(5)) == 11
        function = f.compiled_function
        assert f(MyClass(7), 1, MyClass(7)) == 15
        assert function != f.compiled_function
        assert f(MyClass(5), 2, MyClass(5)) == 12
        assert function == f.compiled_function

    def test_mutable_static_arguments(self):
        """Test QJIT with mutable static arguments."""

        @dataclass
        class MyClass:
            """Test class"""

            val_0: int
            val_1: int

            def __hash__(self):
                return hash(str(self))

        @qjit(static_argnums=1)
        def f(x: int, y: MyClass):
            return x + y.val_0 + y.val_1

        my_obj = MyClass(5, 5)
        assert f(1, my_obj) == 11
        function = f.compiled_function
        # Changing mutable object should introduce re-compilation.
        my_obj.val_1 = 3
        assert f(1, my_obj) == 9
        assert function != f.compiled_function

    def test_qnode_with_static_arguments(self, capsys):
        """Test if QJIT static arguments pass through QNode correctly."""
        dev = qml.device("lightning.qubit", wires=1)

        @qjit(static_argnums=(1,))
        @qml.qnode(dev)
        def circuit(x, c):
            print("Inside QNode:", c)
            qml.RY(c, 0)
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        circuit(0.5, 0.5)
        captured = capsys.readouterr()
        assert captured.out.strip() == "Inside QNode: 0.5"

    def test_qnode_nested_with_static_arguments(self, capsys):
        """Test if QJIT static arguments pass through QNode correctly."""
        dev = qml.device("lightning.qubit", wires=1)

        @qjit(static_argnums=(1,))
        @qml.qnode(dev)
        def circuit(x, c):
            print("Inside QNode:", c)
            qml.RY(c, 0)
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        @qjit(static_argnums=(1,))
        def wrapper(x, c):
            return circuit(x, c)

        wrapper(0.5, 0.5)
        captured = capsys.readouterr()
        assert captured.out.strip() == "Inside QNode: 0.5"

    def test_qnode_switch_params(self, capsys):
        """Test if QJIT static arguments pass through QNode correctly when params are switched."""
        dev = qml.device("lightning.qubit", wires=1)

        @qjit(static_argnums=(0,))
        @qml.qnode(dev)
        def circuit(c, x):
            print("Inside QNode:", c)
            qml.RY(c, 0)
            qml.RX(x, 0)
            return qml.expval(qml.PauliZ(0))

        @qjit(static_argnums=(1,))
        def wrapper(x, c):
            return circuit(c, x)

        wrapper(0.5, 0.5)
        captured = capsys.readouterr()
        assert captured.out.strip() == "Inside QNode: 0.5"

    def test_qnode_nested_not_qnode(self, capsys):
        """Test if QJIT static arguments pass through nested Qjit calls with no QNodes."""
        dev = qml.device("lightning.qubit", wires=1)

        @qjit(static_argnums=(0,))
        def circuit(c, x):
            print("Inside QNode:", c)
            return x * c

        @qjit(static_argnums=(1,))
        def wrapper(x, c):
            return circuit(c, x)

        wrapper(0.5, 0.5)
        captured = capsys.readouterr()
        assert captured.out.strip() == "Inside QNode: 0.5"

    def test_static_argnames(self):
        # pylint: disable=inconsistent-return-statements
        """Test static arguments specified by names"""

        @qjit(static_argnames="y")
        def f(x, y):
            if y < 10:
                return x + y

        assert f(1, 2) == 3

        @qjit(static_argnames="x")
        def g(x, y):
            if y < 10:
                return x + y

        with pytest.raises(
            TracerBoolConversionError, match="Attempted boolean conversion of traced"
        ):
            g(1, 2)

        @qjit(static_argnames=("x", "y"))
        def h(x, y):
            if x < 10 and y < 10:
                return x + y

        assert h(1, 2) == 3

        @qjit(static_argnames=("x"), static_argnums=[1])
        def p(x, y):
            if x < 10 and y < 10:
                return x + y

        assert p(1, 2) == 3

        @qjit(static_argnames=("y"), static_argnums=[0])
        def q(x, y):
            if x < 10 and y < 10:
                return x + y

        assert q(1, 2) == 3

        @qjit(static_argnames=("y"), static_argnums=[1])
        def r(x, y):
            if y < 10:
                return x + y

        assert r(1, 2) == 3


if __name__ == "__main__":
    pytest.main(["-x", __file__])
