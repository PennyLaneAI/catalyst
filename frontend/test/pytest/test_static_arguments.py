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

import random
import warnings
from timeit import default_timer as timer
from dataclasses import dataclass

import jax
import numpy as np
import pennylane as qml
import pytest
from jax import numpy as jnp
from numpy import pi

from catalyst import for_loop, grad, measure, qjit

class TestStaticArguments:
    """Test QJIT with static arguments."""

    def test_zero_static_argument(self):
        """Test QJIT with zero static argument."""

        @qjit
        def f0(
            x: int,
        ):
            return x

        @qjit(static_argnums=())
        def f1(
            x: int,
        ):
            return x

        @qjit(static_argnums=None)
        def f2(
            x: int,
        ):
            return x

        assert f0(1) == 1
        assert f1(2) == 2
        assert f2(3) == 3

    def test_out_of_bounds_static_argument(self):
        """Test QJIT with invalid static argument index."""

        @qjit(static_argnums=100)
        def f(
            x: int,
        ):
            return x

        assert f(1) == 1

    def test_one_static_argument(self):
        """Test QJIT with one static argument."""

        @dataclass
        class MyClass:
            """Test class"""
            val: int

            def __hash__(self):
                return hash(str(self))

        @qjit(static_argnums=1)
        def f(
            x: int,
            y: MyClass,
        ):
            return x + y.val

        assert f(1, MyClass(5)) == 6
        function = f.compiled_function
        assert f(1, MyClass(7)) == 8
        assert function != f.compiled_function
        # Same static argument should not trigger re-compilation.
        assert f(2, MyClass(5)) == 7
        assert function == f.compiled_function

    def test_multiple_static_arguments(self):
        """Test QJIT with more than one static arguments."""

        @dataclass
        class MyClass:
            """Test class"""
            val: int

            def __hash__(self):
                return hash(str(self))

        @qjit(static_argnums=(2, 0))
        def f(
            x: MyClass,
            y: int,
            z: MyClass
        ):
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
            val0: int
            val1: int

            def __hash__(self):
                return hash(str(self))

        @qjit(static_argnums=1)
        def f(
            x: int,
            y: MyClass,
        ):
            return x + y.val0 + y.val1

        myObj = MyClass(5, 5)
        assert f(1, myObj) == 11
        function = f.compiled_function
        # Changing mutable object should introduce re-compilation.
        myObj.val1 = 3
        assert f(1, myObj) == 9
        assert function != f.compiled_function



if __name__ == "__main__":
    pytest.main(["-x", __file__])