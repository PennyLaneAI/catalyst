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

import re

import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from catalyst import debug, for_loop, qjit


class TestDebugPrint:
    """Test suite for the runtime print functionality."""

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            (True, "1\n"),  # TODO: True/False would be nice
            (3, "3\n"),
            (3.5, "3.5\n"),
            (3 + 4j, "(3,4)\n"),
            (np.array(3), "3\n"),
            (jnp.array(3), "3\n"),
            (jnp.array(3.000001), "3\n"),  # TODO: show more precision
            (jnp.array(3.1 + 4j), "(3.1,4)\n"),
            (jnp.array([3]), "[3]\n"),
            (jnp.array([3, 4, 5]), "[3,  4,  5]\n"),
            (
                jnp.array([[3, 4], [5, 6], [7, 8]]),
                "[[3,   4], \n [5,   6], \n [7,   8]]\n",
            ),
        ],
    )
    def test_function_arguments(self, capfd, arg, expected):
        """Test printing of arbitrary JAX tracer values."""

        @qjit
        def test(x):
            debug.print(x)

        out, err = capfd.readouterr()
        assert err == ""
        assert out == ""

        test(arg)

        out, err = capfd.readouterr()
        assert err == ""
        assert expected == out

    def test_optional_descriptor(self, capfd):
        """Test the optional memref descriptor functionality."""

        @qjit
        def test(x):
            debug.print(x, memref=True)

        out, err = capfd.readouterr()
        assert err == ""
        assert out == ""

        test(jnp.array([[1, 2, 3], [4, 5, 6]]))

        memref = (
            r"MemRef: base\@ = [0-9a-fx]+ rank = 2 offset = 0 "
            r"sizes = \[2, 3\] strides = \[3, 1\] data ="
            "\n"
        ) + re.escape("[[1,   2,   3], \n [4,   5,   6]]\n")

        regex = re.compile("^" + memref + "$")  # match exactly: ^ - start, $ - end

        out, err = capfd.readouterr()
        assert err == ""
        assert regex.match(out)

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            (0, ""),
            (1, "0\n"),
            (6, "0\n1\n2\n3\n4\n5\n"),
        ],
    )
    def test_intermediate_values(self, capfd, arg, expected):
        """Test printing of arbitrary JAX tracer values."""

        @qjit
        def test(n):
            @for_loop(0, n, 1)
            def loop(i):
                debug.print(i)

            loop()

        out, err = capfd.readouterr()
        assert err == ""
        assert out == ""

        test(arg)

        out, err = capfd.readouterr()
        assert err == ""
        assert expected == out

    class MyObject:
        def __init__(self, string):
            self.string = string

        def __str__(self):
            return f"MyObject({self.string})"

    @pytest.mark.parametrize(
        ("arg", "expected"), [(3, "3\n"), ("hi", "hi\n"), (MyObject("hello"), "MyObject(hello)\n")]
    )
    def test_compile_time_values(self, capfd, arg, expected):
        """Test printing of arbitrary Python objects, including strings."""

        @qjit
        def test():
            debug.print(arg)

        out, err = capfd.readouterr()
        assert err == ""
        assert out == ""

        test()

        out, err = capfd.readouterr()
        assert err == ""
        assert out == expected

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            (True, "True\n"),
            (3, "3\n"),
            (3.5, "3.5\n"),
            (3 + 4j, "(3+4j)\n"),
            (np.array(3), "3\n"),
            (np.array([3]), "[3]\n"),
            (jnp.array(3), "3\n"),
            (jnp.array([3]), "[3]\n"),
            (jnp.array([[3, 4], [5, 6], [7, 8]]), "[[3 4]\n [5 6]\n [7 8]]\n"),
            ("hi", "hi\n"),
            (MyObject("hello"), "MyObject(hello)\n"),
        ],
    )
    def test_no_qjit(self, capfd, arg, expected):
        """Test printing in interpreted mode."""

        debug.print(arg)

        out, err = capfd.readouterr()
        assert err == ""
        assert out == expected

    def test_multiple_prints(self, capfd):
        "Test printing strings in multiple prints"

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def func1():
            debug.print("hello")
            return qml.state()

        @qjit
        def func2():
            func1()
            debug.print("goodbye")
            return

        func2()
        out, err = capfd.readouterr()
        assert err == ""
        assert out == "hello\ngoodbye\n"


if __name__ == "__main__":
    pytest.main(["-x", __file__])
