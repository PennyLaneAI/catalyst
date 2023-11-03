import re

import jax.numpy as jnp
import numpy as np
import pytest

from catalyst import debug_print, for_loop, qjit


class TestDebugPrint:
    """Test suite for the runtime print functionality."""

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            (True, "[1]"),  # TODO: True/False would be nice
            (3, "[3]"),  # TODO: scalars without brackets would be nice
            (3.5, "[3.5]"),
            (3 + 4j, "[(3,4)]"),
            (np.array(3), "[3]"),
            (jnp.array(3), "[3]"),
            (jnp.array(3.000001), "[3]"),  # TODO: show more precision
            (jnp.array(3.1 + 4j), "[(3.1,4)]"),
            (jnp.array([3]), "[3]"),
            (jnp.array([3, 4, 5]), "[3,  4,  5]"),
            (
                jnp.array([[3, 4], [5, 6], [7, 8]]),
                "[[3,   4], \n [5,   6], \n [7,   8]]",
            ),
        ],
    )
    def test_function_arguments(self, capfd, arg, expected):
        """Test printing of arbitrary JAX tracer values."""

        @qjit
        def test(x):
            debug_print(x)

        out, err = capfd.readouterr()
        assert err == ""
        assert out == ""

        test(arg)

        out, err = capfd.readouterr()
        assert err == ""
        assert expected in out

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            (0, ()),
            (1, ("[0]",)),
            (6, ("[0]", "[1]", "[2]", "[3]", "[4]", "[5]")),
        ],
    )
    def test_intermediate_values(self, capfd, arg, expected):
        """Test printing of arbitrary JAX tracer values."""

        @qjit
        def test(n):
            @for_loop(0, n, 1)
            def loop(i):
                debug_print(i)

            loop()

        out, err = capfd.readouterr()
        assert err == ""
        assert out == ""

        test(arg)

        memref_descriptor = (
            r"Unranked Memref base\@ = [0-9a-fx]+ rank = 0 "
            r"offset = 0 sizes = \[\] strides = \[\] data = "
            "\n"
        )

        expected_full = "".join(memref_descriptor + re.escape(exp) + "\n" for exp in expected)
        regex = re.compile("^" + expected_full + "$")  # match exactly: ^ - start, $ - end

        out, err = capfd.readouterr()
        assert err == ""
        assert regex.match(out)

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
            debug_print(arg)

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

        debug_print(arg)

        out, err = capfd.readouterr()
        assert err == ""
        assert out == expected


if __name__ == "__main__":
    pytest.main(["-x", __file__])
