import jax.numpy as jnp
import numpy as np
import pytest

from catalyst import debug_print, for_loop, qjit


class TestDebugPrint:
    """Test suite for the runtime print functionality."""

    @pytest.mark.parametrize(
        ("arg", "expected"),
        [
            (True, "1\n"),  # TODO: is this ok?
            (3, "3\n"),
            (3.5, "3.5\n"),
            (3 + 4j, "3+4j\n"),
            (np.array(3), "3\n"),
            (jnp.array(3), "3\n"),
            (jnp.array(3.000001), "3\n"),  # TODO: fixme, show more precision
            (jnp.array(3.1 + 4j), "3.1+4j\n"),
            (jnp.array([3]), "[ 3 ]\n"),
            (jnp.array([3, 4, 5]), "[ 3 4 5 ]\n"),
            (
                jnp.array([[3, 4], [5, 6], [7, 8]]),
                "[ [ 3 4 ], [ 5 6 ], [ 7 8 ] ]\n",
            ),  # TODO: fixme, print n-d arrays over multiple lines, with proper indents
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
        assert out == expected

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
                debug_print(i)

            loop()

        out, err = capfd.readouterr()
        assert err == ""
        assert out == ""

        test(arg)

        out, err = capfd.readouterr()
        assert err == ""
        assert out == expected

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
