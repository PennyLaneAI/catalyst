import jax
import pennylane as qml
import pytest

from catalyst import api_extensions, qjit, switch, measure
from catalyst.api_extensions.control_flow import SwitchCallable


class TestInterpreted:
    """Test that Catalyst switches can be used with the python interpreter."""

    @pytest.mark.parametrize("i", [-4, -1, 0, 1, 2, 3, 19])
    def test_default(self, i):
        branches = [lambda: 0, lambda: 1, lambda: 2]
        default_branch = lambda: "default"

        assert SwitchCallable(
            i, range(len(branches)), branches, default_branch=default_branch
        )() == (branches[i]() if 0 <= i < len(branches) else "default")

    @pytest.mark.parametrize("x", [12, 9.3, complex(4, 3)])
    def test_1_branch(self, x):
        branches = [lambda x: 2 * x]

        assert SwitchCallable(0, [0], branches)(x) == 2 * x

    @pytest.mark.parametrize(
        "i,x", [(0, 1), (0, 1.3), (1, 1), (1, complex(1, 2)), (2, 1), (2, 19 / 2)]
    )
    def test_3_branch(self, i, x):
        branches = [lambda x: -x, lambda x: 0, lambda x: x]

        assert SwitchCallable(i, range(3), branches)(x) == branches[i](x)

    @pytest.mark.parametrize(
        "i,x",
        [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 5), (5, 6)],
    )
    def test_decorator(self, i, x):
        def foo(i, x):
            @switch(i)
            def new_switch(x):
                return 0

            @new_switch.branch(1)
            def branch_1(x):
                return x

            @new_switch.branch(2)
            def branch_2(x):
                return 2 * x

            @new_switch.default()
            def default_branch(x):
                return -1

            return new_switch(x)

        assert foo(i, x) == (i * x if i in [0, 1, 2] else -1)


class TestClassicalCompiled:
    """Test compiled Catalyst switches."""

    @pytest.mark.parametrize("i", [-1, -4, 0, 1, 2, 19])
    def test_default(self, i):
        @qjit
        def foo(j):
            @switch(j)
            def my_switch():
                return -1

            @my_switch.branch(1)
            def branch_1():
                return 1

            @my_switch.default()
            def branch_2():
                return 3

            return my_switch()

        assert foo(i) == (i * 2 - 1 if i in [0, 1] else 3)

    @pytest.mark.parametrize("x", [1, 3, 5, 6, 8, 9, 11, 12])
    def test_chosen_index(self, x):
        @qjit
        def foo(j, y):
            @switch(j)
            def my_switch(y, case=1):
                return 3 * y + 1

            @my_switch.default()
            def branch_1(y):
                return y // 2

            return my_switch(y)

        def collatz(x):
            if x % 2:
                return x // 2
            return 3 * x + 1

        assert foo(x % 2, x) == collatz(x)

    @pytest.mark.parametrize("i", [-3, -2, -1, 0, 1, 2, 3, 4, 5, 8, 10, 12])
    def test_non_sequential_indices(self, i):
        @qjit
        def foo(j):
            @switch(j)
            def my_switch():
                return 0

            @my_switch.branch(3)
            def branch_3():
                return 3

            @my_switch.branch(-2)
            def branch_m2():
                return -2

            @my_switch.default()
            def default():
                return 10

            return my_switch()

        assert foo(i) == (i if i in [0, 3, -2] else 10)


class TestQuantum:
    """Test compiled Catalyst switches with quantum operations."""

    @pytest.mark.parametrize("i", [0, 1])
    def test_x_gate(self, i):
        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def foo(j):
            @switch(j)
            def my_switch():
                pass

            @my_switch.default()
            def branch():
                qml.X(0)

            my_switch()
            return measure(wires=0)

        assert foo(i) == bool(i)


# TODO test for exceptions
# TODO ensure test suite is comprehensive but not redundant
