import jax
import pennylane as qml
import pytest

from catalyst import api_extensions, qjit, switch
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

    def test_decorator(self):
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

        assert foo(1, 1) == 1
