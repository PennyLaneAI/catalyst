# Copyright 2022-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi
from re import escape

import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from catalyst import qjit
from catalyst.api_extensions.control_flow import Switch, SwitchCallable, switch
from catalyst.utils.exceptions import PlxprCaptureCFCompatibilityError

RETURN_STRUCTURE_MESSAGE = "Control flow requires a consistent return structure across all branches"
SWITCH_DEFAULT_BRANCH_MESSAGE = "Switch requires a default branch."
MISSING_ARGUMENT_MESSAGE = "missing 1 required positional argument"
QUANTUM_OPERATION_MESSAGE = escape(
    "The switch() was not called (or has not been called) in a quantum"
    " context, and thus has no associated quantum operation."
)


class TestInterpreted:
    """Test that Catalyst switches can be used with the python interpreter."""

    def test_no_branches(self):
        """Test that an exception is raised when no branches are provided."""

        with pytest.raises(ValueError, match=SWITCH_DEFAULT_BRANCH_MESSAGE):
            assert SwitchCallable(0, None)

    def test_default_branch(self):
        """Test that a single branch is taken as default."""

        def circuit(i):
            @switch(i)
            def my_switch():
                return 42

            return my_switch()

        assert circuit(0) == 42
        assert circuit(1) == 42

    def test_1_branch(self):
        """Test that a branch catches only the correct case."""

        def circuit(i):
            @switch(i)
            def my_switch():
                return "default"

            @my_switch.branch(0)
            def my_branch():
                return "branch"

            return my_switch()

        assert circuit(-4) == "default"
        assert circuit(-1) == "default"
        assert circuit(0) == "branch"
        assert circuit(1) == "default"
        assert circuit(12) == "default"

    def test_branch_args(self):
        """Test that switch branches can be called with arguments."""

        def circuit(i, x):
            @switch(i)
            def my_switch(y):
                return 1

            @my_switch.branch(1)
            def my_branch(y):
                return 0

            @my_switch.branch(2)
            def my_branch_2(y):
                return y

            @my_switch.branch(0)
            def my_branch_3(y):
                return -y

            return my_switch(x)

        assert circuit(0, 1) == -1
        assert circuit(0, 1.3) == -1.3
        assert circuit(1, 1) == 0
        assert circuit(1, complex(1, 2)) == 0
        assert circuit(2, 1) == 1
        assert circuit(2, 19 / 2) == 19 / 2
        assert circuit(3, 15) == 1
        assert circuit(3, -4) == 1

    def test_non_sequential_indices(self):
        """Test that a switch can be created with non-sequential indices."""

        def circuit(i, x):
            @switch(i)
            def my_switch(y):
                return 0

            @my_switch.branch(11)
            def branch_11(y):
                return y**2

            @my_switch.branch(6)
            def branch_6(y):
                return 2 * y

            @my_switch.branch(9)
            def my_branch_9(y):
                return y

            return my_switch(x)

        assert circuit(9, 2) == 2
        assert circuit(11, 4) == 16
        assert circuit(6, 5) == 10
        assert circuit(-1, 2) == 0

    def test_missing_parameter(self):
        """Test that an exception is raised when parameters are missing."""

        def circuit(i):
            @switch()  # pylint: disable=no-value-for-parameter
            def my_switch():
                return 0

            @my_switch.branch(0)
            def my_branch():
                return 1

            return my_switch()

        with pytest.raises(TypeError, match=MISSING_ARGUMENT_MESSAGE):
            circuit(0)

        def circuit_3(i):
            @switch(i)
            def my_switch():
                return 0

            @my_switch.branch()  # pylint: disable=no-value-for-parameter
            def my_branch():
                return 1

            return my_switch()

        with pytest.raises(TypeError, match=MISSING_ARGUMENT_MESSAGE):
            circuit_3(0)

    @pytest.mark.usefixtures("use_capture")
    def test_fails_capture(self):
        """Test that a switch raises an exception with program capture enabled."""
        if not qml.capture.enabled():
            pytest.skip("capture only test")

        with pytest.raises(PlxprCaptureCFCompatibilityError) as exc_info:

            def circuit(i):
                @switch(i)
                def my_switch():
                    return 0

                @my_switch.branch(0)
                def my_branch():
                    return 1

                return my_switch()

            circuit(0)

        error_msg = str(exc_info.value)
        assert "not supported" in error_msg

    def test_missing_operation(self):
        """Test that operation access in an interpreted context raises an exception."""

        def circuit(i):
            @switch(i)
            def my_switch():
                return 0

            @my_switch.branch(1)
            def my_branch():
                return 1

            my_switch()

            with pytest.raises(AttributeError, match=QUANTUM_OPERATION_MESSAGE):
                assert isinstance(my_switch.operation, Switch)

            return my_switch()

        assert circuit(0) == 0


class TestClassicalCompiled:
    """Test classical compiled Catalyst switches."""

    def test_default_branch(self):
        """Test that a single branch is taken as default."""

        @qjit
        def circuit(i):
            @switch(i)
            def my_switch():
                return 12

            return my_switch()

        assert circuit(0) == 12
        assert circuit(3) == 12

    def test_1_branch(self):
        """Test that a branch catches only the correct case."""

        @qjit
        def circuit(i):
            @switch(i)
            def my_switch():
                return 1

            @my_switch.branch(1)
            def branch_1():
                return 0

            return my_switch()

        assert circuit(-4) == 1
        assert circuit(-1) == 1
        assert circuit(0) == 1
        assert circuit(1) == 0
        assert circuit(2) == 1
        assert circuit(8) == 1

    def test_branch_args(self):
        """Test that branches can accept arguments and keyword arguments."""

        def circuit(i, x, kw=None):
            @switch(i)
            def my_switch(y, kwarg=None):
                return y * kwarg

            @my_switch.branch(2)
            def my_branch(y, kwarg=None):
                return 2 * kwarg

            @my_switch.branch(4)
            def my_branch_2(y, kwarg=None):
                return y - kwarg

            return my_switch(x, kwarg=kw)

        assert circuit(0, 3, kw=12) == 36
        assert circuit(2, 9, kw=4) == 8
        assert circuit(4, 11, kw=4) == 7

    def test_non_sequential_cases(self):
        """Test that cases need not be sequential."""

        @qjit
        def circuit(i):
            @switch(i)
            def my_switch():
                return 0

            @my_switch.branch(3)
            def branch_3():
                return 3

            @my_switch.branch(-2)
            def branch_2():
                return -2

            @my_switch.branch(10)
            def my_branch_10():
                return 10

            return my_switch()

        assert circuit(-2) == -2
        assert circuit(-1) == 0
        assert circuit(0) == 0
        assert circuit(2) == 0
        assert circuit(3) == 3
        assert circuit(10) == 10

    def test_return_type_promotion(self):
        """Test that return types are correctly promoted when applicable."""

        @qjit
        def circuit(i):
            @switch(i)
            def my_switch():
                return 1

            @my_switch.branch(3)
            def my_branch():
                return 1.2

            @my_switch.branch(5)
            def my_branch_2():
                return complex(1, 2.2)

            return my_switch()

        res = circuit(0)
        assert res.dtype is jnp.dtype("complex128")  # pylint: disable=no-member
        assert res == 1

        res = circuit(3)
        assert res.dtype is jnp.dtype("complex128")  # pylint: disable=no-member
        assert res == 1.2

        res = circuit(5)
        assert res.dtype is jnp.dtype("complex128")  # pylint: disable=no-member
        assert res == complex(1, 2.2)

    def test_inconsistent_output_types(self):
        """Test that an exception is raised when incompatible return types are present."""

        @qjit
        def circuit(i):
            @switch(i)
            def my_switch():
                return [1, 2, 3]

            @my_switch.branch(1)
            def my_branch():
                return 0

            @my_switch.branch(2)
            def my_branch_2():
                return (9 / 4, 2)

            return my_switch()

        with pytest.raises(TypeError, match=RETURN_STRUCTURE_MESSAGE):
            circuit(0)

    def test_missing_parameter(self):
        """Test that an exception is raised when parameters are missing."""

        @qjit
        def circuit(i):
            @switch()  # pylint: disable=no-value-for-parameter
            def my_switch():
                return 0

            @my_switch.branch(0)
            def my_branch():
                return 2

            return my_switch()

        with pytest.raises(TypeError, match=MISSING_ARGUMENT_MESSAGE):
            circuit(0)

        @qjit
        def circuit_3(i):
            @switch(i)
            def my_switch():
                return 0

            @my_switch.branch()  # pylint: disable=no-value-for-parameter
            def my_branch():
                return 2

            return my_switch()

        with pytest.raises(TypeError, match=MISSING_ARGUMENT_MESSAGE):
            circuit_3(0)

    @pytest.mark.usefixtures("use_capture")
    def test_fails_capture(self, backend):
        """Test that an exception is raised when program capture is enabled."""
        if not qml.capture.enabled():
            pytest.skip("capture only test")

        with pytest.raises(PlxprCaptureCFCompatibilityError) as exc_info:

            @qjit
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(i):
                @switch(i)
                def my_switch():
                    qml.X(0)

                return my_switch()

            circuit(0)

        error_msg = str(exc_info.value)
        assert "not supported" in error_msg

    def test_missing_operation(self):
        """Test that operation access in classical context raises an exception."""

        @qjit
        def circuit(i):
            @switch(i)
            def my_switch():
                return 2

            @my_switch.branch(2)
            def my_branch():
                return 4

            my_switch()

            with pytest.raises(AttributeError, match=QUANTUM_OPERATION_MESSAGE):
                assert isinstance(my_switch.operation, Switch)

            return my_switch()

        assert circuit(0) == 2
        assert circuit(2) == 4


class TestQuantum:
    """Test compiled Catalyst switches with quantum operations."""

    def test_default_branch(self, backend):
        """Test that a single branch is taken as default."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(i):
            @switch(i)
            def my_switch():
                qml.X(0)

            my_switch()

            return qml.probs()

        assert np.allclose(circuit(0), [0, 1])
        assert np.allclose(circuit(1), [0, 1])

    def test_1_branch(self, backend):
        """Test that the default branch catches all unassigned cases."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(i):
            @switch(i)
            def my_switch():
                qml.RX(pi / 2, wires=0)

            @my_switch.branch(1)
            def branch_1():
                qml.RX(pi / 4, wires=0)

            @my_switch.branch(2)
            def branch_2():
                qml.RX(pi, wires=0)

            my_switch()

            return qml.probs(wires=0)

        assert np.allclose(circuit(0), [0.5, 0.5])
        assert np.allclose(circuit(1), [0.85355339, 0.14644661])
        assert np.allclose(circuit(2), [0, 1])

    def test_branch_args(self, backend):
        """Test that branches can accept arguments."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(i, angle, wire=None):
            @switch(i)
            def my_switch(angle, wire=None):
                qml.RX(angle, wires=wire)

            @my_switch.branch(2)
            def my_branch(angle, wire=None):
                qml.RY(angle, wires=wire)

            @my_switch.branch(0)
            def my_branch2(angle, wire=None):
                qml.RZ(angle, wires=wire)

            my_switch(angle, wire=wire)

            return qml.probs()

        assert np.allclose(circuit(0, pi, wire=0), [1, 0, 0, 0])
        assert np.allclose(circuit(2, pi / 4, wire=1), [0.85355339, 0.14644661, 0, 0])
        assert np.allclose(circuit(4, 3 * pi / 4, wire=1), [0.14644661, 0.85355339, 0, 0])

    def test_non_sequential_cases(self, backend):
        """Test that cases need not be sequential."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(i):
            @switch(i)
            def my_switch():
                qml.RX(0, wires=0)

            @my_switch.branch(3)
            def branch_3():
                qml.RX(pi / 3, wires=0)

            @my_switch.branch(-2)
            def branch_m2():
                qml.RX(pi / -2, wires=0)

            @my_switch.branch(0)
            def my_branch_0():
                qml.RX(pi, wires=0)

            my_switch()

            return qml.probs()

        assert np.allclose(circuit(-2), [0.5, 0.5])
        assert np.allclose(circuit(0), [0, 1])
        assert np.allclose(circuit(3), [0.75, 0.25])
        assert np.allclose(circuit(9), [1, 0])

    def test_return_type_promotion(self, backend):
        """Test that return types are correctly promoted when applicable."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(i):
            @switch(i)
            def my_switch():
                qml.X(0)
                return 1

            @my_switch.branch(3)
            def my_branch():
                qml.RX(pi / 2, wires=0)
                return 1.2

            @my_switch.branch(5)
            def my_branch_2():
                qml.RX(pi / 4, wires=0)
                return complex(1, 2.2)

            return [my_switch(), qml.probs()]

        res = circuit(0)
        assert res[0].dtype is jnp.dtype("complex128")  # pylint: disable=no-member
        assert res[0] == 1
        assert np.allclose(res[1], [0, 1])

        res = circuit(3)
        assert res[0].dtype is jnp.dtype("complex128")  # pylint: disable=no-member
        assert res[0] == 1.2
        assert np.allclose(res[1], [0.5, 0.5])

        res = circuit(5)
        assert res[0].dtype is jnp.dtype("complex128")  # pylint: disable=no-member
        assert res[0] == complex(1, 2.2)
        assert np.allclose(res[1], [0.85355339, 0.14644661])

    def test_inconsistent_output_types(self, backend):
        """Test that an exception is raised when incompatible return types are present."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(i):
            @switch(i)
            def my_switch():
                qml.X(0)
                return [9.1]

            @my_switch.branch(1)
            def my_branch():
                qml.Y(0)
                return 0

            @my_switch.branch(3)
            def my_branch_2():
                qml.Z(0)
                return (1, 2)

            return my_switch()

        with pytest.raises(TypeError, match=RETURN_STRUCTURE_MESSAGE):
            circuit(0)

    def test_missing_parameter(self, backend):
        """Test that an exception is raised when parameters are missing."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(i):
            @switch()  # pylint: disable=no-value-for-parameter
            def my_switch():
                qml.X(0)
                return 0

            @my_switch.branch(0)
            def my_branch():
                qml.Y(0)
                return 2

            return my_switch()

        with pytest.raises(TypeError, match=MISSING_ARGUMENT_MESSAGE):
            circuit(0)

        @qjit
        def circuit_2(i):
            @switch(i)
            def my_switch():
                qml.X(0)
                return 0

            @my_switch.branch()  # pylint: disable=no-value-for-parameter
            def my_branch():
                qml.Y(0)
                return 2

            return my_switch()

        with pytest.raises(TypeError, match=MISSING_ARGUMENT_MESSAGE):
            circuit_2(0)

    @pytest.mark.usefixtures("use_capture")
    def test_fails_capture(self, backend):
        """Test that an exception is raised when program capture is enabled."""
        if not qml.capture.enabled():
            pytest.skip("capture only test")

        with pytest.raises(PlxprCaptureCFCompatibilityError) as exc_info:

            @qjit
            @qml.qnode(qml.device(backend, wires=1))
            def circuit(i):
                @switch(i)
                def my_switch():
                    return 0

                return my_switch()

            circuit(0)

        error_msg = str(exc_info.value)
        assert "not supported" in error_msg

    def test_operation_access(self, backend):
        """Test that switch operations can be accessed in a quantum context."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(i):
            @switch(i)
            def my_switch():
                qml.X(0)

            my_switch()

            if not qml.capture.enabled():
                assert isinstance(my_switch.operation, Switch)

            return qml.probs()

        assert np.allclose(circuit(0), [0, 1])
        assert np.allclose(circuit(1), [0, 1])
