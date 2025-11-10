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

import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

import catalyst
from catalyst import qjit, switch


class TestInterpreted:
    """Test that Catalyst switches can be used with the python interpreter."""

    def test_1_branch(self):
        """
        Test that a switch can be used with a single branch, which is used as the default branch.
        """

        def circuit(i):
            @switch(i)
            def my_switch():
                return 42

            return my_switch()

        assert circuit(0) == 42
        assert circuit(1) == 42

    def test_default_branch(self):
        """Test that manually assigning a default branch catches all unassigned cases."""

        def circuit(i):
            @switch(i)
            def my_switch():
                return "first"

            @my_switch.default()
            def my_default():
                return "default"

            return my_switch()

        assert circuit(-4) == "default"
        assert circuit(-1) == "default"
        assert circuit(0) == "first"
        assert circuit(1) == "default"
        assert circuit(12) == "default"

    def test_branch_args(self):
        """Test that switch branches can be called with arguments."""

        def circuit(i, x):
            @switch(i)
            def my_switch(x):
                return -x

            @my_switch.branch(1)
            def my_branch(x):
                return 0

            @my_switch.branch(2)
            def my_branch_2(x):
                return x

            return my_switch(x)

        assert circuit(0, 1) == -1
        assert circuit(0, 1.3) == -1.3
        assert circuit(1, 1) == 0
        assert circuit(1, complex(1, 2)) == 0
        assert circuit(2, 1) == 1
        assert circuit(2, 19 / 2) == 19 / 2

    def test_chosen_index(self):
        """Test that the initial branch of a switch can be assigned a case."""

        def circuit(i):
            @switch(i, case=12)
            def my_switch():
                return "main"

            @my_switch.default()
            def my_default():
                return "default"

            return my_switch()

        assert circuit(12) == "main"
        assert circuit(0) == "default"

    def test_non_sequential_indices(self):
        """Test that a switch can be created with non-sequential indices."""

        def circuit(i, x):
            @switch(i, case=3)
            def my_switch(x):
                return x

            @my_switch.branch(11)
            def branch_11(x):
                return x**2

            @my_switch.branch(6)
            def branch_6(x):
                return 2 * x

            return my_switch(x)

        assert circuit(3, 2) == 2
        assert circuit(11, 4) == 16
        assert circuit(6, 5) == 10
        assert circuit(-1, 2) == 2

    def test_no_case_parameter(self):
        """Test that a switch raises an error when called without the case argument."""

        def circuit(i):
            @switch()
            def my_switch():
                return 0

            return my_switch()

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            circuit(0)

        def bar(i):
            @switch(i)
            def my_switch():
                return 0

            @my_switch.branch()
            def my_branch():
                return 1

            return my_switch()

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            bar(0)


class TestClassicalCompiled:
    """Test classical compiled Catalyst switches."""

    def test_1_branch(self):
        """
        Test that a switch can be used with a single branch, which is used as the default branch.
        """

        @qjit
        def circuit(i):
            @switch(i)
            def my_switch():
                return 12

            return my_switch()

        assert circuit(0) == 12
        assert circuit(3) == 12

    def test_default_branch(self):
        """Test that the default branch catches all unassigned cases."""

        @qjit
        def circuit(i):
            @switch(i)
            def my_switch():
                return -1

            @my_switch.branch(1)
            def branch_1():
                return 1

            @my_switch.default()
            def branch_2():
                return 3

            return my_switch()

        assert circuit(-4) == 3
        assert circuit(-1) == 3
        assert circuit(0) == -1
        assert circuit(1) == 1
        assert circuit(2) == 3
        assert circuit(8) == 3

    def test_branch_args(self):
        def circuit(i, x, kw=None):
            @switch(i)
            def my_switch(x, kw=None):
                return x * kw

            @my_switch.branch(2)
            def my_branch(x, kw=None):
                return 2 * kw

            @my_switch.default()
            def my_default(x, kw=None):
                return x - kw

            return my_switch(x, kw=kw)

        assert circuit(0, 3, kw=12) == 36
        assert circuit(2, 9, kw=4) == 8
        assert circuit(4, 11, kw=4) == 7

    def test_chosen_initial_index(self):
        @qjit
        def circuit(i, x):
            @switch(i, case=1)
            def my_switch(x):
                return 3 * x + 1

            @my_switch.default()
            def branch_1(x):
                return x // 2

            return my_switch(x)

        assert circuit(1, 1) == 4
        assert circuit(1, 3) == 10
        assert circuit(1, 5) == 16
        assert circuit(1, 9) == 28
        assert circuit(0, 6) == 3
        assert circuit(0, 8) == 4

    def test_non_sequential_indices(self):
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

            @my_switch.default()
            def default():
                return 10

            return my_switch()

        assert circuit(-2) == -2
        assert circuit(-1) == 10
        assert circuit(0) == 0
        assert circuit(1) == 10
        assert circuit(2) == 10
        assert circuit(3) == 3

    def test_return_type_promotion(self):
        @qjit
        def circuit(i):
            @switch(i)
            def my_switch():
                return 1

            @my_switch.branch(3)
            def my_branch():
                return 1.2

            @my_switch.default()
            def my_default():
                return complex(1, 2.2)

            return my_switch()

        assert circuit(0).dtype is jnp.dtype("complex128")

    def test_inconsistent_output_types(self):
        @qjit
        def circuit(i):
            @switch(i)
            def my_switch():
                return "hello"

            @my_switch.branch(1)
            def my_branch():
                return 0

            @my_switch.default
            def my_default():
                return ("a", 2)

            return my_switch()

        with pytest.raises(TypeError):
            circuit(0)

    def test_missing_parameter(self):
        @qjit
        def circuit(i):
            @switch()
            def my_switch():
                return 0

            @my_switch.default
            def my_branch():
                return 2

            return my_switch()

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            circuit(0)

        @qjit
        def bar(i):
            @switch(1)
            def my_switch():
                return 0

            @my_switch.branch()
            def my_branch():
                return 2

            return my_switch()

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            bar(0)


class TestQuantum:
    """Test compiled Catalyst switches with quantum operations."""

    def test_1_branch(self, backend):
        """
        Test that a switch can be used with a single branch, which is used as the default branch.
        """

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(i):
            @switch(i)
            def my_switch():
                qml.X(0)

            my_switch()

            return catalyst.measure(wires=0)

        assert circuit(0)
        assert circuit(3)

    def test_default_branch(self, backend):
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

            @my_switch.default()
            def branch_2():
                qml.RX(pi, wires=0)

            my_switch()

            return qml.probs(wires=0)

        assert np.allclose(circuit(0), [0.5, 0.5])
        assert np.allclose(circuit(1), [0.85355339, 0.14644661])
        assert np.allclose(circuit(2), [0, 1])

    def test_branch_args(self, backend):
        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(i, angle, wire=None):
            @switch(i)
            def my_switch(angle, wire=None):
                qml.RX(angle, wires=wire)

            @my_switch.branch(2)
            def my_branch(angle, wire=None):
                qml.RY(angle, wires=wire)

            @my_switch.default()
            def my_default(angle, wire=None):
                qml.RZ(angle, wires=wire)

            my_switch(angle, wire=wire)

            return qml.probs()

        assert np.allclose(circuit(0, pi, wire=0), [0, 0, 1, 0])
        assert np.allclose(circuit(2, pi / 4, wire=1), [0.85355339, 0.14644661, 0, 0])
        assert np.allclose(circuit(4, 3 * pi / 4, wire=1), [1, 0, 0, 0])

    def test_chosen_initial_index(self, backend):
        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(i):
            @switch(i, case=3)
            def my_switch():
                qml.H(0)

            @my_switch.default()
            def branch_1():
                qml.X(0)

            my_switch()

            return qml.probs()

        assert np.allclose(circuit(0), [0, 1])
        assert np.allclose(circuit(3), [0.5, 0.5])

    def test_non_sequential_indices(self, backend):
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
            def branch_2():
                qml.RX(pi / -2, wires=0)

            @my_switch.default()
            def default():
                qml.RX(pi, wires=0)

            my_switch()

            return qml.probs()

        assert np.allclose(circuit(-2), [0.5, 0.5])
        assert np.allclose(circuit(0), [1, 0])
        assert np.allclose(circuit(3), [0.75, 0.25])
        assert np.allclose(circuit(9), [0, 1])

    def test_return_type_promotion(self, backend):
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

            @my_switch.default()
            def my_default():
                qml.RX(pi / 4, wires=0)
                return complex(1, 2.2)

            return [my_switch(), qml.probs()]

        res = circuit(0)
        assert res[0] == 1 and np.allclose(res[1], [0, 1])

        res = circuit(3)
        assert res[0] == 1.2 and np.allclose(res[1], [0.5, 0.5])

        res = circuit(1)
        assert res[0] == complex(1, 2.2) and np.allclose(res[1], [0.85355339, 0.14644661])

    def test_inconsistent_output_types(self, backend):
        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(i):
            @switch(i)
            def my_switch():
                qml.X(0)
                return "hello"

            @my_switch.branch(1)
            def my_branch():
                qml.Y(0)
                return 0

            @my_switch.default
            def my_default():
                qml.Z(0)
                return ("a", 2)

            return my_switch()

        with pytest.raises(TypeError):
            circuit(0)

    def test_missing_parameter(self, backend):
        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(i):
            @switch()
            def my_switch():
                qml.X(0)
                return 0

            @my_switch.default
            def my_branch():
                qml.Y(0)
                return 2

            return my_switch()

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            circuit(0)

        @qjit
        def bar(i):
            @switch(1)
            def my_switch():
                qml.X(0)
                return 0

            @my_switch.branch()
            def my_branch():
                qml.Y(0)
                return 2

            return my_switch()

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            bar(0)

    def test_fails_capture(self, backend):
        if not qml.capture.enabled():
            pytest.skip("capture only test")

        qml.capture.enable()

        with pytest.raises(TypeError) as exc_info:

            @switch(0)
            def my_switch():
                return 0

        error_msg = str(exc_info.value)
        assert "not supported" in error_msg
