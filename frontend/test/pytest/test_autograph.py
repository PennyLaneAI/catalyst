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

"""PyTests for the AutoGraph source-to-source transformation feature."""

import os

import numpy as np
import pennylane as qml
import pytest

from catalyst import measure, qjit
from catalyst.ag_utils import AutoGraphError, check_cache, converted_code

# pylint: disable=missing-function-docstring


class TestIntegration:
    """Test that the autograph transformations trigger correctly in different settings."""

    def test_unavailable(self, monkeypatch):
        """Check the error produced in the absence of tensorflow."""
        monkeypatch.syspath_prepend(os.path.join(os.path.dirname(__file__), "mock"))

        def fn(x):
            return x**2

        with pytest.raises(ImportError, match="AutoGraph feature in Catalyst requires TensorFlow"):
            qjit(autograph=True)(fn)

    def test_classical_function(self):
        """Test autograph on a purely classical function."""

        @qjit(autograph=True)
        def fn(x):
            return x**2

        assert hasattr(fn.user_function, "ag_unconverted")
        assert fn(4) == 16

    def test_nested_function(self):
        """Test autograph on nested classical functions."""

        def inner(x):
            return x**2

        @qjit(autograph=True)
        def fn(x: int):
            return inner(x)

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(inner)
        assert fn(4) == 16

    def test_qnode(self):
        """Test autograph on a QNode."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def fn(x: float):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert hasattr(fn.user_function, "ag_unconverted")
        assert fn(np.pi) == -1

    def test_indirect_qnode(self):
        """Test autograph on a QNode called from within a classical function."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner(x)

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(inner.func)
        assert fn(np.pi) == -1

    def test_multiple_qnode(self):
        """Test autograph on multiple QNodes called from different classical functions."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner1(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner2(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner1(x) + inner2(x)

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(inner1.func)
        assert check_cache(inner2.func)
        assert fn(np.pi) == -2

    def test_nested_qnode(self):
        """Test autograph on a QNode called from within another QNode."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner1(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner2(x):
            y = inner1(x) * np.pi
            qml.RY(y, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: int):
            return inner2(x)

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(inner1.func)
        assert check_cache(inner2.func)
        # Unsupported by the runtime:
        # assert fn(np.pi) == -2

    def test_nested_qjit(self):
        """Test autograph on a QJIT function called from within the compilation entry point."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner(x)

        assert hasattr(fn.user_function, "ag_unconverted")
        assert check_cache(inner.user_function.func)
        assert fn(np.pi) == -1


class TestCodePrinting:
    """Test that the transformed source code can be printed in different settings."""

    def test_unconverted(self):
        """Test printing on an unconverted function."""

        @qjit(autograph=False)
        def fn(x):
            return x**2

        with pytest.raises(AutoGraphError, match="function was not converted by AutoGraph"):
            converted_code(fn)

    def test_classical_function(self):
        """Test printing on a purely classical function."""

        @qjit(autograph=True)
        def fn(x):
            return x**2

        assert converted_code(fn)

    def test_nested_function(self):
        """Test printing on nested classical functions."""

        def inner(x):
            return x**2

        @qjit(autograph=True)
        def fn(x: int):
            return inner(x)

        assert converted_code(fn)
        assert converted_code(inner)

    def test_qnode(self):
        """Test printing on a QNode."""

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def fn(x: float):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        assert converted_code(fn)

    def test_indirect_qnode(self):
        """Test printing on a QNode called from within a classical function."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner(x)

        assert converted_code(fn)
        assert converted_code(inner)

    def test_multiple_qnode(self):
        """Test printing on multiple QNodes called from different classical functions."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner1(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner2(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner1(x) + inner2(x)

        assert converted_code(fn)
        assert converted_code(inner1)
        assert converted_code(inner2)

    def test_nested_qnode(self):
        """Test printing on a QNode called from within another QNode."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner1(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner2(x):
            y = inner1(x) * np.pi
            qml.RY(y, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: int):
            return inner2(x)

        assert converted_code(fn)
        assert converted_code(inner1)
        assert converted_code(inner2)

    def test_nested_qjit(self):
        """Test printing on a QJIT function called from within the compilation entry point."""

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def inner(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(autograph=True)
        def fn(x: float):
            return inner(x)

        assert converted_code(fn)
        assert converted_code(inner)


class TestConditionals:
    """Test that the autograph transformations produce correct results on conditionals.
    These tests are adapted from the test_conditionals.TestCond class of tests."""

    def test_simple_cond(self):
        """Test basic function with conditional."""

        @qjit(autograph=True)
        def circuit(n):
            if n > 4:
                res = n**2
            else:
                res = n

            return res

        assert circuit(0) == 0
        assert circuit(1) == 1
        assert circuit(2) == 2
        assert circuit(3) == 3
        assert circuit(4) == 4
        assert circuit(5) == 25
        assert circuit(6) == 36

    def test_cond_one_else_if(self):
        """Test a cond with one else_if branch"""

        @qjit(autograph=True)
        def circuit(x):
            if x > 2.7:
                res = x * 4
            elif x > 1.4:
                res = x * 2
            else:
                res = x

            return res

        assert circuit(4) == 16
        assert circuit(2) == 4
        assert circuit(1) == 1

    def test_cond_many_else_if(self):
        """Test a cond with multiple else_if branches"""

        @qjit(autograph=True)
        def circuit(x):
            if x > 4.8:
                res = x * 8
            elif x > 2.7:
                res = x * 4
            elif x > 1.4:
                res = x * 2
            else:
                res = x

            return res

        assert circuit(5) == 40
        assert circuit(3) == 12
        assert circuit(2) == 4
        assert circuit(-3) == -3

    def test_qubit_manipulation_cond(self, backend):
        """Test conditional with quantum operation."""

        @qjit(autograph=True)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x):
            if x > 4:
                qml.PauliX(wires=0)

            return measure(wires=0)

        assert circuit(3) == False
        assert circuit(6) == True

    def test_branch_return_mismatch(self, backend):
        """Test that an exception is raised when the true branch returns a value without an else
        branch.
        """

        def circuit():
            if True:
                res = measure(wires=0)

            return res

        with pytest.raises(
            AutoGraphError, match="Some branches did not define a value for variable 'res'"
        ):
            qjit(autograph=True)(qml.qnode(qml.device(backend, wires=1))(circuit))

    def test_branch_multi_return_mismatch(self, backend):
        """Test that an exception is raised when the return types of all branches do not match."""

        def circuit():
            if True:
                res = measure(wires=0)
            elif False:
                res = 0
            else:
                res = measure(wires=0)

            return res

        with pytest.raises(
            TypeError, match="Conditional requires consistent return types across all branches"
        ):
            qjit(autograph=True)(qml.qnode(qml.device(backend, wires=1))(circuit))


if __name__ == "__main__":
    pytest.main(["-x", __file__])
