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

import numpy as np
import pennylane as qml
import pytest

import catalyst
from catalyst import qjit
from catalyst.ag_primitives import STD
from catalyst.autograph import AutoGraphError, converted_code, _TRANSFORMER as AGT

# pylint: disable=missing-function-docstring


class TestIntegration:
    """Test that the autograph transformations trigger correctly in different settings."""

    def test_unavailable(self, monkeypatch):
        """Check the warning produced in the absence of tensorflow."""
        monkeypatch.setattr(catalyst.compilation_pipelines, "AG_AVAILABLE", False, raising=True)

        def fn(x):
            return x**2

        with pytest.warns(UserWarning, match="autograph feature in Catalyst requires TensorFlow"):
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
        assert AGT.has_cache(inner, STD)
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
        assert AGT.has_cache(inner.func, STD)
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
        assert AGT.has_cache(inner1.func, STD)
        assert AGT.has_cache(inner2.func, STD)
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
        assert AGT.has_cache(inner1.func, STD)
        assert AGT.has_cache(inner2.func, STD)
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
        assert AGT.has_cache(inner.user_function.func, STD)
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

if __name__ == "__main__":
    pytest.main(["-x", __file__])
