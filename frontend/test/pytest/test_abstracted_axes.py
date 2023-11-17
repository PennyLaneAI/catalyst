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

"""
Tests for abstracted axes
"""
from catalyst import qjit
import jax
import pennylane as qml
from numpy.testing import assert_allclose

class TestBasicInterface:
    """Test thas abstracted_axes kwarg does not change any functionality for the time being
    """

    def test_abstracted_axes_dictionary(self):
        """This is a temporary test while dynamism is in development."""

        @qjit(abstracted_axes={0: "n"})
        def identity(a):
            return a

        param = jax.numpy.array([1, 2, 3])
        result = identity(param)
        assert_allclose(param, result)
        assert "tensor<?xi64>" in identity.mlir

    def test_abstracted_axes_dictionary_2(self):
        """This is a temporary test while dynamism is in development."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def foo(a):
            return a

        @qjit(abstracted_axes={0: "n"})
        def identity(a):
            return foo(a)

        param = jax.numpy.array([1, 2, 3])
        result = identity(param)

        assert_allclose(param, result)

    def test_abstracted_axes_dictionary_3(self):
        """This is a temporary test while dynamism is in development."""

        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def foo(a, b):
            return a, b

        @qjit(abstracted_axes={0: "n"})
        def identity(a, b):
            return foo(a, b)

        param = jax.numpy.array([1, 2, 3])
        result = identity(param, param)

        assert_allclose((param, param), result)
