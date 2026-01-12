# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit test module for the tree traversal transform"""

import jax.numpy as jnp
import pennylane as qml
import pytest

from catalyst.python_interface.transforms import (
    TreeTraversalPass,
    tree_traversal_pass,
)

pytestmark = pytest.mark.xdsl


class TestTreeTraversalPass:
    """Unit tests for TreeTraversalPass."""

    def test_pass_preserves_function(self, run_filecheck):
        """Test that the pass preserves the function."""
        program = """
            func.func @test_func() {
                // CHECK: func.func @test_func()
                %0 = "test.op"() : () -> !quantum.bit
                %1 = quantum.custom "PauliX"() %0 : !quantum.bit
                %2, %3 = quantum.measure %1 : i1, !quantum.bit
                %4 = quantum.custom "PauliY"() %3 : !quantum.bit
                %5, %6 = quantum.measure %4 : i1, !quantum.bit
                return
            }
        """

        pipeline = (TreeTraversalPass(),)
        run_filecheck(program, pipeline)


@pytest.mark.usefixtures("use_capture")
class TestTreeTraversalIntegration:
    """Integration tests for the TreeTraversalPass."""

    def test_result_same_as_without_pass(self):
        """Test that the result is the same with and without the pass."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit
        @tree_traversal_pass
        @qml.qnode(dev)
        def circuit_with_pass(x):
            qml.RX(x, wires=0)
            qml.measure(wires=0)
            qml.RY(x, wires=0)
            qml.measure(wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qjit
        @qml.qnode(dev)
        def circuit_without_pass(x):
            qml.RX(x, wires=0)
            qml.measure(wires=0)
            qml.RY(x, wires=0)
            qml.measure(wires=0)
            return qml.expval(qml.PauliZ(0))

        x = 0.5
        result_with_pass = circuit_with_pass(x)
        result_without_pass = circuit_without_pass(x)

        assert jnp.allclose(result_with_pass, result_without_pass)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
