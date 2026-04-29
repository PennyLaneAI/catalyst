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
"""Unit test module for the utilities in xdsl_conversion.py"""

import pennylane as qp
import pytest
from jaxlib.mlir._mlir_libs._mlir.ir import Module

from catalyst.python_interface.inspection.xdsl_conversion import get_mlir_module

pytestmark = pytest.mark.xdsl


class TestGetMLIRModule:
    """Tests the get_mlir_module helper function."""

    def test_standard_circuit(self):
        """Tests a standard circuit."""
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        @qp.qnode(dev)
        def my_workflow():
            qp.X(0)
            return qp.expval(qp.Z(0))

        module = get_mlir_module(my_workflow, (), {})
        assert isinstance(module, Module)

    def test_standard_circuit_with_args_kwargs(self):
        """Tests a standard circuit with args and kwargs."""
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        @qp.qnode(dev)
        def my_workflow(angle, wires=None):
            qp.RX(angle, wires)
            return qp.expval(qp.Z(0))

        module = get_mlir_module(my_workflow, (3.14,), {"wires": [0]})
        assert isinstance(module, Module)

    def test_circuit_with_no_return(self):
        """Tests a standard circuit with no return."""
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit
        @qp.qnode(dev)
        def my_workflow(wire):
            qp.X(wire)

        module = get_mlir_module(my_workflow, (1,), {})
        assert isinstance(module, Module)

    def test_compile_options_not_mutated(self):
        """Ensures that the QJIT'd qnode's compile options are not mutable."""
        dev = qp.device("lightning.qubit", wires=1)

        @qp.qjit(autograph=True)
        @qp.qnode(dev)
        def my_workflow(angle, wires=None):
            qp.RX(angle, wires)
            return qp.expval(qp.Z(0))

        assert my_workflow.compile_options.autograph is True

        _ = get_mlir_module(my_workflow, (3.14,), {"wires": [0]})

        assert my_workflow.compile_options.autograph is True
