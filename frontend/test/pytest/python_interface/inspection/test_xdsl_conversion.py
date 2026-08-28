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

from catalyst.python_interface.conversion import parse_generic_to_xdsl_module
from catalyst.python_interface.dialects.quantum import CtrlOp
from catalyst.python_interface.inspection.xdsl_conversion import (
    get_mlir_module,
    resolve_constant_wire,
)

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


class TestResolveConstantWire:
    """Tests the resolve_constant_wire helper function."""

    def test_ctrl_op_result_threading(self):
        """A wire resolved through a `quantum.ctrl` result should thread back to the matching input
        operand: an ``out_ctrl_qubits`` result to its ``in_ctrl_qubits`` operand, and a target
        ``results`` value to its ``args`` operand.
        """
        program = """
        "builtin.module"() ({
          "func.func"() ({
          ^bb0(%qreg: !quantum.reg):
            %c0 = "arith.constant"() <{value = 0 : i64}> : () -> i64
            %c1 = "arith.constant"() <{value = 1 : i64}> : () -> i64
            %c2 = "arith.constant"() <{value = 2 : i64}> : () -> i64
            %true = "arith.constant"() <{value = true}> : () -> i1
            %q0 = "quantum.extract"(%qreg, %c0) : (!quantum.reg, i64) -> !quantum.bit
            %q1 = "quantum.extract"(%qreg, %c1) : (!quantum.reg, i64) -> !quantum.bit
            %q2 = "quantum.extract"(%qreg, %c2) : (!quantum.reg, i64) -> !quantum.bit
            %oc:2, %ot = "quantum.ctrl"(%q1, %q2, %true, %true, %q0) <{
                operandSegmentSizes = array<i32: 2, 2, 1>,
                resultSegmentSizes = array<i32: 2, 1>}> ({
            ^bb1(%arg: !quantum.bit):
              "quantum.yield"(%arg) : (!quantum.bit) -> ()
            }) : (!quantum.bit, !quantum.bit, i1, i1, !quantum.bit)
                 -> (!quantum.bit, !quantum.bit, !quantum.bit)
            "func.return"() : () -> ()
          }) {function_type = (!quantum.reg) -> (), sym_name = "f"} : () -> ()
        }) : () -> ()
        """
        module = parse_generic_to_xdsl_module(program)
        ctrl = next(op for op in module.walk() if isinstance(op, CtrlOp))

        assert resolve_constant_wire(ctrl.out_ctrl_qubits[0]) == 1
        assert resolve_constant_wire(ctrl.out_ctrl_qubits[1]) == 2

        # A target `results[j]` adds to `args[j]` (the `ssa.index - num_ctrl` branch).
        assert resolve_constant_wire(ctrl.outs[0]) == 0
