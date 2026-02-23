# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for CompilationPass."""
from functools import lru_cache, partial
from typing import Union

import pennylane as qml
import pytest
from xdsl.context import Context
from xdsl.dialects import arith, builtin, test
from xdsl.rewriter import InsertPoint

from catalyst.python_interface import QuantumParser
from catalyst.python_interface.conversion import parse_generic_to_xdsl_module
from catalyst.python_interface.dialects import quantum
from catalyst.python_interface.pass_api import CompilationPass, compiler_transform

pytestmark = pytest.mark.xdsl
# parse_generic_to_xdsl_module doesn't load the test dialect by default
parse_xdsl_str = partial(parse_generic_to_xdsl_module, extra_dialects=(test.Test,))


@pytest.fixture(scope="function")
def ctx():
    """Context to use for applying passes."""
    _ctx = Context()
    # QuantumParser automatically populates ctx with the dialects we want
    _ = QuantumParser(_ctx, "", extra_dialects=(test.Test,))
    return _ctx


@lru_cache
def create_test_pass(greedy: bool, recursive: bool) -> CompilationPass:
    """Helper to create a compilation pass for testing"""
    # pylint: disable=unused-argument

    class MyPass(CompilationPass):
        """Compilation pass for testing."""

        name = "my-pass"
        counts: dict[str, int]

        def __init__(self):
            # Keys should be "default", "gate1", "gate2", "mcm", "ins_ex",
            # corresponding to the different actions.
            self.counts = {
                "default": 0,
                "gate1": 0,
                "gate2": 0,
                "mcm": 0,
                "ins_ex": 0,
                "alloc_dealloc": 0,
            }

        def action(self, op: quantum.CustomOp, rewriter):
            """Default action. Do nothing."""
            assert isinstance(op, quantum.CustomOp)
            self.counts["default"] += 1

    MyPass.greedy = greedy
    MyPass.recursive = recursive

    @MyPass.add_action
    def gate_action1(self, op: quantum.CustomOp, rewriter):
        """Action 1 on gates. Add a constant, and erase the gate if it's a PauliX."""
        assert isinstance(op, quantum.CustomOp)
        self.counts["gate1"] += 1

        if op.gate_name.data == "PauliX":
            cst = arith.ConstantOp.from_int_and_width(1, 64)
            rewriter.insert_op(cst, insertion_point=InsertPoint.before(op))
            new_op = quantum.CustomOp(gate_name="PauliY", in_qubits=op.in_qubits)
            rewriter.replace_op(op, new_op)

    @MyPass.add_action
    def gate_action2(self, op: quantum.CustomOp, rewriter):
        """Action 2 on gates. Erase the gate if it's NOT a PauliX."""
        assert isinstance(op, quantum.CustomOp)
        self.counts["gate2"] += 1

        if op.gate_name.data != "PauliX":
            rewriter.erase_op(op)

    @MyPass.add_action
    def mcm_action(self, op: quantum.MeasureOp, rewriter):
        """Action on mcms. Insert a PauliX gate and erase the MCM."""
        assert isinstance(op, quantum.MeasureOp)
        self.counts["mcm"] += 1
        new_op = quantum.CustomOp(gate_name="PauliX", in_qubits=(op.in_qubit,))
        rewriter.insert_op(new_op, insertion_point=InsertPoint.before(op))

        if self.counts["mcm"] > 1:
            rewriter.erase_op(op)

    # Union type hints, both using 'Union' and '|'

    @MyPass.add_action
    def insert_extract_action(self, op: quantum.InsertOp | quantum.ExtractOp, rewriter):
        """Action on qubit inserts and extracts. Erase the op."""
        assert isinstance(op, (quantum.InsertOp, quantum.ExtractOp))
        self.counts["ins_ex"] += 1

        rewriter.erase_op(op)

    @MyPass.add_action
    def alloc_dealloc_action(self, op: Union[quantum.AllocOp, quantum.DeallocOp], rewriter):
        """Action on quantum register allocation/deallocation ops. Erase the op"""
        assert isinstance(op, (quantum.AllocOp, quantum.DeallocOp))
        self.counts["alloc_dealloc"] += 1

        rewriter.erase_op(op)

    return MyPass


class TestCompilationPass:
    """Unit tests for CompilationPass."""

    # pylint: disable=too-many-arguments,too-many-positional-arguments,redefined-outer-name

    @pytest.mark.parametrize(
        "greedy,recursive,expected_op_types,expected_counts",
        [
            (
                # Non-greedy, non-recursive
                False,
                False,
                # Default will do nothing, but it will see all CustomOps
                # Gate1 will erase the PauliX and add an arith.constant and PauliY
                #   It will also see the PauliZ but not do anything
                # Gate2 will erase the PauliY and PauliZ
                # MCM will add a PauliX before the MCM
                (
                    test.TestOp,
                    test.TestOp,
                    test.TestOp,
                    arith.ConstantOp,
                    quantum.CustomOp,
                    quantum.MeasureOp,
                ),
                {"default": 2, "gate1": 2, "gate2": 2, "mcm": 1, "ins_ex": 0, "alloc_dealloc": 0},
            ),
            (
                # Non-greedy, recursive
                False,
                True,
                # Default will do nothing but it will see all CustomOps
                # Gate1 will erase the PauliX and add an arith.constant and PauliY
                #   It will also see PauliZ, and because recursive=True, PauliY as well
                #   Because the program was modified and recursive=True, we will iterate over
                #   the instructions again and see the PauliY and PauliZ again
                # Gate2 will see and remove PauliY and PauliZ
                # MCM will add a PauliX. recursive=True will trigger another invocation,
                #   and in the second invocation, another PauliX will be added and
                #   the MeasureOp will be erased
                (
                    test.TestOp,
                    test.TestOp,
                    test.TestOp,
                    arith.ConstantOp,
                    quantum.CustomOp,
                    quantum.CustomOp,
                ),
                {"default": 2, "gate1": 5, "gate2": 2, "mcm": 2, "ins_ex": 0, "alloc_dealloc": 0},
            ),
            (
                # Greedy, non-recursive
                True,
                False,
                # Default will see PauliX, do nothing. Gate1 will see PauliX,
                #   replace with arith.ConstantOp + PauliY. Non-recursive
                #   so we don't process the added ops
                # Default will see PauliZ, do nothing. Gate1 will see PauliZ,
                #   do nothing. Gate2 will see PauliZ, erase it
                # MCM will see MCM, add PauliX. Non-recursive so we don't
                #   process the added ops
                (
                    test.TestOp,
                    test.TestOp,
                    test.TestOp,
                    arith.ConstantOp,
                    quantum.CustomOp,
                    quantum.CustomOp,
                    quantum.MeasureOp,
                ),
                {"default": 2, "gate1": 2, "gate2": 1, "mcm": 1, "ins_ex": 0, "alloc_dealloc": 0},
            ),
            (
                # Greedy, recursive
                True,
                True,
                # Default will see PauliX, do nothing. Gate1 will see PauliX,
                #   replace with arith.ConstantOp + PauliY
                # Default will see PauliZ, do nothing. Gate1 will see PauliZ,
                #   do nothing. Gate2 will see PauliZ, erase it
                # Default will see PauliY, do nothing. Gate1 will see PauliY,
                #   do nothing. Gate2 will see PauliY, erase it
                # MCM will see MCM, add PauliX
                # Default will see PauliX, do nothing. Gate1 will see PauliX,
                #   replace with arith.ConstantOp + PauliY
                # Default will see PauliY, do nothing. Gate1 will see PauliY,
                #   do nothing. Gate2 will see PauliY, erase it
                # MCM will see MCM, add PauliX, erase MCM
                # Default will see PauliX, do nothing. Gate1 will see PauliX,
                #   replace with arith.ConstantOp + PauliY
                # Default will see PauliY, do nothing. Gate1 will see PauliY,
                #   do nothing. Gate2 will see PauliY, erase it
                (
                    test.TestOp,
                    test.TestOp,
                    test.TestOp,
                    arith.ConstantOp,
                    arith.ConstantOp,
                    arith.ConstantOp,
                ),
                {"default": 7, "gate1": 7, "gate2": 4, "mcm": 2, "ins_ex": 0, "alloc_dealloc": 0},
            ),
        ],
    )
    def test_actions(self, greedy, recursive, expected_op_types, expected_counts, ctx):
        """Test that the action(s) of a CompilationPass are applied correctly to a module."""
        mod_str = """
            %q0 = "test.op"() : () -> !quantum.bit
            %q1 = "test.op"() : () -> !quantum.bit
            %q2 = "test.op"() : () -> !quantum.bit
            %q3 = quantum.custom "PauliX"() %q0 : !quantum.bit
            %mres, %q4 = quantum.measure %q1 : i1, !quantum.bit
            %q5 = quantum.custom "PauliZ"() %q2 : !quantum.bit
        """
        mod = parse_xdsl_str(mod_str)

        pass_ = create_test_pass(greedy=greedy, recursive=recursive)()
        pass_.apply(ctx, mod)

        print(f"{greedy=}, {recursive=}")
        print(mod)
        print(pass_.counts)
        assert tuple(type(op) for op in mod.ops) == expected_op_types
        assert pass_.counts == expected_counts

    @pytest.mark.parametrize("greedy", [True, False])
    @pytest.mark.parametrize("recursive", [True, False])
    def test_union_type_hints(self, greedy, recursive, ctx):
        """Test that actions that use Unions of operations as their type hints
        work correctly."""
        mod_str = """
            %q0 = "test.op"() : () -> !quantum.bit
            %r1 = "test.op"() : () -> !quantum.reg
            %r2 = quantum.insert %r1[0], %q0 : !quantum.reg, !quantum.bit
            %q1 = quantum.extract %r1[0] : !quantum.reg -> !quantum.bit
            %r3 = quantum.alloc(4) : !quantum.reg
            quantum.dealloc %r1 : !quantum.reg
        """
        mod = parse_xdsl_str(mod_str)

        pass_ = create_test_pass(greedy=greedy, recursive=recursive)()
        pass_.apply(ctx, mod)

        # The action on inserts, extracts, allocs, and deallocs is to erase them, so there
        # shouldn't be any in the modified program.
        assert not any(
            isinstance(
                op, (quantum.InsertOp, quantum.ExtractOp, quantum.AllocOp, quantum.DeallocOp)
            )
            for op in mod.ops
        )
        # insert_extract_action should be invoked for both the quantum.insert and
        # quantum.extract operations
        assert pass_.counts == {
            "default": 0,
            "gate1": 0,
            "gate2": 0,
            "mcm": 0,
            "ins_ex": 2,
            "alloc_dealloc": 2,
        }

    def test_add_action_invalid_args(self):
        """Test that adding actions with invalid arguments raises an error."""
        pass_cls = create_test_pass(greedy=False, recursive=False)

        with pytest.raises(
            ValueError, match="The action must have 3 arguments, with the first one being 'self'"
        ):

            @pass_cls.add_action
            def new_action1(pass_, op, rewriter):
                return

        with pytest.raises(
            ValueError, match="The action must have 3 arguments, with the first one being 'self'"
        ):

            @pass_cls.add_action
            def new_action2(self, op1, op2, rewriter):
                return

    def test_add_action_invalid_type_hint(self):
        """Test that using type hints that are not xDSL operations or unions of xDSL operations
        raises an error."""
        pass_cls = create_test_pass(greedy=False, recursive=False)

        with pytest.raises(TypeError, match="Only Operation types or unions of Operation types"):

            @pass_cls.add_action
            def new_action1(self, op: qml.PauliX, rewriter):
                return

        with pytest.raises(TypeError, match="Only Operation types or unions of Operation types"):

            @pass_cls.add_action
            def new_action1(self, op: qml.PauliX | qml.PauliY, rewriter):
                return

        with pytest.raises(TypeError, match="Only Operation types or unions of Operation types"):

            @pass_cls.add_action
            def new_action1(self, op: Union[qml.PauliX, qml.PauliY], rewriter):
                return

    def test_base_class_add_action_error(self):
        """Test that an error is raised with trying to add an action to the CompilationPass
        base class."""

        def null_action(self, op, rewriter):
            return

        with pytest.raises(TypeError, match="Cannot use 'CompilationPass.add_action'"):
            CompilationPass.add_action(null_action)


class IntegrationTestPass(CompilationPass):
    """Compilation pass for integration testing."""

    name = "integration-pass"

    def action(self, op: quantum.CustomOp, rewriter):
        """Replace H with Y, remove X."""
        if op.gate_name.data == "Hadamard":
            new_op = quantum.CustomOp(
                gate_name="PauliY", in_qubits=op.in_qubits, in_ctrl_qubits=op.in_ctrl_qubits
            )
            rewriter.replace_op(op, new_op)
        elif op.gate_name.data == "PauliX":
            rewriter.replace_op(op, (), op.in_qubits + op.in_ctrl_qubits)


@IntegrationTestPass.add_action
def _(self, op: quantum.MeasureOp, rewriter):
    """Extend the classical MCM value if the MCM occurred right after a PauliY, and erase
    the PauliY."""
    if (
        isinstance(owner := op.in_qubit.owner, quantum.CustomOp)
        and op.in_qubit.owner.gate_name.data == "PauliY"
    ):
        rewriter.replace_op(owner, (), owner.in_qubits + owner.in_ctrl_qubits)
        ext_op = arith.ExtUIOp(op.mres, target_type=builtin.i64)
        rewriter.insert_op(ext_op, InsertPoint.after(op))


integration_pass = compiler_transform(IntegrationTestPass)


@pytest.mark.parametrize("capture", [True, False])
class TestCompilationPassIntegration:
    """Integration tests for CompilationPass."""

    def test_qjit_integration(self, run_filecheck_qjit, capture):
        """Test that passes created using CompilationPass can be used
        with qjit."""

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit(capture=capture)
        @integration_pass
        @qml.qnode(dev)
        def circuit():
            # CHECK-NOT: quantum.custom
            qml.Hadamard(0)
            qml.PauliX(0)
            # CHECK: [[MRES:%.+]], {{%.+}} = quantum.measure
            # CHECK: arith.extui [[MRES]] : i1 to i64
            _ = qml.measure(0)
            return qml.state()

        run_filecheck_qjit(circuit)
        assert qml.math.allclose(circuit(), [1, 0])


if __name__ == "__main__":
    pytest.main(["-x", __file__])
