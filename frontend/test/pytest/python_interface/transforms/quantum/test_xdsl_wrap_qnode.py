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

"""Unit and integration tests for the unified compiler `measurements_from_samples` transform."""

# pylint: disable=line-too-long

import pennylane as qml
import pytest
from pennylane.exceptions import CompileError
from xdsl.context import Context
from xdsl.dialects import func, test

from catalyst.python_interface import QuantumParser
from catalyst.python_interface.conversion import xdsl_from_qjit
from catalyst.python_interface.transforms.quantum.wrap_qnode import WrapQNodePass, get_call_op

pytestmark = pytest.mark.xdsl


class TestWrapQnode:
    """Test that the utility pass WrapQNodePass behaves as expected"""

    def test_wrapping_single_quantum_node(self, run_filecheck):
        """Test the wrap_qnode pass works as expected on a module containing a single quantum.node"""

        program = """
        builtin.module @module_circuit {
            // CHECK: func.func public @circuit() -> tensor<f64> {
            func.func public @circuit() -> (tensor<f64>) attributes {quantum.node}  {
                // CHECK: func.call @circuit.my_pass_name() : () -> tensor<f64>
                // CHECK: func.func public @circuit.my_pass_name() -> tensor<f64> attributes {quantum.node}
                // CHECK: "test.op"() : () -> !quantum.bit
                // CHECK: quantum.namedobs
                // CHECK: quantum.expval
                %0 = "test.op"() : () -> !quantum.bit
                %1 = quantum.namedobs %0[PauliZ] : !quantum.obs
                %2 = quantum.expval %1 : f64
                %3 = "tensor.from_elements"(%2) : (f64) -> tensor<f64>
                func.return %3 : tensor<f64>
            }
        }
        """

        pipeline = (WrapQNodePass("my_pass_name"),)
        run_filecheck(program, pipeline)

    def test_wrapping_multiple_quantum_nodes(self, run_filecheck):
        """Test the wrap_qnode pass works as expected on a module containing multiple quantum.nodes"""

        program = """
        builtin.module @module_circuit {
            // CHECK: func.func public @circuit.1() -> tensor<f64> {
            func.func public @circuit.1() -> (tensor<f64>) attributes {quantum.node} {
                // CHECK: func.call @circuit.1.my_pass_name() : () -> tensor<f64>
                // CHECK: func.func public @circuit.1.my_pass_name() -> tensor<f64> attributes {quantum.node}
                %0 = "test.op"() : () -> !quantum.bit
                %1 = quantum.namedobs %0[PauliZ] : !quantum.obs
                %2 = quantum.expval %1 : f64
                %3 = "tensor.from_elements"(%2) : (f64) -> tensor<f64>
                func.return %3 : tensor<f64>
            }
            // CHECK: func.func public @circuit.2() -> tensor<f64> {
            func.func public @circuit.2() -> (tensor<f64>) attributes {quantum.node} {
                // CHECK: func.call @circuit.2.my_pass_name() : () -> tensor<f64>
                // CHECK: func.func public @circuit.2.my_pass_name() -> tensor<f64> attributes {quantum.node}
                %0 = "test.op"() : () -> !quantum.bit
                %1 = quantum.namedobs %0[PauliZ] : !quantum.obs
                %2 = quantum.expval %1 : f64
                %3 = "tensor.from_elements"(%2) : (f64) -> tensor<f64>
                func.return %3 : tensor<f64>
            }
        }
        """

        pipeline = (WrapQNodePass("my_pass_name"),)
        run_filecheck(program, pipeline)

    def test_nesting_classical_functions(self, run_filecheck):
        """Test that we can apply the pass repeatedly to add nested classical functions
        for different post-processing steps. The first pass applied renames the QNode
        as circuit.a, and calls it within a new classical function, which retains the name
        of the original quantum.node.

        The second pass wraps the new quantum.node, circuit.a, in a classical function (now
        circuit.a), and renames the quantum.node to circuit.a.b.

        In the end, the quantum.node is circuit.a.b, and there are two layers of classical
        functions it is wrapped inside of, circuit.a and then circuit."""

        program = """
        builtin.module @module_circuit {
            // CHECK: func.func public @circuit() -> tensor<f64> {
            func.func public @circuit() -> (tensor<f64>) attributes {quantum.node}  {
                // CHECK: func.call @circuit.a() : () -> tensor<f64>
                // CHECK: func.func public @circuit.a() -> tensor<f64> {
                // CHECK: func.call @circuit.a.b() : () -> tensor<f64>
                // CHECK: func.func public @circuit.a.b() -> tensor<f64> attributes {quantum.node}
                // CHECK: "test.op"() : () -> !quantum.bit
                // CHECK: quantum.namedobs
                // CHECK: quantum.expval
                %0 = "test.op"() : () -> !quantum.bit
                %1 = quantum.namedobs %0[PauliZ] : !quantum.obs
                %2 = quantum.expval %1 : f64
                %3 = "tensor.from_elements"(%2) : (f64) -> tensor<f64>
                func.return %3 : tensor<f64>
            }
        }
        """

        pipeline = (WrapQNodePass("a"), WrapQNodePass("b"))
        run_filecheck(program, pipeline)


class TestGetCallOp:
    """Test that get_call_op behaves as expected when passed a func.FuncOp"""

    def test_get_call_op(self):
        """Test get_call_op retrieves the func.CallOp for a quantum.node that is
        called within the module"""

        program_str = """
            builtin.module @module_circuit {
            func.func public @circuit() -> tensor<f64> {
                %0 = func.call @circuit.test_name() : () -> tensor<f64>
                func.return %0 : tensor<f64>
            }
            func.func public @circuit.test_name() -> tensor<f64> attributes {quantum.node} {
                %0 = "test.op"() : () -> !quantum.bit
                %1 = quantum.namedobs %0[PauliZ] : !quantum.obs
                %2 = quantum.expval %1 : f64
                %3 = tensor.from_elements %2 : tensor<f64>
                func.return %3 : tensor<f64>
            }
        }
        """

        ctx = Context(allow_unregistered=False)
        xdsl_module = QuantumParser(ctx, program_str, extra_dialects=(test.Test,)).parse_module()

        op = None
        for op in xdsl_module.walk():
            if isinstance(op, func.FuncOp) and "quantum.node" in op.attributes:
                break

        call_op = get_call_op(op)
        assert isinstance(call_op, func.CallOp)
        assert call_op.callee.string_value() == op.sym_name.data

    def test_get_call_op_and_pass_integration(self):
        """Test get_call_op retrieves the func.CallOp for a quantum.node when
        WrapQNodePass has been applied to circuit."""

        @xdsl_from_qjit
        @qml.qjit
        @qml.qnode(qml.device("null.qubit", wires=1))
        def circ():
            return qml.expval(qml.Z(0))

        xdsl_module = circ()
        WrapQNodePass("test").apply(None, xdsl_module)

        op = None
        for op in xdsl_module.walk():
            if isinstance(op, func.FuncOp) and "quantum.node" in op.attributes:
                break

        call_op = get_call_op(op)
        assert isinstance(call_op, func.CallOp)
        assert call_op.callee.string_value() == op.sym_name.data

    def test_multiple_calls_raises_error(self):
        """Test that if there is more than one call the QNode, an error is raise.
        This should not happen."""

        program_str = """
            builtin.module @module_circuit {
            func.func public @circuit() -> (tensor<f64>, tensor<f64>) {
                %0 = func.call @circuit.test_name() : () -> tensor<f64>
                %1 = func.call @circuit.test_name() : () -> tensor<f64>
                func.return %0, %1 : tensor<f64>, tensor<f64>
            }
            func.func public @circuit.test_name() -> tensor<f64> attributes {quantum.node} {
                %0 = "test.op"() : () -> !quantum.bit
                %1 = quantum.namedobs %0[PauliZ] : !quantum.obs
                %2 = quantum.expval %1 : f64
                %3 = tensor.from_elements %2 : tensor<f64>
                func.return %3 : tensor<f64>
            }
            }
        """

        xdsl_module = QuantumParser(
            Context(), program_str, extra_dialects=(test.Test,)
        ).parse_module()

        op = None
        for op in xdsl_module.walk():
            if isinstance(op, func.FuncOp) and "quantum.node" in op.attributes:
                break

        with pytest.raises(
            CompileError, match="Expected only one call_op for circuit.test_name, but received 2"
        ):
            _ = get_call_op(op)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
