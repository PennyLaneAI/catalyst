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

from typing import TypeVar

import jax
import pennylane as qml
import pennylane.compiler.python_compiler.quantum_dialect as quantum
from pennylane.compiler.python_compiler.transforms import xdsl_transform
from xdsl import ir, passes
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.pattern_rewriter import *
from xdsl.rewriter import InsertPoint

from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

#############
# Transform #
#############

T = TypeVar("T")


def get_parent_of_type(op: ir.Operation, kind: T) -> T | None:
    """Walk up the parent tree until an op of the specified type is found."""

    while (op := op.parent_op()) and not isinstance(op, kind):
        pass

    return op


class TreeTraversal(RewritePattern):

    def __init__(self):
        self.segment_counter = 0
        self.segment_functions: list[func.FuncOp] = []
        self.alloc_op: quantum.AllocOp = None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: PatternRewriter):
        """Transform a quantum function (qnode) to perform tree-traversal simulation."""
        if not "qnode" in funcOp.attributes:
            return

        module = get_parent_of_type(funcOp, builtin.ModuleOp)
        assert module is not None, "got orphaned qnode function"

        # Start with creating a new QNode function that will perform the tree traversal simulation.
        ttOp = funcOp.clone_without_regions()
        ttOp.sym_name = builtin.StringAttr(ttOp.sym_name.name + ".tree_traversal")
        rewriter.create_block(BlockInsertPoint.at_start(ttOp.body), ttOp.function_type.inputs)
        rewriter.insert_op(ttOp, InsertPoint.at_end(module.body.block))

        self.split_traversal_segments(funcOp, ttOp, rewriter)

        self.initialize_data_structures(ttOp, rewriter)

        self.generate_traversal_code(ttOp, rewriter)

    def split_traversal_segments(
        self, funcOp: func.FuncOp, ttOp: func.FuncOp, rewriter: PatternRewriter
    ):
        """Split the quantum function into segments separated by measure operations."""
        rewriter.insertion_point = InsertPoint.at_start(ttOp.body.block)

        # Ideally try to iterate over the function only once.
        op_iter = funcOp.body.walk()

        # Skip to the start of the first simulation segment.
        value_mapper = {}
        while (op := next(op_iter, None)) and not isinstance(op, quantum.AllocOp):
            rewriter.insert(op.clone(value_mapper))
        assert op is not None, "didn't find an alloc op"
        rewriter.insert(op.clone(value_mapper))
        self.alloc_op = op

        # Generate new functions for each segment separated by a measure op.
        op_list = []
        while op := next(op_iter, None):
            if isinstance(op, quantum.MeasureOp):
                self.clone_ops_into_func(op_list, rewriter)
                op_list = []
            elif isinstance(op, quantum.DeallocOp):
                self.clone_ops_into_func(op_list, rewriter)
                break
            else:
                op_list.append(op)
        assert op is not None, "didn't find a dealloc op"

    def clone_ops_into_func(self, op_list: list[ir.Operation], rewriter: PatternRewriter):
        """Clone a set of ops into a new function."""
        if not op_list:
            return

        module = get_parent_of_type(op_list[0], builtin.ModuleOp)
        assert module is not None, "got orphaned operation"

        input_vals, output_vals = self.gather_segment_io(op_list)

        fun_type = builtin.FunctionType.from_lists(
            [arg.type for arg in input_vals], [res.type for res in output_vals]
        )
        new_func = func.FuncOp(f"quantum_segment_{self.segment_counter}", fun_type)
        self.segment_functions.append(new_func)
        self.segment_counter += 1

        rewriter.insertion_point = InsertPoint.at_start(new_func.body.block)

        value_mapper = dict(zip(input_vals, new_func.args))
        for op in op_list:
            new_op = op.clone(value_mapper)
            rewriter.insert(new_op)

        returnOp = func.ReturnOp(*(value_mapper[res] for res in output_vals))
        rewriter.insert(returnOp)

        rewriter.insert_op(new_func, InsertPoint.at_end(module.body.block))

    @staticmethod
    def gather_segment_io(op_list: list[ir.Operation]) -> tuple[set[ir.SSAValue], set[ir.SSAValue]]:
        """Gather SSA values that need to be passed in and out of the segment to be outlined."""
        inputs = set()
        outputs = set()

        # TODO: The outputs are not quite right, since just because an op result has been used
        #       doesn't mean it won't be needed in a subsequent region.
        #       The inputs should be correct though.
        #       Might also have to be more careful with the quantum input/output values.
        for op in op_list:
            inputs.update(op.operands)
            inputs.difference_update(op.results)

            outputs.update(op.results)
            outputs.difference_update(op.operands)

        return inputs, outputs

    def initialize_data_structures(self, ttOp: func.FuncOp, rewriter: PatternRewriter):
        """Create data structures in the IR required for the dynamic tree traversal."""
        rewriter.insertion_point = InsertPoint.at_end(ttOp.body.block)

        # get the tree depth (for now just the segment count)
        tree_depth = arith.ConstantOp.from_int_and_width(len(self.segment_functions), 64).result
        rewriter.insert(tree_depth.owner)

        # get the qubit count
        if self.alloc_op.nqubits:
            qubit_count = self.alloc_op.nqubits
        else:
            qubit_count = arith.ConstantOp(self.alloc_op.nqubits_attr).result
            rewriter.insert(qubit_count.owner)

        # initialize stack variables

        # statevector storage to allow for rollback
        const_1 = arith.ConstantOp.from_int_and_width(1, 64).result
        rewriter.insert(const_1.owner)
        statevec_size = arith.ShLIOp(const_1, qubit_count).result
        rewriter.insert(statevec_size.owner)
        statevec_type = builtin.TensorType(
            builtin.ComplexType(builtin.f64), (builtin.DYNAMIC_INDEX,)
        )
        # TODO: don't know if this actually works, alternatively we can instantiate a 2D array
        statevec_stack_type = builtin.MemRefType(statevec_type, (builtin.DYNAMIC_INDEX,))
        statevec_stack = memref.AllocOp((tree_depth,), (), statevec_stack_type).results[0]
        rewriter.insert(statevec_stack.owner)

        # probabilities for each branch are tracked here
        probs_stack_type = builtin.MemRefType(builtin.f64, (builtin.DYNAMIC_INDEX,))
        probs_stack = memref.AllocOp((tree_depth,), (), probs_stack_type).results[0]
        rewriter.insert(probs_stack.owner)

        # For the current path, we track whether a node is:
        #  - unvisited: 0
        #  - visited down the left branch: 1
        #  - visited down the right branch: 2
        visited_stack_type = builtin.MemRefType(builtin.i8, (builtin.DYNAMIC_INDEX,))
        visited_stack = memref.AllocOp((tree_depth,), (), visited_stack_type).results[0]
        rewriter.insert(visited_stack.owner)

        # store some useful values for later
        self.tree_depth = tree_depth
        self.statevec_stack = statevec_stack
        self.probs_stack = probs_stack
        self.visited_stack = visited_stack

    def generate_traversal_code(self, ttOp: func.FuncOp, rewriter: PatternRewriter):
        """Create the traversal code of the quantum simulation tree."""
        rewriter.insertion_point = InsertPoint.at_end(ttOp.body.block)


@xdsl_transform
class TTPass(passes.ModulePass):
    name = "tree-traversal"

    def apply(self, ctx: Context, module: builtin.ModuleOp) -> None:

        # # Fixed-point iteration with pattern application, not suited for all kinds of transforms.
        # pattern_list = [TreeTraversal()]
        # greedy_rewriter = GreedyRewritePatternApplier(pattern_list)
        # rewrite_walker = PatternRewriteWalker(greedy_rewriter)
        # rewrite_walker.rewrite_module(module)

        self.apply_on_qnode(module, TreeTraversal())

        module.verify()
        print(module)

    def apply_on_qnode(self, module: builtin.ModuleOp, pattern: RewritePattern):
        """Apply given pattern once to the QNode function in this module."""
        rewriter = PatternRewriter(module)

        qnode = None
        for op in module.ops:
            if isinstance(op, func.FuncOp) and "qnode" in op.attributes:
                qnode = op
                break
        assert qnode is not None, "expected QNode in module"

        pattern.match_and_rewrite(qnode, rewriter)


###########
# Example #
###########

if __name__ == "__main__":

    qml.capture.enable()

    @jax.jit
    def add(a, b):
        return a + b

    @TTPass
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def captured_circuit(x: float):

        qml.Hadamard(wires=0)
        m = qml.measure(0)
        qml.RX(x, wires=0)
        m = qml.measure(0)
        qml.RY(add(x, x), wires=0)
        m = qml.measure(0)
        qml.cond(m, lambda: qml.X(0))

        return qml.state()

    @qml.qjit(keep_intermediate=False, pass_plugins=[getXDSLPluginAbsolutePath()])
    def main(x: float):
        return captured_circuit(x)

    print(main(1.0))
