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

"""Implementation of the Tree-Traversal MCM simulation method as an xDSL transform in Catalyst."""

from dataclasses import dataclass, field
from typing import Type, TypeVar

import jax
import pennylane as qml
import pennylane.compiler.python_compiler.quantum_dialect as quantum
from pennylane.compiler.python_compiler.transforms import xdsl_transform
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf, tensor
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import BlockInsertPoint, InsertPoint

from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

T = TypeVar("T")


def get_parent_of_type(op: Operation, kind: Type[T]) -> T | None:
    """Walk up the parent tree until an op of the specified type is found."""

    while (op := op.parent_op()) and not isinstance(op, kind):
        pass

    return op


@dataclass
class ProgramSegment:
    """A program segment and associated data."""

    ops: list[Operation] = field(default_factory=list)
    mcm: quantum.MeasureOp = None
    inputs: set[SSAValue] = None
    outputs: set[SSAValue] = None
    fun: func.FuncOp = None


class TreeTraversal(RewritePattern):

    def __init__(self):
        self.module: builtin.ModuleOp = None
        self.ttOp: func.FuncOp = None
        self.quantum_segments: list[ProgramSegment] = []

    @op_type_rewrite_pattern
    def match_and_rewrite(self, funcOp: func.FuncOp, rewriter: PatternRewriter):
        """Transform a quantum function (qnode) to perform tree-traversal simulation."""
        if not "qnode" in funcOp.attributes:
            return

        self.module = get_parent_of_type(funcOp, builtin.ModuleOp)
        assert self.module is not None, "got orphaned qnode function"

        # Start with creating a new QNode function that will perform the tree traversal simulation.
        self.setup_traversal_function(funcOp, rewriter)

        self.split_traversal_segments(funcOp, rewriter)

        self.initialize_data_structures(rewriter)

        self.generate_traversal_code(rewriter)

        self.finalize_traversal_function(rewriter)

    def setup_traversal_function(self, funcOp: func.FuncOp, rewriter: RewritePattern):
        """Setup a clone of the original QNode function, which will instead perform TT."""

        ttOp = funcOp.clone_without_regions()
        ttOp.sym_name = builtin.StringAttr(funcOp.sym_name.data + ".tree_traversal")
        rewriter.create_block(BlockInsertPoint.at_start(ttOp.body), ttOp.function_type.inputs)
        rewriter.insert_op(ttOp, InsertPoint.at_end(self.module.body.block))

        self.ttOp = ttOp

    def finalize_traversal_function(self, rewriter: RewritePattern):
        """Complete the function and ensure it's correctly formed, e.g. returning proper results."""
        rewriter.insertion_point = InsertPoint.at_end(self.ttOp.body.block)

        # For now return nothing from the Tree-Traversal function to satisfy the verifier.
        result_vals = []
        for resType in self.ttOp.function_type.outputs:
            assert isinstance(resType, builtin.TensorType)
            result = tensor.EmptyOp((), tensor_type=resType)
            result_vals.append(rewriter.insert(result))

        rewriter.insert(func.ReturnOp(*result_vals))

    def split_traversal_segments(self, funcOp: func.FuncOp, rewriter: PatternRewriter):
        """Split the quantum function into segments separated by measure operations."""
        rewriter.insertion_point = InsertPoint.at_start(self.ttOp.body.block)

        # Ideally try to iterate over the function only once.
        op_iter = funcOp.body.walk()

        # Skip to the start of the first simulation segment.
        value_mapper = {}
        while (op := next(op_iter, None)) and not isinstance(op, quantum.AllocOp):
            rewriter.insert(op.clone(value_mapper))
        assert op is not None, "didn't find an alloc op"
        self.alloc_op = rewriter.insert(op.clone(value_mapper))

        # Split ops into segments divided by measurements.
        quantum_segments = [ProgramSegment()]
        while (op := next(op_iter, None)) and not isinstance(op, quantum.DeallocOp):
            if isinstance(op, quantum.MeasureOp):
                quantum_segments[-1].mcm = op
                quantum_segments.append(ProgramSegment())
            else:
                quantum_segments[-1].ops.append(op)
        assert op is not None, "didn't find a dealloc op"
        self.quantum_segments = quantum_segments

        # Go through the rest of the function to initialize the missing input values set.
        terminal_segment = []
        while op := next(op_iter, None):
            terminal_segment.append(op)

        # Generate new functions for each segment separated by a measure op.
        # We traverse them bottom up first to correctly determine the I/O of each segment.
        missing_input_values, _ = self.gather_segment_io(terminal_segment)
        for segment in reversed(quantum_segments):
            missing_input_values.update(getattr(segment.mcm, "operands", ()))
            inputs, outputs = self.gather_segment_io(segment.ops, missing_input_values)
            segment.inputs, segment.outputs = inputs, outputs

        for idx, segment in enumerate(quantum_segments):
            segment.fun = self.clone_ops_into_func(segment, idx, rewriter)

        # Generate a function table to select the right program segment.

    def clone_ops_into_func(self, segment: ProgramSegment, id: int, rewriter: PatternRewriter):
        """Clone a set of ops into a new function."""
        op_list, input_vals, output_vals = segment.ops, segment.inputs, segment.outputs
        if not op_list:
            return

        fun_type = builtin.FunctionType.from_lists(
            [arg.type for arg in input_vals], [res.type for res in output_vals]
        )
        new_func = func.FuncOp(f"quantum_segment_{id}", fun_type)

        value_mapper = dict(zip(input_vals, new_func.args))
        for op in op_list:
            new_op = op.clone(value_mapper)
            rewriter.insert_op(new_op, InsertPoint.at_end(new_func.body.block))

        returnOp = func.ReturnOp(*(value_mapper[res] for res in output_vals))
        rewriter.insert_op(returnOp, InsertPoint.at_end(new_func.body.block))

        rewriter.insert_op(new_func, InsertPoint.at_end(self.module.body.block))
        return new_func

    @staticmethod
    def gather_segment_io(
        op_list: list[Operation], missing_inputs: set[SSAValue] = None
    ) -> tuple[set[SSAValue], set[SSAValue]]:
        """Gather SSA values that need to be passed in and out of the segment to be outlined."""
        inputs = set()
        outputs = set()
        if missing_inputs is None:
            missing_inputs = set()

        # The segment only needs to return values produced here (i.e. in all op.results) and
        # required by segments further down (i.e. in missing_inputs).
        # The inputs are determined straightforwardly by all operands not defined in this segment.
        # TODO: We might need to be more careful with qubit/register values in the future.
        for op in reversed(op_list):
            inputs.update(op.operands)
            inputs.difference_update(op.results)

            outputs.update(op.results)
        outputs.intersection_update(missing_inputs)

        # Update the information used in subsequent calls.
        missing_inputs.difference_update(outputs)
        missing_inputs.update(inputs)

        return inputs, outputs

    def initialize_data_structures(self, rewriter: PatternRewriter):
        """Create data structures in the IR required for the dynamic tree traversal."""
        rewriter.insertion_point = InsertPoint.at_end(self.ttOp.body.block)

        # get the qubit count
        if self.alloc_op.nqubits:
            qubit_count = arith.IndexCastOp(self.alloc_op.nqubits, builtin.IndexType())
        else:
            qubit_count = arith.ConstantOp.from_int_and_width(
                self.alloc_op.nqubits_attr.value, builtin.IndexType()
            )

        # get the tree depth (for now just the segment count)
        tree_depth = arith.ConstantOp.from_int_and_width(
            len(self.quantum_segments), builtin.IndexType()
        )

        # initialize stack variables #

        # statevector storage to allow for rollback
        c1 = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
        statevec_size = arith.ShLIOp(c1, qubit_count)
        statevec_stack_type = builtin.MemRefType(
            builtin.ComplexType(builtin.f64), (builtin.DYNAMIC_INDEX, builtin.DYNAMIC_INDEX)
        )
        statevec_stack = memref.AllocOp((tree_depth, statevec_size), (), statevec_stack_type)

        # probabilities for each branch are tracked here
        probs_stack_type = builtin.MemRefType(builtin.f64, (builtin.DYNAMIC_INDEX,))
        probs_stack = memref.AllocOp((tree_depth,), (), probs_stack_type)

        # For the current path, we track whether a node is:
        #  - unvisited: 0
        #  - visited down the left branch: 1
        #  - visited down the right branch: 2
        visited_stack_type = builtin.MemRefType(builtin.i8, (builtin.DYNAMIC_INDEX,))
        visited_stack = memref.AllocOp((tree_depth,), (), visited_stack_type)

        for op in (
            qubit_count,
            tree_depth,
            c1,
            statevec_size,
            statevec_stack,
            probs_stack,
            visited_stack,
        ):
            rewriter.insert(op)

        # store some useful values for later
        self.tree_depth = tree_depth.result
        self.statevec_size = statevec_size.result
        self.statevec_stack = statevec_stack.results[0]
        self.probs_stack = probs_stack.results[0]
        self.visited_stack = visited_stack.results[0]

    def generate_traversal_code(self, rewriter: PatternRewriter):
        """Create the traversal code of the quantum simulation tree."""
        rewriter.insertion_point = InsertPoint.at_end(self.ttOp.body.block)

        # loop instruction
        depth_init = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
        restore_init = arith.ConstantOp.from_int_and_width(0, 1)

        conditionBlock = Block(arg_types=(depth_init.result.type, restore_init.result.type))
        bodyBlock = Block(arg_types=conditionBlock.arg_types)
        traversalOp = scf.WhileOp(
            (depth_init, restore_init), conditionBlock.arg_types, (conditionBlock,), (bodyBlock,)
        )

        for op in (depth_init, restore_init, traversalOp):
            rewriter.insert(op)

        # condition block of the while loop
        current_depth, needs_restore = self.check_if_leaf(*conditionBlock.args, rewriter)

        c0 = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
        c2 = arith.ConstantOp.from_int_and_width(2, builtin.IndexType())

        positive_depth = arith.CmpiOp(current_depth, c0, "sge")

        visited_sum = self.sum_array(self.visited_stack, rewriter, conditionBlock)
        max_sum = arith.MuliOp(c2, self.tree_depth)
        not_fully_traversed = arith.CmpiOp(visited_sum, max_sum, "ne")

        final_condition = arith.AndIOp(positive_depth, not_fully_traversed)
        condOp = scf.ConditionOp(final_condition, current_depth, needs_restore)

        for op in (c0, c2, positive_depth, max_sum, not_fully_traversed, final_condition, condOp):
            rewriter.insert_op(op, InsertPoint.at_end(conditionBlock))

        # body block of the while
        current_depth, needs_restore = bodyBlock.args

        node_status = memref.LoadOp.get(self.visited_stack, (current_depth,))
        casted_status = arith.IndexCastOp(node_status, builtin.IndexType())

        cases = builtin.DenseArrayBase.from_list(builtin.i64, [0, 1, 2])
        switchOp = scf.IndexSwitchOp(
            casted_status,
            cases,
            Region(Block()),
            [Region(Block()) for _ in range(len(cases))],
            (current_depth.type, needs_restore.type),
        )

        self.process_node(switchOp, current_depth, needs_restore, rewriter)

        yieldOp = scf.YieldOp(*switchOp.results)

        for op in (node_status, casted_status, switchOp, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(bodyBlock))

    def check_if_leaf(
        self, current_depth: SSAValue, needs_restore: SSAValue, rewriter: PatternRewriter
    ) -> tuple[SSAValue, SSAValue]:
        """Verify whether we've hit the bottom of the tree, and perform update actions."""
        assert isinstance(current_depth.owner, Block)
        assert current_depth.owner == needs_restore.owner
        ip_backup = rewriter.insertion_point
        rewriter.insertion_point = InsertPoint.at_start(current_depth.owner)

        # if instruction
        hit_leaf = arith.CmpiOp(current_depth, self.tree_depth, "eq")
        trueBlock, falseBlock = Block(), Block()
        ifOp = scf.IfOp(
            hit_leaf, (current_depth.type, needs_restore.type), (trueBlock,), (falseBlock,)
        )

        for op in (hit_leaf, ifOp):
            rewriter.insert(op)

        # true branch body
        c1 = arith.ConstantOp.from_int_and_width(1, current_depth.type)
        updated_depth = arith.SubiOp(current_depth, c1)
        true = arith.ConstantOp.from_int_and_width(1, 1)
        yieldOp = scf.YieldOp(updated_depth, true)

        for op in (c1, updated_depth, true, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(trueBlock))

        # false branch body
        yieldOp = scf.YieldOp(current_depth, needs_restore)
        rewriter.insert_op(yieldOp, InsertPoint.at_end(falseBlock))

        rewriter.insertion_point = ip_backup

        return ifOp.results

    def process_node(
        self,
        switchOp: scf.IndexSwitchOp,
        current_depth: SSAValue,
        needs_restore: SSAValue,
        rewriter: PatternRewriter,
    ):
        """Update data structures and effect transition from one node to the next."""
        defaultBlock = switchOp.default_region.block
        unvisitedBlock, leftVisitedBlock, rightVisitedBlock = (
            reg.block for reg in switchOp.case_regions
        )

        # handle unvisited region: need to ge left
        c1 = arith.ConstantOp.from_int_and_width(1, self.visited_stack.type.element_type)
        storeOp = memref.StoreOp.get(c1, self.visited_stack, (current_depth,))

        # TODO: add simulation segment

        c1_ = arith.ConstantOp.from_int_and_width(1, current_depth.type)
        updated_depth = arith.AddiOp(current_depth, c1_)

        yieldOp = scf.YieldOp(updated_depth, needs_restore)

        for op in (c1, c1_, storeOp, updated_depth, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(unvisitedBlock))

        # handle left visited region: need to go right
        c2 = arith.ConstantOp.from_int_and_width(2, self.visited_stack.type.element_type)
        storeOp = memref.StoreOp.get(c2, self.visited_stack, (current_depth,))

        trueBlock, falseBlock = Block(), Block()
        updated_restore = scf.IfOp(
            needs_restore, (needs_restore.type,), (trueBlock,), (falseBlock,)
        )
        self.handle_restore(needs_restore, current_depth, trueBlock, falseBlock, rewriter)

        # TODO: add simulation segment

        c1 = arith.ConstantOp.from_int_and_width(1, current_depth.type)
        updated_depth = arith.AddiOp(current_depth, c1)

        yieldOp = scf.YieldOp(updated_depth, updated_restore)

        for op in (c1, c2, storeOp, updated_restore, updated_depth, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(leftVisitedBlock))

        # handle right visited region: need to go back up
        c0 = arith.ConstantOp.from_int_and_width(0, self.visited_stack.type.element_type)
        storeOp = memref.StoreOp.get(c0, self.visited_stack, (current_depth,))  # erase tracks

        c1 = arith.ConstantOp.from_int_and_width(1, current_depth.type)
        updated_depth = arith.SubiOp(current_depth, c1)

        yieldOp = scf.YieldOp(updated_depth, needs_restore)

        for op in (c0, c1, storeOp, updated_depth, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(rightVisitedBlock))

        # handle default region, TODO: ideally should raise a runtime exception here
        cm1 = arith.ConstantOp.from_int_and_width(-1, current_depth.type)  # will end traversal
        yieldOp = scf.YieldOp(cm1, needs_restore)

        for op in (cm1, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(defaultBlock))

    def handle_restore(
        self,
        needs_restore: SSAValue,
        current_depth: SSAValue,
        trueBlock: Block,
        falseBlock: Block,
        rewriter: PatternRewriter,
    ):
        """Restore statevector to previous state when entering a new branch in the tree."""

        # true branch, restore statevector from the stack
        targetType = builtin.MemRefType(
            builtin.ComplexType(builtin.f64),
            (builtin.DYNAMIC_INDEX,),  # rank-reduce: get rid of leading size-1 dimension
            builtin.StridedLayoutAttr([1], None),  # needed due to dynamic indexing
        )
        statevec = memref.SubviewOp.get(
            self.statevec_stack, (current_depth, 0), (1, self.statevec_size), (1, 1), targetType
        )
        # TODO: do something with the statevector (quantum op)

        false = arith.ConstantOp.from_int_and_width(0, 1)
        yieldOp = scf.YieldOp(false)

        for op in (false, statevec, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(trueBlock))

        # false branch, no restore needed
        rewriter.insert_op(scf.YieldOp(needs_restore), InsertPoint.at_end(falseBlock))

    @staticmethod
    def sum_array(arg: SSAValue, rewriter: PatternRewriter, insert_into: Block) -> SSAValue:
        """Generate a sum reduction using the scf dialect. Produces an integer sum of index type."""
        assert isinstance(arg.type, builtin.MemRefType), "expected memref value to sum"
        assert len(arg.type.shape) == 1, "expected 1D memref to sum"
        assert isinstance(arg.type.element_type, builtin.IntegerType), "expected int memref to sum"
        ip_backup = rewriter.insertion_point
        rewriter.insertion_point = InsertPoint.at_end(insert_into)

        # TODO: reduce not available in either linalg or stablehlo dialects

        # loop instruction
        c0 = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
        c1 = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
        l = memref.DimOp.from_source_and_index(arg, c0)

        sum_init = arith.ConstantOp.from_int_and_width(0, builtin.i64)

        bodyBlock = Block(arg_types=(builtin.IndexType(), sum_init.result.type))
        reduced_sum = scf.ForOp(c0, l, c1, iter_args=(sum_init,), body=bodyBlock)
        casted_sum = arith.IndexCastOp(reduced_sum, builtin.IndexType())

        for op in (c0, c1, l, sum_init, reduced_sum, casted_sum):
            rewriter.insert(op)

        # loop body
        iter_index, last_sum = bodyBlock.args

        val = memref.LoadOp.get(arg, (iter_index,))
        casted_val = arith.ExtSIOp(val, last_sum.type)
        curr_sum = arith.AddiOp(casted_val, last_sum)
        yieldOp = scf.YieldOp(curr_sum)

        for op in (val, casted_val, curr_sum, yieldOp):
            rewriter.insert_op(op, InsertPoint.at_end(bodyBlock))

        rewriter.insertion_point = ip_backup
        return casted_sum


@xdsl_transform
class TTPass(ModulePass):
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
