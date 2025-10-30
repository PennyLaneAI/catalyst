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
from itertools import chain
from typing import Type, TypeVar, List, Tuple, Optional

import jax
import pennylane as qml
from pennylane.compiler.python_compiler import compiler_transform
from pennylane.compiler.python_compiler.dialects import quantum
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf, tensor
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern,  op_type_rewrite_pattern, PatternRewriteWalker
from xdsl.rewriter import BlockInsertPoint, InsertPoint

from xdsl.transforms.scf_for_loop_unroll import ScfForLoopUnrollPass #, UnrollLoopPattern
from pennylane.compiler.python_compiler.visualization.xdsl_conversion import  resolve_constant_params

from xdsl.printer import Printer

from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

T = TypeVar("T")

##############################################################################
# Some useful utils
##############################################################################
def get_parent_of_type(op: Operation, kind: Type[T]) -> T | None:
    """Walk up the parent tree until an op of the specified type is found."""
    while (op := op.parent_op()) and not isinstance(op, kind):
        pass
    return op


def print_mlir(op, msg="", should_print: bool = True):
    should_print = False
    if should_print:
        printer = Printer()
        print("-"*100)
        print(f"// Start || {msg}")
        if isinstance(op, Region):
            printer.print_region(op)
        elif isinstance(op, Block):
            printer.print_block(op)
        elif isinstance(op, Operation):
            printer.print_op(op)
        print(f"\n// End {msg}")
        print("-"*100)

def print_ssa_values(values, msg="SSA Values || ", should_print:bool = True):
    should_print = False
    if should_print:
        print(f"// {msg}")
        for val in values:
            print(f"  - {val}")

##############################################################################
# xDSL Transform If Operator Partitioning
##############################################################################

# A structure to hold the op and its depth during the search.
IfOpWithDepth = Tuple[scf.IfOp, int]
class IfOperatorPartitioningPass(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter) -> None:
        """Partition the if operation into separate branches for each operator."""

        self.original_func_op = op
        # self.duplicate_if_op(op, rewriter)
        # self.split_if_op(op, rewriter)
        # print("%"*120)
        # print("%"*120)
        # print("%"*120)
        self.split_nested_if_ops(op, rewriter)
        print_mlir(op, "After splitting nested IfOps:")
        print("%"*120)
        print("%"*120)
        print("%"*120)
        # self.flatten_if_ops(op, rewriter)

        print_mlir(op, "After flattening deeper IfOps:")
        self.find_deeper_if_ops(op, rewriter)
        print("+"*120)
        print("+"*120)
        print("+"*120)
        print_mlir(op, "After flattening deeper IfOps:")
        self.find_deeper_if_ops(op, rewriter)

        # Validate SSA values after flattening
        # self.validate_ssa_values(op)


    def __init__(self):
        self.module: builtin.ModuleOp = None
        self.original_func_op: func.FuncOp = None
        self.holder_returns : dict[scf.IfOp, scf.IfOp] = {}




    def _find_deepest_if_recursive(self,op: Operation, current_depth: int, max_depth_ops: List[IfOpWithDepth]) -> None:
        """
        Helper function to recursively traverse the IR, tracking the max depth
        of scf.If operations found so far.
        """

        # Iterate over all nested regions (then_region, else_region, etc.)
        for region in op.regions:
            for block in region.blocks:
                for child_op in block.ops:

                    new_depth = current_depth

                    if isinstance(child_op, scf.IfOp):
                        # Found an IfOp, increase the depth for the ops *inside* its regions.
                        # This IfOp itself is at 'current_depth + 1'.
                        new_depth = current_depth + 1

                        # --- Check and Update Max Depth List ---

                        # 1. Is this deeper than the current max? (First find or deeper op)
                        if not max_depth_ops or new_depth > max_depth_ops[0][1]:
                            # It's a new maximum depth! Clear the old list and start fresh.
                            max_depth_ops.clear()
                            max_depth_ops.append((child_op, new_depth))

                        # 2. Is this at the same depth as the current max? (A tie)
                        elif new_depth == max_depth_ops[0][1]:
                            # Add it to the list of winners.
                            max_depth_ops.append((child_op, new_depth))

                    # Recursively search inside this child op (regardless of its type)
                    # We pass the potentially *increased* new_depth.
                    self._find_deepest_if_recursive(child_op, new_depth, max_depth_ops)


    def get_deepest_nested_ifs(self, parent_if_op: scf.IfOp) -> IfOpWithDepth:
        """
        Finds the scf.if operation(s) nested at the maximum depth inside the parent_if_op.

        Args:
            parent_if_op: The scf.if operation to start the search from (e.g., IfOp A).

        Returns:
            A list of scf.if operations found at the deepest nesting level (e.g., [IfOp C]).
            If no nested IfOps are found, returns an empty list.
        """
        # The parent IfOp A is at depth 0, so its immediate children (B, D) are at depth 1.
        # We initialize the search list.
        deepest_ops_with_depth: List[IfOpWithDepth] = [(None, 0)]

        # Start the recursion. We look *inside* the regions of the parent_if_op.
        self._find_deepest_if_recursive(parent_if_op, 0, deepest_ops_with_depth)

        # Extract only the IfOp objects from the list of (IfOp, depth) tuples.
        # If the list is empty, this returns [].
        # return [op for op, depth in deepest_ops_with_depth]
        return deepest_ops_with_depth

    def find_deeper_if_ops(self, main_op: func.FuncOp, rewriter: PatternRewriter) -> None:
        # Find the deeper nested  if op
        op_walk = main_op.body.blocks[0].ops
        for current_op in op_walk:
            if isinstance(current_op, scf.IfOp):
                # has_nested_if_ops, nested_if_ops = self.get_nested_if_ops(current_op)
                # for nested in nested_if_ops:
                #     print_mlir(nested, "Nested if op")
                # # if has_nested_if_ops:
                # #     deeper_if_op = current_op
                # #     break


                deeper_with_depth = self.get_deepest_nested_ifs(current_op)

                depth = deeper_with_depth[0][1] #if deeper_with_depth else None

                if deeper_with_depth[0][0] is None:
                    break

                deeper_if_ops = [op for op, d in deeper_with_depth]


                for d in deeper_if_ops[0:1]:
                    # print_mlir(d, "Deeper IfOp found:")
                    # print_mlir(d.parent_op(), "Parent of Deeper IfOp found:")

                    self.flatten_if_ops_deep(d.parent_op(), rewriter)

                # print_mlir(main_op, "Main op after flattening deeper IfOps:")

                print(f"Deeper IfOp depth found: {depth}")

    def flatten_if_ops_deep(self, main_op: scf.IfOp, rewriter: PatternRewriter) -> None:

        print_output = True

        if isinstance(main_op, scf.IfOp):

            outer_if_op = main_op

            new_outer_if_op_output = [out for out in outer_if_op.results]
            new_outer_if_op_output_types = [out.type for out in outer_if_op.results]

            print_ssa_values(new_outer_if_op_output, "Outer IfOp outputs before flattening:", print_output)


            has_nested_if_ops, nested_if_ops = self.get_nested_if_ops(outer_if_op)
            where_to_insert = outer_if_op

            self.holder_returns = {}

            for inner_op in nested_if_ops:

                # if len(inner_op.results) > 1:
                #     continue
                print("/"*120)
                print_mlir(outer_if_op.parent_op(), "Before move_simple_inner_if_op_2_outer_test: Outer IfOp before flattening:", print_output)

                where_to_insert, outer_if_op = self.move_simple_inner_if_op_2_outer_test(
                                                inner_op, outer_if_op, new_outer_if_op_output, new_outer_if_op_output_types, where_to_insert, rewriter,
                                                )

            # detach and erase old outer if op
            for hold_op in self.holder_returns.keys():
                hold_op.detach()
                hold_op.erase()


        print_mlir(main_op,"Main op after creating new outer IfOp:", print_output)
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------


    def move_simple_inner_if_op_2_outer_test(self,
                                        inner_op: scf.IfOp, outer_if_op: scf.IfOp, new_outer_if_op_output: list[SSAValue], new_outer_if_op_output_types: list[Type], where_to_insert: scf.IfOp,
                                        rewriter: PatternRewriter,
                                        ) -> None:
        print_output = True

        print_mlir(inner_op, "Inner op before flattening:", print_output)

        missing_values_outer = self.analyze_missing_values_for_ops([outer_if_op])
        definition_outer = self.analyze_definitions_for_ops([outer_if_op])

        missing_values_inner = self.analyze_missing_values_for_ops([inner_op])

        print_ssa_values(missing_values_outer, "Missing values for outer IfOp before flattening inner IfOp:", print_output)
        print_ssa_values(definition_outer, "Definitions for outer IfOp before flattening inner IfOp:", print_output)
        print_ssa_values(missing_values_inner, "Missing values for inner IfOp before flattening:", print_output)

        ssa_needed_from_outer = set(missing_values_inner).intersection(set(definition_outer))
        print_ssa_values(ssa_needed_from_outer, "SSA values needed from outer IfOp before flattening inner IfOp:", print_output)

        # select only definition outer
        missing_values_inner = [mv for mv in missing_values_inner if mv in definition_outer]
        print_ssa_values(missing_values_outer, "Missing values for outer IfOp before flattening inner IfOp:", print_output)

        for mv in ssa_needed_from_outer:
            if not isinstance(mv.type, quantum.QuregType):
                new_outer_if_op_output.append(mv)
                new_outer_if_op_output_types.append(mv.type)

        print_ssa_values(new_outer_if_op_output, "New outer IfOp outputs after flattening inner IfOp:", print_output)
        print_ssa_values(new_outer_if_op_output_types, "New outer IfOp output types after flattening inner IfOp:", print_output)

        # Matching qreg
        required_outputs = outer_if_op.results
        print_ssa_values(required_outputs, "Outputs for outer IfOp:", print_output)

        inner_results = inner_op.results
        print_ssa_values(inner_results, "Outputs for inner IfOp:", print_output)

        # Replace the qreg from the inner IfOp with the immediate outer IfOp qreg
        # This dont affect the inner IfOp since its qreg is only used in quantum ops inside its regions
        qreg_if_op_inner = [mv for mv in missing_values_inner if isinstance(mv.type, quantum.QuregType)]

        for result in inner_results:
            if isinstance(result.type, quantum.QuregType):
                result.replace_by(qreg_if_op_inner[0])

        qreg_if_op_outer = [output for output in where_to_insert.results if isinstance(output.type, quantum.QuregType)]

        assert len(qreg_if_op_outer) == 1, "Expected exactly one quantum register in outer IfOp results."

        # print_mlir(outer_if_op, "Outer IfOp before flattening inner IfOp:", print_output)
        # print_mlir(inner_op, "Inner IfOp before flattening inner IfOp:", print_output)

        # print_mlir(outer_if_op.parent_op(), "Main Op before flattening inner IfOp:", print_output)


        # holder_returns = {}

        if len(inner_results) == 1 :
            inner_op.detach()
        else:
            # add a new attribute to mark it as flattened
            inner_op.attributes["old_return"] = builtin.StringAttr("true")

        # Create comprehensive value mapping for all values used in both regions
        value_mapper = {}
        value_mapper[qreg_if_op_inner[0]] = qreg_if_op_outer[0]

        # expand the current attr_dict
        attr_dict = inner_op.attributes.copy()
        attr_dict.update({"flattened": builtin.StringAttr("true")})

        ###########################################################
        # new_inner_op = self.create_if_op_partition(
        #     rewriter,
        #     inner_op.true_region,
        #     where_to_insert,
        #     value_mapper,
        #     where_to_insert,
        #     conditional=inner_op.cond,  # Use the original condition
        #     attr_dict=attr_dict
        # )

        if_region = inner_op.true_region

        block = if_region.blocks

        true_ops = [op for op in if_region.blocks[0].ops]

        new_true_block = Block()

        self.clone_operations_to_block(
            true_ops,
            new_true_block,
            value_mapper
        )

        # --------------------------------------------------------------------------

        false_inner_ops = [op for op in inner_op.false_region.blocks[0].ops]

        new_false_block = None

        if len(false_inner_ops) == 1 and isinstance(false_inner_ops[0], scf.YieldOp):
            # If the false region only contains a yield operation, we can create an empty block

            # Create a new empty block for false region
            new_false_block = Block()

            # Create a yield operation for false region using the same return types as the original IfOp
            # yield_false = scf.YieldOp(previous_IfOp.results[0])
            yield_false = scf.YieldOp(where_to_insert.results[0])

            # Create a new empty block for false region
            new_false_block.add_op(yield_false)

        else:
            false_block_inner = inner_op.false_region.detach_block(0)
            false_ops = [op for op in false_block_inner.ops]

            new_false_block = Block()

            value_mapper = {qreg_if_op_inner[0]: qreg_if_op_outer[0]}
            self.clone_operations_to_block(
                false_ops,
                new_false_block,
                value_mapper
            )


        new_if_op_attrs = where_to_insert.attributes.copy()
        # new_if_op_attrs = inner_op.attributes.copy()
        new_if_op_attrs.update(attr_dict or {})
        # --------------------------------------------------------------------------
        # Create new IfOp with cloned regions
        # scf.IfOp (
        # cond: SSAValue | Operation,
        # return_types: Sequence[Attribute],
        # true_region: Region | Sequence[Block] | Sequence[Operation],
        # false_region: Region | Sequence[Block] | Sequence[Operation] | None = None,
        # attr_dict: dict[str, Attribute] | None = None,
        # )

        # if conditional is None:
        #     conditional = previous_IfOp.cond

        needs_to_update_conditional = True

        if inner_op.cond.owner.attributes.get("old_return",None) is not None and isinstance(inner_op.cond.owner, scf.IfOp):
            hold_return = inner_op.cond.owner
            return_index = list(hold_return.results).index(inner_op.cond)
            conditional = self.holder_returns[hold_return].results[return_index]
            needs_to_update_conditional = False

            for res in hold_return.results:
                if res in missing_values_inner:
                    remove_index = missing_values_inner.index(res)
                    missing_values_inner.pop(remove_index)

        else:
            conditional = inner_op.cond

        new_inner_op = scf.IfOp(
            conditional,
            # previous_IfOp.result_types,
            # where_to_insert.result_types,
            inner_op.result_types,
            [new_true_block], # cloned_true_region,
            [new_false_block], # false_region,
            new_if_op_attrs
        )
        rewriter.insert_op(new_inner_op, InsertPoint.after(where_to_insert))

        new_inner_op_ops = list(chain(*[op.walk() for op in [new_inner_op]]))

        where_to_insert.results[0].replace_by_if(new_inner_op.results[0], lambda use: use.operation not in new_inner_op_ops)

        # print_mlir(inner_op, "Old Inner IfOp after flattening:", print_output)
        # print_mlir(new_inner_op.parent_op(), "New Inner IfOp after flattening:", print_output)

        where_to_insert = new_inner_op
        if len(inner_results) == 1 :
            inner_op.erase()
        else:
            self.holder_returns[inner_op] = new_inner_op
        ################################################################3
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------

        # Create a new outer IfOp that includes the new outputs needed from the inner IfOp


        # True block --------------------------
        # return_types = new_outer_if_op_output_types

        true_block = outer_if_op.true_region.detach_block(0)

        true_yield_op = [op for op in true_block.ops if isinstance(op, scf.YieldOp)][-1]

        print_mlir(true_yield_op, "True Yield Op for outer IfOp", print_output)

        new_res = [res for res in true_yield_op.operands] + [ ssa for ssa in  missing_values_inner if not isinstance(ssa.type, quantum.QuregType) ]
        return_types = [new_r.type for new_r in new_res]

        new_true_yield_op = scf.YieldOp( *new_res )

        rewriter.replace_op(true_yield_op, new_true_yield_op)

        print_mlir(new_true_yield_op, "New True Yield Op for outer IfOp", print_output)

        # False block --------------------------
        false_block = outer_if_op.false_region.detach_block(0)
        # false_block = Block()
        # create a false value

        false_op_res = []

        if needs_to_update_conditional:
            false_op = arith.ConstantOp(builtin.IntegerAttr(0, builtin.IntegerType(1)))
            false_op_res.append(false_op.result)
            rewriter.insert_op(false_op, InsertPoint.at_start(false_block))

        false_yield_op = [op for op in false_block.ops if isinstance(op, scf.YieldOp)][-1]

        new_res = [res for res in false_yield_op.operands] + false_op_res

        new_false_yield_op = scf.YieldOp( *new_res )

        rewriter.replace_op(false_yield_op, new_false_yield_op)

        # add a the top of the block

        new_outer_if_op = scf.IfOp(
            outer_if_op.cond,
            return_types,
            [true_block],
            [false_block],
            outer_if_op.attributes.copy()
        )

        rewriter.insert_op(new_outer_if_op, InsertPoint.before(outer_if_op))

        print_ssa_values(outer_if_op.results,"old outer_if results", print_output)
        print_ssa_values(new_outer_if_op.results,"new outer_if results", print_output)

        for old_result, new_result in zip(outer_if_op.results, new_outer_if_op.results):
            old_result.replace_by(new_result)
        # outer_if_op.results[0].replace_by(new_outer_if_op.results[0])

        outer_if_op.detach()

        outer_if_op.erase()

        outer_if_op = new_outer_if_op

        # new_inner_op.cond = outer_if_op.results[-1]
        if needs_to_update_conditional:
            new_cond = new_inner_op.cond
            new_cond.replace_by_if(outer_if_op.results[-1], lambda use: use.operation in [new_inner_op])

        return where_to_insert, outer_if_op



    def move_complex_inner_if_op_2_outer(self,
                                        inner_op: scf.IfOp, outer_if_op: scf.IfOp, new_outer_if_op_output: list[SSAValue], new_outer_if_op_output_types: list[Type], where_to_insert: scf.IfOp,
                                        rewriter: PatternRewriter) -> None:
        print_output = True

        print_mlir(inner_op, "Inner op before flattening:", print_output)

        missing_values_outer = self.analyze_missing_values_for_ops([outer_if_op])
        definition_outer = self.analyze_definitions_for_ops([outer_if_op])

        missing_values_inner = self.analyze_missing_values_for_ops([inner_op])

        print_ssa_values(missing_values_outer, "Missing values for outer IfOp before flattening inner IfOp:", print_output)
        print_ssa_values(definition_outer, "Definitions for outer IfOp before flattening inner IfOp:", print_output)
        print_ssa_values(missing_values_inner, "Missing values for inner IfOp before flattening:", print_output)

        ssa_needed_from_outer = set(missing_values_inner).intersection(set(definition_outer))
        print_ssa_values(ssa_needed_from_outer, "SSA values needed from outer IfOp before flattening inner IfOp:", print_output)

        for mv in ssa_needed_from_outer:
            if not isinstance(mv.type, quantum.QuregType):
                new_outer_if_op_output.append(mv)
                new_outer_if_op_output_types.append(mv.type)

        print_ssa_values(new_outer_if_op_output, "New outer IfOp outputs after flattening inner IfOp:", print_output)
        print_ssa_values(new_outer_if_op_output_types, "New outer IfOp output types after flattening inner IfOp:", print_output)

        print_mlir(outer_if_op, "Outer IfOp before flattening inner IfOp:", print_output)
        # Matching qreg
        required_outputs = outer_if_op.results
        print_ssa_values(required_outputs, "Outputs for outer IfOp:", print_output)

        inner_results = inner_op.results

        qreg_if_op_inner = [mv for mv in missing_values_inner if isinstance(mv.type, quantum.QuregType)]

        for result in inner_results:
            if isinstance(result.type, quantum.QuregType):
                result.replace_by(qreg_if_op_inner[0])

        qreg_if_op_outer = [output for output in outer_if_op.results if isinstance(output.type, quantum.QuregType)]

        assert len(qreg_if_op_outer) == 1, "Expected exactly one quantum register in outer IfOp results."

        print_mlir(outer_if_op, "Outer IfOp before flattening inner IfOp:", print_output)

        # Trying to upate the yield op to detach the inner if op

        yield_operands_outer_from_inner = []
        yield_op_outer = None

        remove_new_outer_if_op_output = []

        for res in inner_op.results:
            if res.first_use is None:
                continue
            if isinstance(res.first_use.operation, scf.YieldOp):
                yield_operands_outer_from_inner.append(res)
                yield_op_outer = res.first_use.operation

                l_yield_operands = list(yield_op_outer.operands)
                remove_new_outer_if_op_output.append(l_yield_operands.index(res))

        for remove_item in remove_new_outer_if_op_output:
            new_outer_if_op_output.pop(remove_item)
            new_outer_if_op_output_types.pop(remove_item)

        reminder_operands = [op for op in yield_op_outer.operands if op not in yield_operands_outer_from_inner]

        new_yield = scf.YieldOp( *reminder_operands )
        rewriter.replace_op(yield_op_outer, new_yield)

        print_mlir(outer_if_op, "Outer IfOp before flattening inner IfOp:", print_output)
        inner_op.detach()

        # Create comprehensive value mapping for all values used in both regions
        value_mapper = {}
        value_mapper[qreg_if_op_inner[0]] = qreg_if_op_outer[0]

        # expand the current attr_dict
        attr_dict = inner_op.attributes.copy()
        attr_dict.update({"flattened": builtin.StringAttr("true")})

        ###########################################################
        new_inner_op = self.create_if_op_partition(
            rewriter,
            inner_op.true_region,
            where_to_insert,
            value_mapper,
            where_to_insert,
            conditional=inner_op.cond,  # Use the original condition
            attr_dict=attr_dict
        )

        where_to_insert = new_inner_op
        inner_op.erase()
        ################################################################3
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------

        # Create a new outer IfOp that includes the new outputs needed from the inner IfOp

        print_mlir(outer_if_op, "Outer IfOp before flattening inner IfOp:", print_output)
        print_mlir(inner_op, "Outer IfOp before flattening inner IfOp:", print_output)


        # True block --------------------------
        return_types = new_outer_if_op_output_types

        true_block = outer_if_op.true_region.detach_block(0)

        true_yield_op = [op for op in true_block.ops if isinstance(op, scf.YieldOp)][-1]

        print_mlir(true_yield_op, "True Yield Op for outer IfOp", print_output)

        new_res = [res for res in true_yield_op.operands] + [missing_values_inner[0]]

        new_true_yield_op = scf.YieldOp( *new_res )

        rewriter.replace_op(true_yield_op, new_true_yield_op)

        print_mlir(new_true_yield_op, "New True Yield Op for outer IfOp", print_output)

        # False block --------------------------
        false_block = outer_if_op.false_region.detach_block(0)
        # false_block = Block()
        # create a false value
        false_op = arith.ConstantOp(builtin.IntegerAttr(0, builtin.IntegerType(1)))

        false_yield_op = [op for op in false_block.ops if isinstance(op, scf.YieldOp)][-1]

        new_res = [res for res in false_yield_op.operands] + [false_op.result]

        new_false_yield_op = scf.YieldOp( *new_res )

        rewriter.replace_op(false_yield_op, new_false_yield_op)

        # add a the top of the block
        rewriter.insert_op(false_op, InsertPoint.at_start(false_block))

        new_outer_if_op = scf.IfOp(
            outer_if_op.cond,
            return_types,
            [true_block],
            [false_block],
            outer_if_op.attributes.copy()
        )

        rewriter.insert_op(new_outer_if_op, InsertPoint.before(outer_if_op))

        print_ssa_values(outer_if_op.results,"old outer_if results", print_output)
        print_ssa_values(new_outer_if_op.results,"new outer_if results", print_output)

        for old_result, new_result in zip(outer_if_op.results, new_outer_if_op.results):
            old_result.replace_by(new_result)
        # outer_if_op.results[0].replace_by(new_outer_if_op.results[0])

        outer_if_op.detach()

        outer_if_op.erase()

        outer_if_op = new_outer_if_op

        # new_inner_op.cond = outer_if_op.results[-1]
        new_cond = new_inner_op.cond
        new_cond.replace_by_if(outer_if_op.results[-1], lambda use: use.operation in [new_inner_op])

        return where_to_insert, outer_if_op



    def move_simple_inner_if_op_2_outer(self,
                                        inner_op: scf.IfOp, outer_if_op: scf.IfOp, new_outer_if_op_output: list[SSAValue], new_outer_if_op_output_types: list[Type], where_to_insert: scf.IfOp,
                                        rewriter: PatternRewriter) -> None:
        print_output = True

        print_mlir(inner_op, "Inner op before flattening:", print_output)

        missing_values_outer = self.analyze_missing_values_for_ops([outer_if_op])
        definition_outer = self.analyze_definitions_for_ops([outer_if_op])

        missing_values_inner = self.analyze_missing_values_for_ops([inner_op])

        print_ssa_values(missing_values_outer, "Missing values for outer IfOp before flattening inner IfOp:", print_output)
        print_ssa_values(definition_outer, "Definitions for outer IfOp before flattening inner IfOp:", print_output)
        print_ssa_values(missing_values_inner, "Missing values for inner IfOp before flattening:", print_output)

        ssa_needed_from_outer = set(missing_values_inner).intersection(set(definition_outer))
        print_ssa_values(ssa_needed_from_outer, "SSA values needed from outer IfOp before flattening inner IfOp:", print_output)

        for mv in ssa_needed_from_outer:
            if not isinstance(mv.type, quantum.QuregType):
                new_outer_if_op_output.append(mv)
                new_outer_if_op_output_types.append(mv.type)

        print_ssa_values(new_outer_if_op_output, "New outer IfOp outputs after flattening inner IfOp:", print_output)
        print_ssa_values(new_outer_if_op_output_types, "New outer IfOp output types after flattening inner IfOp:", print_output)

        print_mlir(outer_if_op, "Outer IfOp before flattening inner IfOp:", print_output)
        # Matching qreg
        required_outputs = outer_if_op.results
        print_ssa_values(required_outputs, "Outputs for outer IfOp:", print_output)

        inner_results = inner_op.results

        qreg_if_op_inner = [mv for mv in missing_values_inner if isinstance(mv.type, quantum.QuregType)]

        for result in inner_results:
            if isinstance(result.type, quantum.QuregType):
                result.replace_by(qreg_if_op_inner[0])

        qreg_if_op_outer = [output for output in outer_if_op.results if isinstance(output.type, quantum.QuregType)]

        assert len(qreg_if_op_outer) == 1, "Expected exactly one quantum register in outer IfOp results."

        print_mlir(outer_if_op, "Outer IfOp before flattening inner IfOp:", print_output)


        inner_op.detach()

        # Create comprehensive value mapping for all values used in both regions
        value_mapper = {}
        value_mapper[qreg_if_op_inner[0]] = qreg_if_op_outer[0]

        # expand the current attr_dict
        attr_dict = inner_op.attributes.copy()
        attr_dict.update({"flattened": builtin.StringAttr("true")})

        ###########################################################
        new_inner_op = self.create_if_op_partition(
            rewriter,
            inner_op.true_region,
            where_to_insert,
            value_mapper,
            where_to_insert,
            conditional=inner_op.cond,  # Use the original condition
            attr_dict=attr_dict
        )

        where_to_insert = new_inner_op
        inner_op.erase()
        ################################################################3
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------
        # ----------------------------------------------------------------

        # Create a new outer IfOp that includes the new outputs needed from the inner IfOp


        # True block --------------------------
        return_types = new_outer_if_op_output_types

        true_block = outer_if_op.true_region.detach_block(0)

        true_yield_op = [op for op in true_block.ops if isinstance(op, scf.YieldOp)][-1]

        print_mlir(true_yield_op, "True Yield Op for outer IfOp", print_output)

        new_res = [res for res in true_yield_op.operands] + [missing_values_inner[0]]

        new_true_yield_op = scf.YieldOp( *new_res )

        rewriter.replace_op(true_yield_op, new_true_yield_op)

        print_mlir(new_true_yield_op, "New True Yield Op for outer IfOp", print_output)

        # False block --------------------------
        false_block = outer_if_op.false_region.detach_block(0)
        # false_block = Block()
        # create a false value
        false_op = arith.ConstantOp(builtin.IntegerAttr(0, builtin.IntegerType(1)))

        false_yield_op = [op for op in false_block.ops if isinstance(op, scf.YieldOp)][-1]

        new_res = [res for res in false_yield_op.operands] + [false_op.result]

        new_false_yield_op = scf.YieldOp( *new_res )

        rewriter.replace_op(false_yield_op, new_false_yield_op)

        # add a the top of the block
        rewriter.insert_op(false_op, InsertPoint.at_start(false_block))

        new_outer_if_op = scf.IfOp(
            outer_if_op.cond,
            return_types,
            [true_block],
            [false_block],
            outer_if_op.attributes.copy()
        )

        rewriter.insert_op(new_outer_if_op, InsertPoint.before(outer_if_op))

        print_ssa_values(outer_if_op.results,"old outer_if results", print_output)
        print_ssa_values(new_outer_if_op.results,"new outer_if results", print_output)

        for old_result, new_result in zip(outer_if_op.results, new_outer_if_op.results):
            old_result.replace_by(new_result)
        # outer_if_op.results[0].replace_by(new_outer_if_op.results[0])

        outer_if_op.detach()

        outer_if_op.erase()

        outer_if_op = new_outer_if_op

        # new_inner_op.cond = outer_if_op.results[-1]
        new_cond = new_inner_op.cond
        new_cond.replace_by_if(outer_if_op.results[-1], lambda use: use.operation in [new_inner_op])

        return where_to_insert, outer_if_op


    def flatten_if_ops(self, main_op: func.FuncOp, rewriter: PatternRewriter) -> None:
        number_if_op = 0

        op_walk = main_op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.IfOp):
                number_if_op += 1

        print(f"If_op: {number_if_op}")

        op_walk = main_op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.IfOp):

                outer_if_op = current_op

                new_outer_if_op_output = [out for out in outer_if_op.results]
                new_outer_if_op_output_types = [out.type for out in outer_if_op.results]

                has_nested_if_ops, nested_if_ops = self.get_nested_if_ops(outer_if_op)
                where_to_insert = outer_if_op

                for inner_op in nested_if_ops:

                    # Move the inner if-statement to be directly after the outer if-statement
                    missing_values = self.analyze_missing_values_for_ops([inner_op])
                    print_ssa_values(missing_values, "Missing values for nested IfOp:")

                    # Adding the missing value to new outer IfOp outputs and output types
                    for mv in missing_values[0:1]:
                        if mv not in new_outer_if_op_output:
                            new_outer_if_op_output.append(mv)
                            new_outer_if_op_output_types.append(mv.type)

                    required_outputs = self.analyze_required_outputs([outer_if_op], outer_if_op.next_op)
                    print_ssa_values(required_outputs, "Required outputs for nested IfOp:")

                    print_mlir(inner_op,"Inner op before flattening")

                    # Get quantum register from missing values BEFORE detaching
                    qreg_if_op = [mv for mv in missing_values if isinstance(mv.type, quantum.QuregType)]

                    # Find the quantum register that should be the input to the new IfOp
                    # This should be the quantum register that comes from the outer IfOp's true branch
                    input_quantum_reg = None
                    for req_output in required_outputs:
                        if isinstance(req_output.type, quantum.QuregType):
                            input_quantum_reg = req_output
                            break

                    original_if_op_results = inner_op.results[0]
                    original_if_op_results.replace_by(qreg_if_op[0])


                    inner_op.detach()
                    # remmeber to delete the detachec inner_op

                    # Create comprehensive value mapping for all values used in both regions
                    value_mapper = {}

                    # Map the problematic quantum register reference to the correct input
                    value_mapper[qreg_if_op[0]] = input_quantum_reg

                    # expand the current attr_dict
                    attr_dict = inner_op.attributes.copy()
                    attr_dict.update({"flattened": builtin.StringAttr("true")})


                    ###########################################################
                    new_inner_op = self.create_if_op_partition(
                        rewriter,
                        inner_op.true_region,
                        where_to_insert,
                        value_mapper,
                        where_to_insert,
                        conditional=inner_op.cond,  # Use the original condition
                        attr_dict=attr_dict
                    )

                    where_to_insert = new_inner_op
                    inner_op.erase()
                    ################################################################3
                    # ----------------------------------------------------------------
                    # ----------------------------------------------------------------
                    # ----------------------------------------------------------------

                    # Create a new outer IfOp that includes the new outputs needed from the inner IfOp


                    # True block --------------------------
                    return_types = new_outer_if_op_output_types

                    true_block = outer_if_op.true_region.detach_block(0)

                    true_yield_op = [op for op in true_block.ops if isinstance(op, scf.YieldOp)][-1]

                    print_mlir(true_yield_op, "True Yield Op")

                    new_res = [res for res in true_yield_op.operands] + [missing_values[0]]

                    new_true_yield_op = scf.YieldOp( *new_res )

                    rewriter.replace_op(true_yield_op, new_true_yield_op)

                    print_mlir(new_true_yield_op, "New True Yield Op")

                    # False block --------------------------
                    false_block = outer_if_op.false_region.detach_block(0)
                    # false_block = Block()
                    # create a false value
                    false_op = arith.ConstantOp(builtin.IntegerAttr(0, builtin.IntegerType(1)))

                    false_yield_op = [op for op in false_block.ops if isinstance(op, scf.YieldOp)][-1]

                    new_res = [res for res in false_yield_op.operands] + [false_op.result]

                    new_false_yield_op = scf.YieldOp( *new_res )

                    rewriter.replace_op(false_yield_op, new_false_yield_op)

                    # add a the top of the block
                    rewriter.insert_op(false_op, InsertPoint.at_start(false_block))

                    new_outer_if_op = scf.IfOp(
                        outer_if_op.cond,
                        return_types,
                        [true_block],
                        [false_block],
                        outer_if_op.attributes.copy()
                    )

                    rewriter.insert_op(new_outer_if_op, InsertPoint.before(outer_if_op))

                    print_ssa_values(outer_if_op.results,"old outer_if results")
                    print_ssa_values(new_outer_if_op.results,"new outer_if results")

                    for old_result, new_result in zip(outer_if_op.results, new_outer_if_op.results):
                        old_result.replace_by(new_result)
                    # outer_if_op.results[0].replace_by(new_outer_if_op.results[0])

                    print_mlir(main_op, "Before remove Outer_if Op")

                    outer_if_op.detach()

                    outer_if_op.erase()

                    outer_if_op = new_outer_if_op

                    # new_inner_op.cond = outer_if_op.results[-1]
                    new_cond = new_inner_op.cond
                    new_cond.replace_by_if(outer_if_op.results[-1], lambda use: use.operation in [new_inner_op])

                    print_mlir(main_op,"Main op after creating new outer IfOp:")
                    # ----------------------------------------------------------------
                    # ----------------------------------------------------------------
                    # ----------------------------------------------------------------



        missing_values = self.analyze_missing_values_for_ops([main_op])
        print_ssa_values(missing_values, "Missing values for nested IfOp:")

    def validate_ssa_values(self, op: func.FuncOp) -> None:
        """Validate that all SSA values are properly defined before use."""
        print("="*120)
        print("VALIDATING SSA VALUES")
        print("="*120)

        defined_values = set()

        # Walk through all operations in order
        for current_op in op.walk():
            # print(f"Checking operation: {current_op}")

            # Check operands (values used by this operation)
            for i, operand in enumerate(current_op.operands):
                if operand not in defined_values:
                    print(f"// ERROR: Operation")
                    print(f"  {current_op}")
                    print(f"// uses undeclared SSA value ")
                    print(f"  {operand} ")
                    print(f"// operand index {i}")
                    print("//" + "-"*80)
                    # print(f"Index operation is {self.get_idx(current_op)}")
                    # print(f"Operation type: {type(current_op)}")
                    # print(f"Available defined values: {[str(v) for v in defined_values]}")
                    # print(f"All operands: {current_op.operands}")

                    # Try to find where this value should have been defined
                    self.find_missing_definition(operand, op)

            # Add this operation's results to defined values
            for result in current_op.results:
                defined_values.add(result)
                # print(f"  Defined: {result}")

            # Handle block arguments for regions
            if hasattr(current_op, 'regions') and current_op.regions:
                for region in current_op.regions:
                    for block in region.blocks:
                        for arg in block.args:
                            defined_values.add(arg)
                            print(f"  Block arg defined: {arg}")

        print("="*120)
        print("SSA VALIDATION COMPLETE")
        print("="*120)

    def find_missing_definition(self, missing_value: SSAValue, func_op: func.FuncOp) -> None:
        """Try to find where a missing SSA value should have been defined."""
        print(f"Searching for definition of missing value: {missing_value}")

        # Look for operations that might produce this value
        for op in func_op.walk():
            for result in op.results:
                if str(result) == str(missing_value):
                    print(f"Found potential definition in: {op}")
                    print(f"But this operation might have been detached or erased")
                    return

        print(f"No definition found for {missing_value} - this value was likely from a detached/erased operation")

    def get_nested_if_ops(self, op: scf.IfOp) -> tuple[bool, list[scf.IfOp]]:
        nested_if_ops = []
        # Only check the immediate operations in the true region (not nested deeper)
        for inner_op in op.true_region.block.ops:
            if isinstance(inner_op, scf.IfOp):
                nested_if_ops.append(inner_op)
        # Only check the immediate operations in the false region (not nested deeper)
        for inner_op in op.false_region.block.ops:
            if isinstance(inner_op, scf.IfOp):
                nested_if_ops.append(inner_op)
        return len(nested_if_ops) > 0, nested_if_ops

    def looking_for_nested_if_ops(self, op: scf.IfOp) -> bool:
        for inner_op in op.true_region.ops:
            if isinstance(inner_op, scf.IfOp):
                return True
        for inner_op in op.false_region.ops:
            if isinstance(inner_op, scf.IfOp):
                return True
        return False


    def split_nested_if_ops(self, op: func.FuncOp, rewriter: PatternRewriter, go_deeper: bool = False) -> None:


        # print_mlir(op, "Processing scf.IfOp:")
        if go_deeper and isinstance(op, scf.IfOp):

            # print_mlir(op, "Processing scf.IfOp with go_deeper=True:")
            # Process true region
            true_region = op.true_region
            for inner_op in true_region.ops:
                if isinstance(inner_op, scf.IfOp):
                    have_nested_if_ops = self.looking_for_nested_if_ops(inner_op)

                    print(f"have_nested_if_ops on True: {have_nested_if_ops}")

                    if have_nested_if_ops:
                        self.split_nested_if_ops(inner_op, rewriter, go_deeper=True)
                        self.split_if_op(inner_op, rewriter)
                    if not have_nested_if_ops:
                        self.split_if_op(inner_op, rewriter)

            # Process false region
            false_region = op.false_region
            for inner_op in false_region.ops:
                if isinstance(inner_op, scf.IfOp):
                    have_nested_if_ops = self.looking_for_nested_if_ops(inner_op)

                    print(f"have_nested_if_ops on False: {have_nested_if_ops}")

                    if have_nested_if_ops:
                        self.split_nested_if_ops(inner_op, rewriter, go_deeper=True)
                        self.split_if_op(inner_op, rewriter)
                    if not have_nested_if_ops:
                        self.split_if_op(inner_op, rewriter)
            return

        op_walk = op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.IfOp):

                have_nested_if_ops = self.looking_for_nested_if_ops(current_op)

                print(f"have_nested_if_ops: {have_nested_if_ops}")

                if have_nested_if_ops:
                    self.split_nested_if_ops(current_op, rewriter, go_deeper=True)
                    self.split_if_op(current_op, rewriter)

                if not have_nested_if_ops:
                    self.split_if_op(current_op, rewriter)


    def split_if_op(self, op: func.FuncOp, rewriter: PatternRewriter) -> None:
        # Find scf.IfOp

        # print_mlir(op, "Processing scf.IfOp:")

        print_mid_step = False

        op_walk = op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.IfOp):

                print_mlir(current_op, "Processing scf.IfOp:", print_mid_step)

                # Analyze missing values for the IfOp
                missing_values = self.analyze_missing_values_for_ops([current_op])
                print_ssa_values(missing_values, "Missing values for IfOp:", print_mid_step)

                # Get outputs required by operations after the IfOp
                required_outputs = self.analyze_required_outputs(
                    [current_op], current_op.next_op
                )
                print_ssa_values(required_outputs, "Required outputs after IfOp:", print_mid_step)

                # Get quantum register from missing values
                qreg_if_op = [mv for mv in missing_values if isinstance(mv.type, quantum.QuregType)]
                print_ssa_values(qreg_if_op, "Quantum register for IfOp:", print_mid_step)

                return_vals_if_op = [ro for ro in required_outputs if ro in current_op.results]
                print_ssa_values(return_vals_if_op, "Return values for IfOp:", print_mid_step)

                # True and False regions
                true_region = current_op.true_region
                false_region = current_op.false_region

                print_mlir(true_region, "True Region:", print_mid_step)
                # print("Block Args True Region:", true_region.blocks[0].arg_types)
                print_mlir(false_region, "False Region:", print_mid_step)
                # print("Block Args False Region:", false_region.blocks[0].arg_types)


                # --------------------------------------------------------------------------
                # New partitioning logic for True region
                # --------------------------------------------------------------------------

                # value_mapper = {qreg_if_op[0]: current_op.results[0]}
                value_mapper = {}

                attr_dict = {"partition": builtin.StringAttr("true_branch")}

                new_if_op_4_true = self.create_if_op_partition(
                    rewriter,
                    true_region,
                    current_op,
                    value_mapper,
                    current_op,
                    attr_dict=attr_dict
                )

                print_mlir(new_if_op_4_true, "New IfOp for True Branch:", print_mid_step)

                # --------------------------------------------------------------------------
                # New partitioning logic for False region
                # --------------------------------------------------------------------------
                # Add the negation of the condition to the false branch if needed

                true_op = arith.ConstantOp(builtin.IntegerAttr(1, builtin.IntegerType(1)))
                not_op = arith.XOrIOp(current_op.cond, true_op.result)

                # Insert not_op after new_if_op
                for new_op in [not_op, true_op]:
                    rewriter.insert_op(new_op, InsertPoint.after(new_if_op_4_true))

                value_mapper = {qreg_if_op[0]: new_if_op_4_true.results[0]}

                attr_dict = {"partition": builtin.StringAttr("false_branch")}

                new_if_op_4_false = self.create_if_op_partition(
                    rewriter,
                    false_region,
                    new_if_op_4_true,
                    value_mapper,
                    not_op,
                    conditional=not_op.result,
                    attr_dict=attr_dict
                )
                print_mlir(new_if_op_4_false, "New IfOp for False Branch:", print_mid_step)

                # --------------------------------------------------------------------------

                print_mlir(op, "Function after IfOp Partitioning:", print_mid_step)

                original_if_op_results = current_op.results[0]
                original_if_op_results.replace_by(qreg_if_op[0])

                list_op_if = [curr_op for curr_op in current_op.walk()]
                # Remove the ops in the original IfOp
                for if_op in list_op_if[::-1]:
                    # print_mlir(op, "Erasing original IfOp ops:")
                    if_op.detach()
                    if_op.erase()

                print_mlir(op, "Function after remove original If", print_mid_step)


    def create_if_op_partition(self,
                               rewriter: PatternRewriter,
                               if_region: Region,
                               previous_IfOp: scf.IfOp,
                               value_mapper: dict[SSAValue, SSAValue],
                               op_where_insert_after: Operation,
                               conditional: SSAValue = None,
                               attr_dict: dict[str, builtin.Attribute] = None
                               ) -> scf.IfOp:


        block = if_region.blocks

        true_ops = [op for op in if_region.blocks[0].ops]

        new_true_block = Block()

        self.clone_operations_to_block(
            true_ops,
            new_true_block,
            value_mapper
        )

        # --------------------------------------------------------------------------
        # Create a new empty block for false region
        new_false_block = Block()

        # Create a yield operation for false region using the same return types as the original IfOp
        yield_false = scf.YieldOp(previous_IfOp.results[0])

        # Create a new empty block for false region
        new_false_block.add_op(yield_false)

        new_if_op_attrs = previous_IfOp.attributes.copy()
        new_if_op_attrs.update(attr_dict or {})
        # --------------------------------------------------------------------------
        # Create new IfOp with cloned regions
        # scf.IfOp (
        # cond: SSAValue | Operation,
        # return_types: Sequence[Attribute],
        # true_region: Region | Sequence[Block] | Sequence[Operation],
        # false_region: Region | Sequence[Block] | Sequence[Operation] | None = None,
        # attr_dict: dict[str, Attribute] | None = None,
        # )

        if conditional is None:
            conditional = previous_IfOp.cond


        new_if_op_4_true = scf.IfOp(
            conditional,
            previous_IfOp.result_types,
            [new_true_block], # cloned_true_region,
            [new_false_block], # false_region,
            new_if_op_attrs
        )
        rewriter.insert_op(new_if_op_4_true, InsertPoint.after(op_where_insert_after))

        new_if_op_4_true_ops = list(chain(*[op.walk() for op in [new_if_op_4_true]]))

        previous_IfOp.results[0].replace_by_if(new_if_op_4_true.results[0], lambda use: use.operation not in new_if_op_4_true_ops)

        return new_if_op_4_true


    def duplicate_if_op(self, op: func.FuncOp, rewriter: PatternRewriter) -> None:
        # Find scf.IfOp
        print_mlir(op, "Found scf.IfOp:")

        op_walk = op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.IfOp):
                print_mlir(current_op, "Processing scf.IfOp:")

                # Here you can implement the partitioning logic
                # For demonstration, we will just print the then and else regions
                true_region = current_op.true_region
                false_region = current_op.false_region

                print_mlir(true_region, "True Region:")
                print_mlir(false_region, "False Region:")

                # You can add your partitioning logic here
                # clone the If op  before the current_op

                missing_values = self.analyze_missing_values_for_ops([current_op])
                print(f"Missing values for IfOp: {missing_values}")

                quantum_register = None

                for missing_value in missing_values:
                    if isinstance(missing_value.op, quantum.InsertOp):
                        quantum_register = missing_value

                value_mapper = { quantum_register: current_op.results[0] }


                cloned_if_op = current_op.clone(value_mapper)
                rewriter.insert_op(cloned_if_op, InsertPoint.after(current_op))

                cloned_if_op_ops = list(chain(*[op.walk() for op in [cloned_if_op]]))

                new_missing_values = self.analyze_missing_values_for_ops([cloned_if_op])
                print(f"Missing values for cloned IfOp: {new_missing_values}")

                for missing_value in new_missing_values:
                    if isinstance(missing_value.op, quantum.InsertOp):
                        quantum_register = missing_value

                current_op.results[0].replace_by_if(cloned_if_op.results[0], lambda use: use.operation not in cloned_if_op_ops)


    def analyze_missing_values_for_ops(self, ops: list[Operation]) -> list[SSAValue]:
        """get missing values for ops
        Given a list of operations, return the values that are missing from the operations.
        """
        ops_walk = list(chain(*[op.walk() for op in ops]))

        ops_defined_values = set()
        all_operands = set()

        for nested_op in ops_walk:
            ops_defined_values.update(nested_op.results)
            all_operands.update(nested_op.operands)

            if hasattr(nested_op, "regions") and nested_op.regions:
                for region in nested_op.regions:
                    for block in region.blocks:
                        ops_defined_values.update(block.args)

        missing_values = list(all_operands - ops_defined_values)
        missing_values = [v for v in missing_values if v is not None]

        return missing_values

    def analyze_definitions_for_ops(self, ops: list[Operation]) -> list[SSAValue]:
        """get defined values for ops
        Given a list of operations, return the values that are defined by the operations.
        """
        # ops_walk = list(chain(*[op.walk() for op in ops]))
        ops_walk = []

        for op in ops:
            for region in op.regions:
                for block in region.blocks:
                    for child_op in block.ops:
                        ops_walk.append(child_op)

        ops_defined_values = set()

        for nested_op in ops_walk:
            ops_defined_values.update(nested_op.results)

            if hasattr(nested_op, "regions") and nested_op.regions:
                for region in nested_op.regions:
                    for block in region.blocks:
                        ops_defined_values.update(block.args)

        return list(ops_defined_values)

    def analyze_required_outputs(
            self, ops: list[Operation], terminal_op: Operation, new_original_func_op: func.FuncOp = None
        ) -> list[SSAValue]:
        """get required outputs for ops
        Given a list of operations and a terminal operation, return the values that are
        required by the operations after the terminal operation.
        Noted: It's only consdider the values that are defined in the operations and required by
        the operations after the terminal operation!
        """
        ops_walk = list(chain(*[op.walk() for op in ops]))

        ops_defined_values = set()

        for nested_op in ops_walk:
            ops_defined_values.update(nested_op.results)

        required_outputs = set()
        found_terminal = False

        body_walk = self.original_func_op.body.walk()

        if new_original_func_op is not None:
            body_walk = new_original_func_op.body.walk()

        for op in body_walk:
            if op == terminal_op:
                found_terminal = True
                continue

            if found_terminal:
                for operand in op.operands:
                    if operand in ops_defined_values:
                        required_outputs.add(operand)

        return list(required_outputs)

    def clone_operations_to_block(self, ops_to_clone, target_block, value_mapper):
        """Clone operations to target block, use value_mapper to update references"""
        for op in ops_to_clone:
            cloned_op = op.clone(value_mapper)
            target_block.add_op(cloned_op)

            self.update_value_mapper_recursively(op, cloned_op, value_mapper)

    def update_value_mapper_recursively(self, orig_op, cloned_op, value_mapper):
        """update value_mapper for all operations in operation"""
        for orig_result, new_result in zip(orig_op.results, cloned_op.results):
            value_mapper[orig_result] = new_result

        for orig_region, cloned_region in zip(orig_op.regions, cloned_op.regions):
            self.update_region_value_mapper(orig_region, cloned_region, value_mapper)

    def update_region_value_mapper(self, orig_region, cloned_region, value_mapper):
        """update value_mapper for all operations in region"""
        for orig_block, cloned_block in zip(orig_region.blocks, cloned_region.blocks):
            for orig_arg, cloned_arg in zip(orig_block.args, cloned_block.args):
                value_mapper[orig_arg] = cloned_arg

            for orig_nested_op, cloned_nested_op in zip(orig_block.ops, cloned_block.ops):
                self.update_value_mapper_recursively(orig_nested_op, cloned_nested_op, value_mapper)


@compiler_transform
class IfOpPartitionTTPass(ModulePass):
    name = "if-operator-partitioning-ttpass"

    def apply(self, ctx: Context, module: builtin.ModuleOp) -> None:

        print_mlir(module, "Before IfOpPartitionTTPass:")

        self.apply_on_qnode(module, IfOperatorPartitioningPass())

        print_mlir(module, "After IfOpPartitionTTPass:")

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





##############################################################################
# xDSL Transform Unroll Static For Loop
##############################################################################

class UnrollLoopPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter) -> None:

        # # create a new  bounds from arith.index_cast to arith.constant if needed
        # for bound in [op.lb, op.ub, op.step]:
        #     if isinstance(bound.owner, arith.IndexCastOp):
        #         index_cast_op = bound.owner
        #         if isinstance(index_cast_op.inp.owner, arith.ConstantOp):
        #             const_op = index_cast_op.inp.owner
        #             rewriter.insert(
        #                 arith.ConstantOp(
        #                     const_op.value, result_type=bound.type
        #                 ),
        #                 InsertPoint.before(op)
        #             )



        lb_op_val = resolve_constant_params(op.lb.owner.operands[0].owner.operands[0].owner.output)
        ub_op_val = resolve_constant_params(op.ub.owner.operands[0].owner.operands[0].owner.output)
        step_op_val = resolve_constant_params(op.step.owner.operands[0].owner.operands[0].owner.output)

        # From lb_op_val  create a new arith.ConstantOp
        lb_const_op = rewriter.insert(
            arith.ConstantOp(
                builtin.IntegerAttr(lb_op_val,  builtin.IndexType())
            ) , InsertPoint.before(op)
        )


        if (
            not isinstance(lb_op := op.lb.owner, arith.ConstantOp)
            or not isinstance(ub_op := op.ub.owner, arith.ConstantOp)
            or not isinstance(step_op := op.step.owner, arith.ConstantOp)
        ):
            return


        assert isinstance(lb_op.value, builtin.IntegerAttr)
        assert isinstance(ub_op.value, builtin.IntegerAttr)
        assert isinstance(step_op.value, builtin.IntegerAttr)

        lb = lb_op.value.value.data
        ub = ub_op.value.value.data
        step = step_op.value.value.data

        iter_args: tuple[SSAValue, ...] = op.iter_args

        i_arg, *block_iter_args = op.body.block.args

        for i in range(lb, ub, step):
            i_op = rewriter.insert(
                arith.ConstantOp(builtin.IntegerAttr(i, lb_op.value.type))
            )
            i_op.result.name_hint = i_arg.name_hint

            value_mapper: dict[SSAValue, SSAValue] = {
                arg: val for arg, val in zip(block_iter_args, iter_args, strict=True)
            }
            value_mapper[i_arg] = i_op.result

            for inner_op in op.body.block.ops:
                if isinstance(inner_op, scf.YieldOp):
                    iter_args = tuple(
                        value_mapper.get(val, val) for val in inner_op.arguments
                    )
                else:
                    rewriter.insert(inner_op.clone(value_mapper))

        rewriter.replace_matched_op((), iter_args)

class UnrollStaticForLoop(RewritePattern):
    """Unroll static for loops as an xDSL transform in Catalyst."""

    def __init__(self):
        self.module: builtin.ModuleOp = None


    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter):
        """Transform a quantum function (qnode) to perform tree-traversal simulation."""
        self.original_func_op = func_op

        printer = Printer()

        if "qnode" not in func_op.attributes:
            return

        self.module = get_parent_of_type(func_op, builtin.ModuleOp)
        assert self.module is not None, "got orphaned qnode function"

        print("*"*120)
        print("Before UnrollStaticForLoop:")
        printer.print_op(self.module)
        print("*"*120)

@compiler_transform
class UnrollStaticLoopTTPass(ModulePass):
    name = "tree-traversal"

    def apply(self, ctx: Context, module: builtin.ModuleOp) -> None:
        # self.apply_on_qnode(module, IfOperatorPartitioningPass())
        PatternRewriteWalker(UnrollLoopPattern()).rewrite_module(module)
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



    @qml.qjit(keep_intermediate=False, pass_plugins=[getXDSLPluginAbsolutePath()], autograph=True)
    @IfOpPartitionTTPass
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def captured_circuit_1(c1: int, c2: int, c3: int):
        # Create a superposition state with different probabilities
        qml.H(0)
        # cond_2 = qml.measure(1)

        def ansatz_true():
            qml.X(0)

            def nested_ansatz_true():
                qml.Y(0)

                def nested_nested_ansatz_true():
                    qml.H(0)
                def nested_nested_ansatz_false():
                    qml.T(0)

                cond_3 = c3 == 1

                qml.cond(cond_3,
                         nested_nested_ansatz_true,
                         nested_nested_ansatz_false)()

            def nested_ansatz_false():
                qml.Z(0)

                def nested_nested_ansatz_true():
                    qml.H(0)
                def nested_nested_ansatz_false():
                    qml.T(0)

                cond_3 = c3 == 1

                qml.cond(cond_3,
                         nested_nested_ansatz_true,
                         nested_nested_ansatz_false)()


            cond_2 = c2 == 1

            qml.cond(cond_2,
                     nested_ansatz_true,
                     nested_ansatz_false)()

        def ansatz_false():
            qml.S(0)

        qml.cond(c1 == 1,
                 ansatz_true,
                 ansatz_false)()

        # qml.S(2)

        return qml.state()

    print("-"*40)
    print(captured_circuit_1(1,1,1))
    print("-"*40)
    print(captured_circuit_1(1,1,0))
    print("-"*40)
    print(captured_circuit_1(1,0,0))
    print("-"*40)
    print(captured_circuit_1(1,0,1))
    print("-"*40)
    print(captured_circuit_1(0,0,0))


    # -----------------------------------------------------------------------
    # def captured_circuit_1(x: float, y: float, z: int):
        # Create a superposition state with different probabilities
    #     qml.H(0)
    #     # cond_2 = qml.measure(1)

    #     def ansatz_true():
    #         qml.X(0)

    #         def nested_ansatz_true():
    #             qml.Y(0)

    #             def nested_nested_ansatz_true():
    #                 qml.H(0)
    #             def nested_nested_ansatz_false():
    #                 qml.T(0)

    #             cond_3 = x > 3.4

    #             qml.cond(cond_3,
    #                      nested_nested_ansatz_true,
    #                      nested_nested_ansatz_false)()

    #         def nested_ansatz_false():
    #             qml.Z(0)

    #         cond_2 = x > 2.4

    #         qml.cond(cond_2,
    #                  nested_ansatz_true,
    #                  nested_ansatz_false)()

    #     def ansatz_false():
    #         qml.S(0)

    #     cond_1 = x > 1.0
    #     qml.cond(cond_1,
    #              ansatz_true,
    #              ansatz_false)()

    #     # qml.S(2)

    #     return qml.state()

    # print(captured_circuit_1(0.5, 0.3, 1))
    # print("-"*40)
    # print(captured_circuit_1(1.5, 0.3, 1))
    # print("-"*40)
    # print(captured_circuit_1(2.5, 0.3, 1))
    # print("-"*40)
    # print(captured_circuit_1(3.5, 0.3, 1))


        # qml.H(0)

        # index =  1 + z

        # def ansatz_true():
        #     qml.H(1)

        #     index_lvl_1 = 1 + index

        #     def nested_ansatz_true():

        #         index_lvl_2 = index_lvl_1 + 1

        #         qml.X(index_lvl_2)

        #     def nested_ansatz_false():

        #         index_lvl_2 = index_lvl_1 - 1

        #         qml.X(index_lvl_2)

        #         def nested_nested_ansatz_true():
        #             qml.Y(4)
        #         def nested_nested_ansatz_false():
        #             qml.Y(3)

        #         qml.cond(x > 3.4,
        #                  nested_nested_ansatz_true,
        #                  nested_nested_ansatz_false)()

        #     qml.H(index_lvl_1)

        #     qml.cond(x > 2.4,
        #              nested_ansatz_true,
        #              nested_ansatz_false)()

        # def ansatz_false():
        #     qml.H(1)

        # qml.cond(x > 1.4,
        #          ansatz_true,
        #          ansatz_false)()
        # # qml.cond(x > 1.4, ansatz_true)(y)


        # # qml.RX(y, 0)
        # qml.S(2)

        # return qml.state()
        # return qml.expval(qml.PauliZ(0))

    # print(captured_circuit_1(1.5, 0.3, 1))
