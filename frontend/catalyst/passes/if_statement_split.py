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
    # should_print = False
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
    # should_print = False
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
        # Detect mcm inside If statement
        flat_if = self.detect_mcm_in_if_ops(op)

        if not flat_if:
            return

        # print(f"FDX: Should flat: {flat_if}")

        # Split IfOps into only true branches
        self.split_nested_if_ops(op, rewriter)

        # print_mlir(op, "After splitting IfOps:")

        # Flatten nested IfOps
        self.flatten_nested_IfOps(op, rewriter)


    def __init__(self):
        self.module: builtin.ModuleOp = None
        self.original_func_op: func.FuncOp = None
        self.holder_returns : dict[scf.IfOp, scf.IfOp] = {}

    def detect_mcm_in_if_ops(self, op: func.FuncOp) -> bool:
        """Detect if there are measurement-controlled operations inside IfOps."""
        op_walk = op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.IfOp):
                # Check if there are measurement-controlled operations inside the IfOp
                for inner_op in current_op.true_region.ops:
                    if isinstance(inner_op, quantum.MeasureOp):
                        return True
                for inner_op in current_op.false_region.ops:
                    if isinstance(inner_op, quantum.MeasureOp):
                        return True
        return False

    def flatten_nested_IfOps(self, main_op: func.FuncOp, rewriter: PatternRewriter) -> None:
        """Flatten nested scf.IfOps into a single level scf.IfOp."""

        # Check for deepest nested IfOps
        nested_IfOp = self.get_deepest_nested_ifs(main_op)

        depth = nested_IfOp[0][1] if nested_IfOp else 0
        target_if_op = nested_IfOp[0][0] if nested_IfOp else None

        if depth > 1:
            self.flatten_if_ops_deep(target_if_op.parent_op(), rewriter)
            self.flatten_nested_IfOps(main_op, rewriter)
        else:
            return

    def get_deepest_nested_ifs(self, parent_if_op: scf.IfOp) -> IfOpWithDepth:
        """Finds the scf.if operation(s) nested at the maximum depth inside the parent_if_op."""
        # The parent IfOp A is at depth 0, so its immediate children (B, D) are at depth 1.
        # We initialize the search list.
        deepest_ops_with_depth: List[IfOpWithDepth] = [(None, 0)]

        # Start the recursion. We look *inside* the regions of the parent_if_op.
        self._find_deepest_if_recursive(parent_if_op, 0, deepest_ops_with_depth)

        # Extract only the IfOp objects from the list of (IfOp, depth) tuples.
        return deepest_ops_with_depth

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
                        # the if should have the attribute  contain_mcm = "true"

                        contain_mcm = "contain_mcm" in child_op.attributes

                        if not contain_mcm:
                            continue

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


    def flatten_if_ops_deep(self, main_op: scf.IfOp, rewriter: PatternRewriter) -> None:
        """Flatten nested scf.IfOps into a single level scf.IfOp."""

        if isinstance(main_op, scf.IfOp):

            outer_if_op = main_op

            new_outer_if_op_output = [out for out in outer_if_op.results]
            new_outer_if_op_output_types = [out.type for out in outer_if_op.results]

            has_nested_if_ops, nested_if_ops = self.get_nested_if_ops(outer_if_op)
            where_to_insert = outer_if_op

            # Holder for IfOps that are kept for updating SSA values later
            self.holder_returns = {}

            for inner_op in nested_if_ops:

                where_to_insert, outer_if_op = self.move_simple_inner_if_op_2_outer(
                                                inner_op, outer_if_op, new_outer_if_op_output, new_outer_if_op_output_types, where_to_insert, rewriter,
                                                )

            # detach and erase old outer if op
            for hold_op in self.holder_returns.keys():
                hold_op.detach()
                hold_op.erase()

    def move_simple_inner_if_op_2_outer(self,
                                        inner_op: scf.IfOp, outer_if_op: scf.IfOp, new_outer_if_op_output: list[SSAValue], new_outer_if_op_output_types: list[Type], where_to_insert: scf.IfOp,
                                        rewriter: PatternRewriter,
                                        ) -> None:
        """Move simple inner IfOp after the outer IfOp."""

        # print_mlir(outer_if_op,"Outer If")
        # print_mlir(inner_op,"Inner If")

        definition_outer = self.analyze_definitions_for_ops([outer_if_op])
        missing_values_inner = self.analyze_missing_values_for_ops([inner_op])

        ssa_needed_from_outer = set(missing_values_inner).intersection(set(definition_outer))

        # Select only definition outer
        # Use list to preserve order
        missing_values_inner = [mv for mv in missing_values_inner if mv in definition_outer]

        for mv in ssa_needed_from_outer:
            if not isinstance(mv.type, quantum.QuregType):
                new_outer_if_op_output.append(mv)
                new_outer_if_op_output_types.append(mv.type)

        # Matching qreg

        inner_results = inner_op.results

        # Replace the qreg from the inner IfOp with the immediate outer IfOp qreg
        # This dont affect the inner IfOp since its qreg is only used in quantum ops inside its regions
        qreg_if_op_inner = [mv for mv in missing_values_inner if isinstance(mv.type, quantum.QuregType)]

        for result in inner_results:
            if isinstance(result.type, quantum.QuregType):
                result.replace_by(qreg_if_op_inner[0])

        qreg_if_op_outer = [output for output in where_to_insert.results if isinstance(output.type, quantum.QuregType)]

        assert len(qreg_if_op_outer) == 1, "Expected exactly one quantum register in outer IfOp results."

        # Detach inner_op from its parent before modifying
        if len(inner_results) == 1 :
            inner_op.detach()
        else:
            # Add a new attribute to mark it as flattened
            inner_op.attributes["old_return"] = builtin.StringAttr("true")


        # expand the current attr_dict
        attr_dict = inner_op.attributes.copy()
        attr_dict.update({"flattened": builtin.StringAttr("true")})

        ############################################################################################
        # Create new inner IfOp with updated regions

        # ------------------------------------------------------------------------------------------
        # Inner true region

        # Create comprehensive value mapping for all values used in both regions
        value_mapper = {}
        value_mapper[qreg_if_op_inner[0]] = qreg_if_op_outer[0]

        inner_true_region = inner_op.true_region

        true_ops = [op for op in inner_true_region.blocks[0].ops]

        new_true_block = Block()

        self.clone_operations_to_block(
            true_ops,
            new_true_block,
            value_mapper
        )

        # ------------------------------------------------------------------------------------------
        # Inner false region

        false_inner_ops = [op for op in inner_op.false_region.blocks[0].ops]

        new_false_block = None

        if len(false_inner_ops) == 1 and isinstance(false_inner_ops[0], scf.YieldOp):
            # If the false region only contains a yield operation, we can create an empty block

            # Create a new empty block for false region
            new_false_block = Block()

            # Create a yield operation for false region using the same return types as the original IfOp
            yield_false = scf.YieldOp(where_to_insert.results[0])

            # Create a new empty block for false region
            new_false_block.add_op(yield_false)

        else:
            # If the false region contains other operations, clone them as usual
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
        new_if_op_attrs.update(attr_dict or {})
        # ------------------------------------------------------------------------------------------
        # Create new IfOp with cloned regions

        # Check if we need to update the conditional, if the conditional not depends on previous IfOp results
        # that have been removed, then we need to update it
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
            inner_op.result_types,
            [new_true_block],
            [new_false_block],
            new_if_op_attrs
        )
        rewriter.insert_op(new_inner_op, InsertPoint.after(where_to_insert))


        # Update uses of old inner IfOp results to new inner IfOp results
        new_inner_op_ops = list(chain(*[op.walk() for op in [new_inner_op]]))
        where_to_insert.results[0].replace_by_if(new_inner_op.results[0], lambda use: use.operation not in new_inner_op_ops)

        where_to_insert = new_inner_op

        # Detach and erase old inner IfOp
        if len(inner_results) == 1 :
            inner_op.erase()
        else:
            self.holder_returns[inner_op] = new_inner_op
            update_unused_cond = False
            unused_op = None
            for op in self.holder_returns.keys():
                for res in op.results:
                    if inner_op.cond == res:
                        update_unused_cond = True
                        unused_op = op
            if update_unused_cond:
                inner_op.cond.replace_by(unused_op.cond)
        ############################################################################################
        # Create a new outer IfOp that includes the new outputs needed from the inner IfOp

        # ------------------------------------------------------------------------------------------
        # Outer true block

        true_block = outer_if_op.true_region.detach_block(0)

        true_yield_op = [op for op in true_block.ops if isinstance(op, scf.YieldOp)][-1]

        # Merge the existing true yield operands with the missing values from inner IfOp
        new_res = [res for res in true_yield_op.operands] + [ ssa for ssa in  missing_values_inner if not isinstance(ssa.type, quantum.QuregType) ]
        return_types = [new_r.type for new_r in new_res]

        new_true_yield_op = scf.YieldOp( *new_res )

        rewriter.replace_op(true_yield_op, new_true_yield_op)

        # ------------------------------------------------------------------------------------------
        # Outer false block

        # Detach the false block to preserve SSA dependencies
        false_block = outer_if_op.false_region.detach_block(0)

        false_op_res = []

        if needs_to_update_conditional:
            false_op = arith.ConstantOp(builtin.IntegerAttr(0, builtin.IntegerType(1)))
            false_op_res.append(false_op.result)
            rewriter.insert_op(false_op, InsertPoint.at_start(false_block))

        false_yield_op = [op for op in false_block.ops if isinstance(op, scf.YieldOp)][-1]

        new_res = [res for res in false_yield_op.operands] + false_op_res

        new_false_yield_op = scf.YieldOp( *new_res )

        rewriter.replace_op(false_yield_op, new_false_yield_op)


        # ------------------------------------------------------------------------------------------
        # Create new IfOp with cloned regions
        new_outer_if_op = scf.IfOp(
            outer_if_op.cond,
            return_types,
            [true_block],
            [false_block],
            outer_if_op.attributes.copy()
        )

        # Add it at the top of the block
        rewriter.insert_op(new_outer_if_op, InsertPoint.before(outer_if_op))

        for old_result, new_result in zip(outer_if_op.results, new_outer_if_op.results):
            old_result.replace_by(new_result)

        outer_if_op.detach()
        outer_if_op.erase()

        outer_if_op = new_outer_if_op

        if needs_to_update_conditional:
            new_cond = new_inner_op.cond
            new_cond.replace_by_if(outer_if_op.results[-1], lambda use: use.operation in [new_inner_op])

        return where_to_insert, outer_if_op

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

    def split_nested_if_ops(self, op: func.FuncOp, rewriter: PatternRewriter, go_deeper: bool = False) -> None:
        """Recursively split nested scf.IfOps into separate branches for true and false regions."""

        if go_deeper and isinstance(op, scf.IfOp):

            # Process true region
            true_region = op.true_region
            for inner_op in true_region.ops:
                if isinstance(inner_op, scf.IfOp):
                    have_nested_if_ops = self.looking_for_nested_if_ops(inner_op)

                    # Recursively split deeper nested IfOps first
                    if have_nested_if_ops:
                        self.split_nested_if_ops(inner_op, rewriter, go_deeper=True)
                        self.split_if_op(inner_op, rewriter)
                    # Deepest level, split directly
                    if not have_nested_if_ops:
                        self.split_if_op(inner_op, rewriter)

            # Process false region
            false_region = op.false_region
            for inner_op in false_region.ops:
                if isinstance(inner_op, scf.IfOp):
                    have_nested_if_ops = self.looking_for_nested_if_ops(inner_op)

                    # Recursively split deeper nested IfOps first
                    if have_nested_if_ops:
                        self.split_nested_if_ops(inner_op, rewriter, go_deeper=True)
                        self.split_if_op(inner_op, rewriter)
                    # Deepest level, split directly
                    if not have_nested_if_ops:
                        self.split_if_op(inner_op, rewriter)
            return

        # Initial call to split nested IfOps in the function
        op_walk = op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.IfOp) and self.detect_mcm_in_if_ops(current_op):
            # if isinstance(current_op, scf.IfOp):

                have_nested_if_ops = self.looking_for_nested_if_ops(current_op)

                if have_nested_if_ops:
                    self.split_nested_if_ops(current_op, rewriter, go_deeper=True)
                    self.split_if_op(current_op, rewriter)

                if not have_nested_if_ops:
                    self.split_if_op(current_op, rewriter)

    def looking_for_nested_if_ops(self, op: scf.IfOp) -> bool:
        for inner_op in op.true_region.ops:
            if isinstance(inner_op, scf.IfOp):
                return True
        for inner_op in op.false_region.ops:
            if isinstance(inner_op, scf.IfOp):
                return True
        return False

    def split_if_op(self, op: func.FuncOp, rewriter: PatternRewriter) -> None:
        """Split an scf.IfOp into separate branches for true and false regions."""

        op_walk = op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.IfOp):

                # Analyze missing values for the IfOp
                missing_values = self.analyze_missing_values_for_ops([current_op])

                # Get outputs required by operations after the IfOp
                required_outputs = self.analyze_required_outputs(
                    [current_op], current_op.next_op
                )

                # Get quantum register from missing values
                qreg_if_op = [mv for mv in missing_values if isinstance(mv.type, quantum.QuregType)]

                # True and False regions
                true_region = current_op.true_region
                false_region = current_op.false_region

                # --------------------------------------------------------------------------
                # New partitioning logic for True region
                # --------------------------------------------------------------------------

                value_mapper = {}

                attr_dict = {
                                "contain_mcm": builtin.StringAttr("true"),
                                "partition": builtin.StringAttr("true_branch"),
                            }

                new_if_op_4_true = self.create_if_op_partition(
                    rewriter,
                    true_region,
                    current_op,
                    value_mapper,
                    current_op,
                    attr_dict=attr_dict
                )

                # --------------------------------------------------------------------------
                # New partitioning logic for False region
                # --------------------------------------------------------------------------
                # Add the negation of the condition to the false branch if needed

                true_op = arith.ConstantOp(builtin.IntegerAttr(1, builtin.IntegerType(1)))
                not_op = arith.XOrIOp(current_op.cond, true_op.result)

                # Insert not_op after new_if_op
                for new_op in [not_op, true_op]:
                    rewriter.insert_op(new_op, InsertPoint.after(new_if_op_4_true))

                #--------------------------------------------------------------------------
                # Create the new IfOp for the false region
                #--------------------------------------------------------------------------

                value_mapper = {qreg_if_op[0]: new_if_op_4_true.results[0]}

                attr_dict = {
                            "contain_mcm": builtin.StringAttr("true"),
                            "partition": builtin.StringAttr("false_branch"),
                             }

                _ = self.create_if_op_partition(
                    rewriter,
                    false_region,
                    new_if_op_4_true,
                    value_mapper,
                    not_op,
                    conditional=not_op.result,
                    attr_dict=attr_dict
                )

                # --------------------------------------------------------------------------

                original_if_op_results = current_op.results[0]
                original_if_op_results.replace_by(qreg_if_op[0])

                list_op_if = [curr_op for curr_op in current_op.walk()]

                # Remove the ops in the original IfOp
                for if_op in list_op_if[::-1]:
                    if_op.detach()
                    if_op.erase()

    def create_if_op_partition(self,
                               rewriter: PatternRewriter,
                               if_region: Region,
                               previous_IfOp: scf.IfOp,
                               value_mapper: dict[SSAValue, SSAValue],
                               op_where_insert_after: Operation,
                               conditional: SSAValue = None,
                               attr_dict: dict[str, builtin.Attribute] = None
                               ) -> scf.IfOp:

        """Create a new IfOp partition with cloned regions and updated value mapping."""


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
            [new_true_block],
            [new_false_block],
            new_if_op_attrs
        )
        rewriter.insert_op(new_if_op_4_true, InsertPoint.after(op_where_insert_after))

        new_if_op_4_true_ops = list(chain(*[op.walk() for op in [new_if_op_4_true]]))

        previous_IfOp.results[0].replace_by_if(new_if_op_4_true.results[0], lambda use: use.operation not in new_if_op_4_true_ops)

        return new_if_op_4_true

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

        # print_mlir(module, "Before IfOpPartitionTTPass:")

        self.apply_on_qnode(module, IfOperatorPartitioningPass())

        # print_mlir(module, "After IfOpPartitionTTPass:")

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



    # @qml.qjit(keep_intermediate=False, pass_plugins=[getXDSLPluginAbsolutePath()], autograph=True)
    # @IfOpPartitionTTPass
    # @qml.qnode(qml.device("lightning.qubit", wires=3))
    def captured_circuit_1(c1: int, c2: int, c3: int, c4: int):
        # Create a superposition state with different probabilities
        qml.H(0); qml.H(1); qml.H(2)
        qml.H(1)
        cond_2 = qml.measure(1)

        def ansatz_true():
            qml.RX(1.1,0)

        def ansatz_false():
            qml.Z(0)

        qml.cond(c1 == 1,
                 ansatz_true,
                 ansatz_false)()

        def ansatz_1_true():
            qml.RX(1.1,0)

            # def ansatz_nested_true():
            #     cond_2 = qml.measure(1)
            #     qml.RX(1.1, 0)

            # def ansatz_nested_false():
            #     qml.Z(0)

            # qml.cond(c2 == 1,
            #          ansatz_nested_true,
            #          ansatz_nested_false)()

        def ansatz_1_false():
            cond_2 = qml.measure(1)
            qml.Z(0)

        qml.cond(c2 == 1,
                 ansatz_1_true,
                 ansatz_1_false)()

        def ansatz_2_true():
            qml.RX(1.1,0)

            def ansatz_nested_true():
                cond_2 = qml.measure(1)
                qml.RX(1.1, 0)

            #     def ansatz_nested_nested_true():
            #         qml.RX(1.1,0)
            #     def ansatz_nested_nested_false():
            #         qml.Z(0)
            #     qml.cond(c4 == 1,
            #              ansatz_nested_nested_true,
            #              ansatz_nested_nested_false)()

            def ansatz_nested_false():
                qml.Z(0)

            qml.cond(c3 == 1,
                     ansatz_nested_true,
                     ansatz_nested_false)()

        def ansatz_2_false():
            # cond_2 = qml.measure(1)
            qml.Z(0)

        qml.cond(c2 == 1,
                 ansatz_2_true,
                 ansatz_2_false)()


        qml.S(2)

        return qml.state()

    @qml.qjit(keep_intermediate=False, pass_plugins=[getXDSLPluginAbsolutePath()], autograph=True)
    @IfOpPartitionTTPass
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def holder_function(c1: int, c2: int, c3: int, c4: int):
        return captured_circuit_1(c1, c2, c3, c4)


    @qml.qjit(keep_intermediate=False, pass_plugins=[getXDSLPluginAbsolutePath()], autograph=True)
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    def ref_function(c1: int, c2: int, c3: int, c4: int):
        return captured_circuit_1(c1, c2, c3, c4)


    permutations = [
                    (1,1,1,1),
                    (1,1,1,0),
                    (1,1,0,1),
                    (1,1,0,0),
                    (1,0,1,1),
                    (1,0,0,1),
                    (0,1,1,1),
                    (0,1,0,1),
                    (0,0,1,1),
                    (0,1,1,0),
                    (0,0,1,0),
                    (0,0,0,1),
                    (1,0,0,0),
                    (1,0,1,0),
                    (0,0,0,0)
                    ]
    for perm in permutations:
        captured_result = holder_function(perm[0], perm[1], perm[2], perm[3])
        reference_result = ref_function(perm[0], perm[1], perm[2], perm[3])
        print(f"Testing permutation c1={perm[0]}, c2={perm[1]}, c3={perm[2]}, c4={perm[3]}")
        print("Captured Result: ", captured_result)
        print("Reference Result:", reference_result)
        assert qml.math.allclose(captured_result, reference_result), "Results do not match!"

    print("All tests passed!")


    # print("-"*40)
    # print(captured_circuit_1(1,1,1))
    # print("-"*40)
    # print(captured_circuit_1(1,1,0))
    # print("-"*40)
    # print(captured_circuit_1(1,0,0))
    # print("-"*40)
    # print(captured_circuit_1(1,0,1))
    # print("-"*40)
    # print(captured_circuit_1(0,0,0))


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
