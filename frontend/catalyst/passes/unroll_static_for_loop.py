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
from typing import Type, TypeVar

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


def print_mlir(op, msg=""):
    printer = Printer()
    print("-"*100)
    print(f"// Start {msg}")
    if isinstance(op, Region):
        printer.print_region(op)
    elif isinstance(op, Block):
        printer.print_block(op)
    elif isinstance(op, Operation):
        printer.print_op(op)
    print(f"\n// End {msg}")
    print("-"*100)

def print_ssa_values(values, msg="SSA Values"):
    print(f"// {msg}")
    for val in values:
        print(f"  - {val}")

##############################################################################
# xDSL Transform If Operator Partitioning
##############################################################################

class IfOperatorPartitioningPass(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter) -> None:
        """Partition the if operation into separate branches for each operator."""

        self.original_func_op = op
        # self.duplicate_if_op(op, rewriter)
        self.split_if_op(op, rewriter)


    def __init__(self):
        self.module: builtin.ModuleOp = None
        self.original_func_op: func.FuncOp = None

    def split_if_op(self, op: func.FuncOp, rewriter: PatternRewriter) -> None:
        # Find scf.IfOp

        op_walk = op.walk()
        for current_op in op_walk:
            if isinstance(current_op, scf.IfOp):

                print_mlir(current_op, "Processing scf.IfOp:")

                # Analyze missing values for the IfOp
                missing_values = self.analyze_missing_values_for_ops([current_op])
                print_ssa_values(missing_values, "Missing values for IfOp:")

                # Get outputs required by operations after the IfOp
                required_outputs = self.analyze_required_outputs(
                    [current_op], current_op.next_op
                )
                print_ssa_values(required_outputs, "Required outputs after IfOp:")

                # Get quantum register from missing values
                qreg_if_op = [mv for mv in missing_values if isinstance(mv.type, quantum.QuregType)]
                print_ssa_values(qreg_if_op, "Quantum register for IfOp:")

                return_vals_if_op = [ro for ro in required_outputs if ro in current_op.results]
                print_ssa_values(return_vals_if_op, "Return values for IfOp:")

                # True and False regions
                true_region = current_op.true_region
                false_region = current_op.false_region

                print_mlir(true_region, "True Region:")
                print("Block Args True Region:", true_region.blocks[0].arg_types)
                print_mlir(false_region, "False Region:")
                print("Block Args False Region:", false_region.blocks[0].arg_types)


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

                print_mlir(new_if_op_4_true, "New IfOp for True Branch:")

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
                print_mlir(new_if_op_4_false, "New IfOp for False Branch:")

                # --------------------------------------------------------------------------

                print_mlir(op, "Function after IfOp Partitioning:")

                original_if_op_results = current_op.results[0]
                original_if_op_results.replace_by(qreg_if_op[0])

                list_op_if = [curr_op for curr_op in current_op.walk()]
                # Remove the ops in the original IfOp
                for op in list_op_if[::-1]:
                    print_mlir(op, "Erasing original IfOp ops:")
                    op.detach()
                    op.erase()


    def create_if_op_partition(self,
                               rewriter: PatternRewriter,
                               if_region: Region,
                               previous_IfOp: scf.IfOp,
                               value_mapper: dict[SSAValue, SSAValue],
                               op_where_insert_after: Operation,
                               conditional: SSAValue = None,
                               attr_dict: dict[str, builtin.Attribute] = None
                               ) -> scf.IfOp:

        true_ops = list(chain(*[op.walk() for op in if_region.blocks]))

        new_true_block = Block()

        self.clone_operations_to_block(
            true_ops,
            new_true_block,
            value_mapper
        )
        print_mlir(new_true_block, "Cloned True Block after cloning ops:")


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

    def analyze_required_outputs(
            self, ops: list[Operation], terminal_op: Operation
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
        for op in self.original_func_op.body.walk():
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

    @qml.qjit(keep_intermediate=False, pass_plugins=[getXDSLPluginAbsolutePath()], autograph=False)
    @IfOpPartitionTTPass
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def captured_circuit_1(x: float, y: float):
        # Create a superposition state with different probabilities

        qml.H(0)

        index =  1

        def ansatz_true(y: float):
            # qml.measure(index)
            # qml.X(0)
            # qml.measure(index)

            # qml.ctrl(qml.RX(y,0),1)
            qml.H(1)
            qml.H(2)

        def ansatz_false(y: float):
            # qml.measure(index)
            # qml.Y(0)
            # qml.measure(index)
            # qml.RY(y,0)
            qml.H(1)
            qml.Y(0)

        qml.cond(x > 1.4, ansatz_true, ansatz_false)(y)
        # qml.cond(x > 1.4, ansatz_true)(y)

        # qml.RX(y, 0)
        qml.S(2)

        return qml.state()
        # return qml.expval(qml.PauliZ(0))

    print(captured_circuit_1(1.5, 0.3))
