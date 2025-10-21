# State Evolution Outlining Implementation
from dataclasses import dataclass, field
from itertools import chain
from typing import Type, TypeVar

import logging
import numpy as np
import pennylane as qml

from catalyst.passes.xdsl_plugin import getXDSLPluginAbsolutePath

from pennylane.compiler.python_compiler import compiler_transform
from pennylane.compiler.python_compiler.dialects import quantum

from xdsl.context import Context
from xdsl.dialects import builtin, func
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.rewriter import InsertPoint

T = TypeVar("T")

logger = logging.getLogger(__name__)
logger.disabled = True

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_parent_of_type(op: Operation, kind: Type[T]) -> T | None:
    """Walk up the parent tree until an op of the specified type is found."""
    while (op := op.parent_op()) and not isinstance(op, kind):
        pass
    return op


class SplitNonCommutingRegion(RewritePattern):
    """Split non-commuting region into multiple regions"""

    def __init__(self):
        self.module: builtin.ModuleOp = None

    def find_begin_and_end_op(self):
        """return the first and last operation in the function"""
        ops = list(self.original_func_op.body.ops)
        return ops[0], ops[-2]

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

    def create_dup_function(self, rewriter: PatternRewriter, i: int):
        """Create a new function for the dup region by fully cloning the original function."""
        print(f"create_dup_function for commuting region {i}")

        # Use the same signature as the original function
        original_func_type = self.original_func_op.function_type
        input_types = list(original_func_type.inputs.data)
        output_types = list(original_func_type.outputs.data)
        fun_type = builtin.FunctionType.from_lists(input_types, output_types)

        dup_func = func.FuncOp(self.original_func_op.sym_name.data + ".dup." + str(i), fun_type)
        rewriter.insert_op(dup_func, InsertPoint.at_end(self.module.body.block))

        # Map original function arguments to dup function arguments
        dup_block = dup_func.regions[0].block
        orig_block = self.original_func_op.body.block
        value_mapper = {}
        for orig_arg, dup_arg in zip(orig_block.args, dup_block.args):
            value_mapper[orig_arg] = dup_arg

        # Clone all operations except the return statement
        ops_to_clone = []
        return_op = None
        for op in orig_block.ops:
            if isinstance(op, func.ReturnOp):
                return_op = op
            else:
                ops_to_clone.append(op)

        # Clone operations
        self.clone_operations_to_block(ops_to_clone, dup_block, value_mapper)

        # Clone the return statement
        if return_op:
            return_values = [value_mapper.get(val, val) for val in return_op.operands]
            new_return_op = func.ReturnOp(*return_values)
            dup_block.add_op(new_return_op)

        # Remove expvals from other groups and update return statement
        self.removeGroup(dup_func, i)

        print("create_dup_function successfully")

        return dup_func

    def removeGroup(self, dup_func: func.FuncOp, target_group: int):
        """Remove expvals from other groups and update return statement.

        For the dup function of group i, this removes all expval operations that
        belong to other groups, traces their results to the return statement,
        and removes those return values.
        """
        print(f"removeGroup: keeping only group {target_group}")

        # Find all expval operations in the dup function
        expval_ops_to_remove = []
        for op in dup_func.body.walk():
            if isinstance(op, quantum.ExpvalOp):
                # Check if this expval has a group attribute
                if "group" in op.attributes:
                    group_attr = op.attributes["group"]
                    group_id = group_attr.value.data

                    # If it's not the target group, mark for removal
                    if group_id != target_group:
                        expval_ops_to_remove.append(op)

        if not expval_ops_to_remove:
            print(f"No expvals to remove for group {target_group}")
            return

        # Collect all operations in the chain from expval to return
        ops_to_remove = set()
        values_to_remove_from_return = set()

        for expval_op in expval_ops_to_remove:
            expval_result = expval_op.results[0]

            # Trace and collect all operations from expval to return
            chain_ops, return_values = self.trace_and_collect_chain(expval_result, dup_func)
            ops_to_remove.update(chain_ops)
            values_to_remove_from_return.update(return_values)

            # Include the expval operation itself
            ops_to_remove.add(expval_op)

            # And the observable operation that feeds the expval
            assert len(expval_op.operands) > 0, "expval_op should have at least one operand"
            obs_value = expval_op.operands[0]
            obs_op = obs_value.owner
            if isinstance(
                obs_op,
                (
                    quantum.NamedObsOp,
                    quantum.ComputationalBasisOp,
                    quantum.HamiltonianOp,
                    quantum.TensorOp,
                ),
            ):
                ops_to_remove.add(obs_op)

        # Update the return statement first
        self.update_return_statement(dup_func, values_to_remove_from_return)

        # Now delete operations in the correct order (from users to producers)
        print(f"Removing {len(ops_to_remove)} operations in the chain")
        self.delete_operations_in_order(ops_to_remove)

        print(f"removeGroup completed for group {target_group}")

    def delete_operations_in_order(self, ops_to_remove: set[Operation]):
        """Delete operations in the correct order (from users to producers)"""
        remaining = set(ops_to_remove)

        while remaining:
            can_delete = []

            for op in remaining:
                can_delete_op = True
                for result in op.results:
                    for use in result.uses:
                        user_op = use.operation
                        if user_op in remaining:
                            can_delete_op = False
                            break
                    if not can_delete_op:
                        break

                if can_delete_op:
                    can_delete.append(op)

            if not can_delete:
                print(f"  WARNING: Cannot delete remaining {len(remaining)} operations")
                for op in remaining:
                    print(f"    Stuck operation: {op.__class__.__name__}")
                    for result in op.results:
                        for use in result.uses:
                            if use.operation in remaining:
                                print(
                                    f"      Still used by remaining: {use.operation.__class__.__name__}"
                                )
                break

            # Delete the operations that can be safely deleted
            for op in can_delete:
                print(f"  Removing: {op.__class__.__name__}")
                op.detach()
                op.erase()
                remaining.remove(op)

    def trace_and_collect_chain(
        self, value: SSAValue, func_op: func.FuncOp
    ) -> tuple[set[Operation], set[SSAValue]]:
        """Trace a value forward and collect all operations in the chain to return.

        Returns:
            - Set of operations in the chain (e.g., tensor.from_elements)
            - Set of values that reach the return statement
        """
        # Find the return operation
        return_op = None
        for op in func_op.body.ops:
            if isinstance(op, func.ReturnOp):
                return_op = op
                break

        if not return_op:
            return set(), set()

        ops_in_chain = set()
        return_values = set()

        # BFS to trace forward from the value
        to_trace = {value}
        traced = set()

        while to_trace:
            current_val = to_trace.pop()
            if current_val in traced:
                continue
            traced.add(current_val)

            # Check if this value reaches the return
            if current_val in return_op.operands:
                return_values.add(current_val)

            # Follow all uses of this value
            for use in current_val.uses:
                user_op = use.operation

                # Don't follow into return operation
                if user_op == return_op:
                    continue

                # Add this operation to the chain if it's removable
                if self.is_removable_op(user_op):
                    ops_in_chain.add(user_op)

                # Add all results to trace
                for result in user_op.results:
                    if result not in traced:
                        to_trace.add(result)

        return ops_in_chain, return_values

    def update_return_statement(self, func_op: func.FuncOp, values_to_remove: set[SSAValue]):
        """Update the return statement to remove specified values."""
        # Find the return operation
        return_op = None
        for op in func_op.body.ops:
            if isinstance(op, func.ReturnOp):
                return_op = op
                break

        if not return_op:
            return

        # Filter out values to remove
        new_return_values = [val for val in return_op.operands if val not in values_to_remove]

        # Create new return operation
        new_return_op = func.ReturnOp(*new_return_values)

        # Replace the old return operation
        return_op.detach()
        return_op.erase()  # Important: erase to remove operand uses
        func_op.body.block.add_op(new_return_op)

        # Update function signature
        new_output_types = [val.type for val in new_return_values]
        input_types = [arg.type for arg in func_op.body.block.args]
        new_fun_type = builtin.FunctionType.from_lists(input_types, new_output_types)
        func_op.function_type = new_fun_type

        print(
            f"Updated return statement: removed {len(values_to_remove)} values, kept {len(new_return_values)} values"
        )

    def is_removable_op(self, op: Operation) -> bool:
        """Check if an operation can be safely removed."""
        # Quantum operations that are safe to remove if unused
        if isinstance(
            op,
            (
                quantum.ExpvalOp,
                quantum.NamedObsOp,
                quantum.ComputationalBasisOp,
                quantum.HamiltonianOp,
                quantum.TensorOp,
            ),
        ):
            return True

        # Check operation name for tensor/stablehlo operations
        if hasattr(op, "name"):
            op_name = str(op.name)
            # Tensor operations
            if "tensor.from_elements" in op_name:
                return True
            if "tensor.extract" in op_name:
                return True
            # StableHLO operations (constants, etc.)
            if "stablehlo.constant" in op_name:
                return True

        # Check by operation class name as fallback
        class_name = op.__class__.__name__
        if "FromElements" in class_name or "Extract" in class_name or "Constant" in class_name:
            return True

        return False

    def calculate_num_commuting_region(self):
        """calculate the number of commuting region"""
        # Find all expval operations in the original function
        expval_ops = []
        for op in self.original_func_op.body.walk():
            if isinstance(op, quantum.ExpvalOp):
                expval_ops.append(op)

        if not expval_ops:
            return 0

        # For each expval, find the qubit it uses
        expval_to_qubit = {}
        for expval_op in expval_ops:
            # The expval operation takes an observable as input
            observable = expval_op.operands[0]

            # Trace back to find the qubit used by this observable
            qubit = self.get_qubit_from_observable(observable)
            if qubit is not None:
                expval_to_qubit[expval_op] = qubit

        # Group expvals by their qubits
        qubit_to_group = {}
        group_counter = 0

        for expval_op, qubit in expval_to_qubit.items():
            if qubit not in qubit_to_group:
                qubit_to_group[qubit] = group_counter
                group_counter += 1

            # Tag the expval operation with the group attribute
            group_id = qubit_to_group[qubit]
            expval_op.attributes["group"] = builtin.IntegerAttr(group_id, builtin.IntegerType(64))

        return group_counter

    def get_qubit_from_observable(self, observable: SSAValue) -> SSAValue | None:
        """Get the qubit used by an observable operation.

        Traces back from an observable to find the qubit it operates on.
        Handles NamedObsOp, ComputationalBasisOp, HamiltonianOp, and TensorOp.
        """
        if not hasattr(observable, "owner") or observable.owner is None:
            return None

        obs_op = observable.owner

        # For NamedObsOp and ComputationalBasisOp, the first operand is the qubit
        if isinstance(obs_op, (quantum.NamedObsOp, quantum.ComputationalBasisOp)):
            if len(obs_op.operands) > 0:
                return obs_op.operands[0]

        # For HamiltonianOp and TensorOp, we need to handle multiple qubits
        # For simplicity, we'll use the first qubit
        elif isinstance(obs_op, (quantum.HamiltonianOp, quantum.TensorOp)):
            if len(obs_op.operands) > 0:
                # For these, operands might be other observables
                # We need to recursively find the qubit
                return self.get_qubit_from_observable(obs_op.operands[0])

        return None

    def analyze_group_return_positions(self, num_groups: int) -> dict[int, list[int]]:
        """Analyze which return value positions belong to each group.

        Returns a dict mapping group_id -> list of return value positions
        Example: {0: [0, 1, 3, 5], 1: [2, 6], 2: [4, 7]}
        """
        print(f"Analyzing return positions for {num_groups} groups")

        # Find the return operation
        return_op = None
        for op in self.original_func_op.body.ops:
            if isinstance(op, func.ReturnOp):
                return_op = op
                break

        if not return_op:
            return {}

        # For each return value, trace back to find its group
        group_positions = {i: [] for i in range(num_groups)}

        for position, return_value in enumerate(return_op.operands):
            # Trace back to find the expval operation
            group_id = self.find_group_for_return_value(return_value)
            if group_id is not None:
                group_positions[group_id].append(position)
                print(f"  Return position {position} belongs to group {group_id}")

        return group_positions

    def find_group_for_return_value(self, return_value: SSAValue) -> int | None:
        """Trace back from a return value to find which group's expval produced it."""
        # BFS backward to find expval
        to_check = [return_value]
        checked = set()

        while to_check:
            val = to_check.pop(0)
            if val in checked:
                continue
            checked.add(val)

            op = val.owner

            # If we found an expval, check its group
            if isinstance(op, quantum.ExpvalOp):
                if "group" in op.attributes:
                    group_attr = op.attributes["group"]
                    return group_attr.value.data

            # Otherwise, check operands
            for operand in op.operands:
                if operand not in checked:
                    to_check.append(operand)

        return None

    def replace_original_with_calls(
        self,
        rewriter: PatternRewriter,
        dup_functions: list[func.FuncOp],
        group_return_positions: dict[int, list[int]],
    ):
        """Replace original function body with calls to dup functions.

        Args:
            dup_functions: List of duplicate functions (one per group)
            group_return_positions: Dict mapping group_id -> list of return positions
        """
        print("Replacing original function with calls to dup functions")

        original_block = self.original_func_op.body.block

        for op in reversed(self.original_func_op.body.ops):
            op.detach()
            op.erase()

        # Collect parameters needed for dup function calls
        # Dup functions take the same parameters as the original begin/end region
        # Look at what state_evolution_call was using and find corresponding values
        call_args = list(original_block.args)  # Use function arguments as base

        group_results = {}  # group_id -> list of result values

        for group_id, dup_func in enumerate(dup_functions):
            print(f"  Creating call to {dup_func.sym_name.data}")

            # Get the function signature to determine result types
            func_type = dup_func.function_type
            result_types = list(func_type.outputs.data)

            # Create the call operation
            call_op = func.CallOp(dup_func.sym_name.data, call_args, result_types)
            original_block.add_op(call_op)

            # Store results for this group
            group_results[group_id] = list(call_op.results)
            print(f"    Group {group_id} returns {len(call_op.results)} values")

        # Reconstruct the return statement in the original order
        # Calculate total number of return values
        total_returns = sum(len(positions) for positions in group_return_positions.values())
        final_return_values = [None] * total_returns

        for group_id, positions in group_return_positions.items():
            group_vals = group_results[group_id]
            for i, position in enumerate(positions):
                if i < len(group_vals):
                    final_return_values[position] = group_vals[i]
                    print(f"    Position {position} <- group {group_id} result {i}")

        # Filter out None values (in case something went wrong)
        final_return_values = [v for v in final_return_values if v is not None]

        # Create new return operation
        return_op = func.ReturnOp(*final_return_values)
        original_block.add_op(return_op)

        print(f"  Created return with {len(final_return_values)} values")

    def split_commuting_region(self, rewriter: PatternRewriter):
        """split commuting region into multiple regions"""

        # Calculate the number of commuting region
        num_commuting_region = self.calculate_num_commuting_region()

        # Analyze return value positions for each group
        group_return_positions = self.analyze_group_return_positions(num_commuting_region)

        # Create dup function for each commuting region
        print(f"split commuting region into {num_commuting_region} regions")
        dup_functions = []
        for i in range(num_commuting_region):
            dup_func = self.create_dup_function(rewriter, i)
            dup_functions.append(dup_func)

        # Replace original function body with calls to dup functions
        self.replace_original_with_calls(rewriter, dup_functions, group_return_positions)

        print("after split commuting region")
        print(self.module)

    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter):
        """Split commuting region into multiple regions"""
        self.module = get_parent_of_type(func_op, builtin.ModuleOp)
        assert self.module is not None, "got orphaned qnode function"
        self.original_func_op = func_op

        # Split commuting region into multiple regions
        self.split_commuting_region(rewriter)
        print("after split non-commuting region")
        print(self.module)


@compiler_transform
class SplitNonCommutingRegionPass(ModulePass):
    name = "split-non-commuting-region"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        self.apply_on_qnode(op, SplitNonCommutingRegion())

        print("after split non-commuting region")
        print(op)

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


if __name__ == "__main__":
    qml.capture.enable()

    dev = qml.device("lightning.qubit", wires=2)

    @qml.qjit(pass_plugins=[getXDSLPluginAbsolutePath()])
    @SplitNonCommutingRegionPass
    @qml.qnode(dev, shots=1000)
    def circ(x: int):
        @qml.for_loop(0, x, 1)
        def loop(i):
            qml.RX(0.12, 0)

        loop()
        return qml.expval(qml.X(0)), qml.expval(qml.Y(0)), qml.expval(qml.X(1))

    print(circ(3))
