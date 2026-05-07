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

"""This file contains the implementation of the split_non_commuting transform,
written using xDSL. Each qnode function with measurements on overlapping qubits is
split into separate cloned functions, one per qubit-disjoint group of measurements."""

from dataclasses import dataclass

from xdsl import context, passes
from xdsl.dialects import builtin, func
from xdsl.rewriter import InsertPoint, Rewriter

from catalyst.python_interface.dialects.quantum import ExpvalOp, NamedObsOp
from catalyst.python_interface.pass_api import compiler_transform


def _split_func(func_op: func.FuncOp) -> None:
    """Split a single qnode FuncOp into per-measurement-group functions."""
    # Collect ExpvalOps from the top-level block only (measurements are never nested).
    expval_ops = [op for op in func_op.body.blocks[0].ops if isinstance(op, ExpvalOp)]
    if not expval_ops:
        return

    return_op = func_op.get_return_op()
    if return_op is None:
        return

    # Map each return-value position to its producing ExpvalOp via tensor.from_elements.
    ret_pos_to_expval: dict[int, ExpvalOp] = {}
    for i, ret_arg in enumerate(return_op.arguments):
        def_op = ret_arg.op
        if isinstance(def_op, ExpvalOp):
            ret_pos_to_expval[i] = def_op
        else:
            for operand in def_op.operands:
                if isinstance(operand.op, ExpvalOp):
                    ret_pos_to_expval[i] = operand.op
                    break

    # Determine the qubit SSA value for each ExpvalOp (via NamedObsOp).
    expval_to_qubit: dict[int, object] = {}
    for expval_op in expval_ops:
        obs_def = expval_op.obs.op
        expval_to_qubit[id(expval_op)] = obs_def.qubit if isinstance(obs_def, NamedObsOp) else None

    # Greedy qubit-disjoint grouping: assign each ExpvalOp to the first group
    # whose qubit set does not overlap with this measurement's qubit.
    groups: list[list] = []
    group_qubits: list[set] = []

    for expval_op in expval_ops:
        qubit = expval_to_qubit[id(expval_op)]
        placed = False
        for g_idx in range(len(groups)):
            if qubit is None or qubit not in group_qubits[g_idx]:
                groups[g_idx].append(expval_op)
                if qubit is not None:
                    group_qubits[g_idx].add(qubit)
                placed = True
                break
        if not placed:
            groups.append([expval_op])
            group_qubits.append({qubit} if qubit is not None else set())

    func_name = func_op.sym_name.data
    expval_to_ret_pos = {id(ev): pos for pos, ev in ret_pos_to_expval.items()}

    # For each group: clone the FuncOp, remove non-group measurements, update type.
    clones: list[tuple[list, func.FuncOp]] = []
    last_inserted = func_op

    for g_idx, group in enumerate(groups):
        group_ids = {id(ev) for ev in group}
        clone = func_op.clone()
        clone.properties["sym_name"] = builtin.StringAttr(f"{func_name}.group.{g_idx}")

        clone_expvals = [op for op in clone.body.blocks[0].ops if isinstance(op, ExpvalOp)]
        clone_ret_op = clone.get_return_op()
        clone_ret_args = list(clone_ret_op.arguments)

        # Find clone return positions that belong to measurements outside this group.
        positions_to_remove: set[int] = set()
        expvals_to_erase = []

        for i, orig_expval in enumerate(expval_ops):
            if id(orig_expval) not in group_ids:
                clone_expval = clone_expvals[i]
                expvals_to_erase.append(clone_expval)
                for use in clone_expval.expval.uses:
                    from_elems = use.operation
                    for j, ret_arg in enumerate(clone_ret_args):
                        if ret_arg is from_elems.results[0]:
                            positions_to_remove.add(j)
                            break

        # Replace clone return op first, then erase the stripped measurement chains.
        new_clone_ret_args = [
            v for j, v in enumerate(clone_ret_args) if j not in positions_to_remove
        ]
        Rewriter.replace_op(clone_ret_op, func.ReturnOp(*new_clone_ret_args))

        for clone_expval in expvals_to_erase:
            obs_def = clone_expval.obs.op
            for use in list(clone_expval.expval.uses):
                Rewriter.erase_op(use.operation)
            Rewriter.erase_op(clone_expval)
            if isinstance(obs_def, NamedObsOp) and not list(obs_def.obs.uses):
                Rewriter.erase_op(obs_def)

        clone.update_function_type()
        Rewriter.insert_op(clone, InsertPoint.after(last_inserted))
        last_inserted = clone
        clones.append((group, clone))

    # Replace the original function body with calls to each group function.
    orig_block = func_op.body.blocks[0]
    orig_ops = list(orig_block.ops)
    orig_ret_op = func_op.get_return_op()

    call_info = []
    for group, clone_func in clones:
        clone_name = clone_func.sym_name.data
        clone_ret_types = list(clone_func.function_type.outputs.data)
        call_op = func.CallOp(clone_name, [], clone_ret_types)
        Rewriter.insert_op(call_op, InsertPoint.before(orig_ret_op))
        call_info.append((group, call_op))

    # Map each call result back to its original return position.
    new_ret_vals: dict[int, object] = {}
    for group, call_op in call_info:
        for local_idx, orig_expval in enumerate(group):
            ret_pos = expval_to_ret_pos.get(id(orig_expval))
            if ret_pos is not None:
                new_ret_vals[ret_pos] = call_op.res[local_idx]

    n_ret = len(orig_ret_op.arguments)
    new_ret_op = func.ReturnOp(*[new_ret_vals[i] for i in range(n_ret)])
    Rewriter.replace_op(orig_ret_op, new_ret_op)

    # Erase all original ops (safe_erase=False since some may still have cross-references).
    for op in reversed([op for op in orig_ops if op is not orig_ret_op]):
        Rewriter.erase_op(op, safe_erase=False)


@dataclass(frozen=True)
class SplitNonCommutingPass(passes.ModulePass):
    """Pass for splitting qnode functions with non-commuting measurements.

    Each ``func.func`` with a ``qnode`` attribute whose measurements act on
    overlapping qubits is replaced by a stub that calls one cloned function per
    qubit-disjoint measurement group.  The original return order is preserved.
    """

    name = "xdsl-split-non-commuting"

    def apply(self, _ctx: context.Context, module: builtin.ModuleOp) -> None:
        """Apply the split non-commuting pass."""
        func_ops = [
            op
            for op in module.body.walk()
            if isinstance(op, func.FuncOp) and "qnode" in op.attributes
        ]
        for func_op in func_ops:
            _split_func(func_op)


split_non_commuting_pass = compiler_transform(SplitNonCommutingPass)
