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
"""Function for applying high-level pattern-matching to xDSL modules."""

from functools import partial, wraps
from inspect import signature
from typing import Callable

import pennylane as qml
from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import builtin, func, pdl
from xdsl.ir import Block, Operation, Region
from xdsl.passes import PassPipeline
from xdsl.traits import SymbolTable
from xdsl.transforms.apply_pdl import ApplyPDLPass

from catalyst.jax_primitives import decomposition_rule
from catalyst.jit import QJIT
from catalyst.python_interface import QuantumParser
from catalyst.python_interface.conversion import xdsl_from_qjit
from catalyst.python_interface.dialects import quantum

qml.capture.enable()


def _rewrite_mod(
    mod: builtin.ModuleOp, pattern: Callable, rewrite: Callable, num_wires: int
) -> None:

    args = list(range(num_wires))

    @xdsl_from_qjit
    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=num_wires))
    def mod_fn():
        @decomposition_rule(is_qreg=False, num_params=0)
        def pattern__(*args__):
            return pattern(*args__)

        @decomposition_rule(is_qreg=False, num_params=0)
        def rewrite__(*args__):
            return rewrite(*args__)

        pattern__(*args)
        rewrite__(*args)

        return qml.state()

    dummy_mod = mod_fn()

    qnode_mod: builtin.ModuleOp = [
        o for o in dummy_mod.body.ops if isinstance(o, builtin.ModuleOp)
    ][0]
    pattern_fn: func.FuncOp = SymbolTable.lookup_symbol(qnode_mod, "pattern__")
    rewrite_fn: func.FuncOp = SymbolTable.lookup_symbol(qnode_mod, "rewrite__")

    pattern_fn.body.block.erase_op(pattern_fn.body.block.last_op)
    rewrite_fn.body.block.erase_op(rewrite_fn.body.block.last_op)

    def create_pdl_op(
        op: Operation,
        cur_values: list[pdl.ValueType],
        value_map: dict[quantum.QubitSSAValue, int],
        pdl_value_map: dict[pdl.ValueType, int],
        type_map: dict[quantum.QubitSSAValue, pdl.TypeType],
    ):
        attr_vals = []
        attr_names = []
        for attr_name, attr in (op.properties | op.attributes).items():
            attr_names.append(builtin.StringAttr(attr_name))
            attr_vals.append(pdl.AttributeOp(attr))

        operand_vals = []
        res_type_vals = []
        for val in op.operands:
            pdl_val = cur_values[value_map[val]]
            operand_vals.append(pdl_val)
            res_type_vals.append(type_map[val])

        pdl_op = pdl.OperationOp(
            op_name=op.name,
            attribute_value_names=attr_names,
            attribute_values=attr_vals,
            operand_values=operand_vals,
            type_values=res_type_vals,
        )

        pdl_results = [
            pdl.ResultOp(index=i, parent=pdl_op).results[0] for i in range(len(op.results))
        ]
        for i, (result, pdl_result) in enumerate(zip(op.results, pdl_results, strict=True)):
            value_idx = value_map[op.operands[i]]
            value_map[result] = value_idx
            pdl_value_map[pdl_result] = value_idx
            type_map[result] = res_type_vals[i]
            cur_values[value_idx] = pdl_result

        return pdl_op

    pdl_ops: list[pdl.OperationOp] = []
    starting_values: list[pdl.ValueType] = []
    cur_values: list[pdl.ValueType] = []
    value_map: dict[quantum.QubitSSAValue, int] = {}
    pdl_value_map: dict[pdl.ValueType, int] = {}
    type_map: dict[quantum.QubitSSAValue, pdl.TypeType] = {}

    pattern_block: Block = Block()
    with ImplicitBuilder(pattern_block):

        # Initialization
        for i, (arg, arg_type) in enumerate(
            zip(pattern_fn.body.block.args, pattern_fn.body.block.arg_types)
        ):
            # Create pdl types
            pdl_type = pdl.TypeOp(arg_type).results[0]
            type_map[arg] = pdl_type

            # Create pdl values
            pdl_val = pdl.OperandOp(value_type=pdl_type).results[0]
            starting_values.append(pdl_val)
            cur_values.append(pdl_val)
            value_map[arg] = i
            pdl_value_map[pdl_val] = i

        # Create pdl ops
        for op in pattern_fn.body.ops:
            pdl_op = create_pdl_op(
                op=op,
                cur_values=cur_values,
                value_map=value_map,
                pdl_value_map=pdl_value_map,
                type_map=type_map,
            )
            pdl_ops.append(pdl_op)

    rewrite_block: Block = Block()
    with ImplicitBuilder(rewrite_block):

        rw_value_map: dict[quantum.QubitSSAValue, int] = {}
        rw_pdl_ops: list[pdl.OperationOp] = []
        rw_terminal_values: list[pdl.ValueType] = []

        # Rewrite block initialization
        for i, (p_arg, r_arg) in enumerate(
            zip(pattern_fn.body.block.args, rewrite_fn.body.block.args, strict=True)
        ):
            rw_terminal_values.append(starting_values[i])
            rw_value_map[r_arg] = i
            type_map[r_arg] = type_map[p_arg]

        # Create pdl ops for operations in the rewrite pattern
        for op in rewrite_fn.body.ops:
            pdl_op = create_pdl_op(
                op=op,
                cur_values=rw_terminal_values,
                value_map=rw_value_map,
                pdl_value_map=pdl_value_map,
                type_map=type_map,
            )
            rw_pdl_ops.append(pdl_op)

        # Replace all operations in the original pattern with values generated by
        # the rewrite pattern
        for op, pdl_op in tuple(zip(pattern_fn.body.ops, pdl_ops, strict=True))[::-1]:
            repl_vals = []
            for pdl_val in pdl_op.operand_values:
                if isinstance(pdl_val.op, pdl.ResultOp):
                    repl_vals.append(pdl_val)
                else:
                    idx = pdl_value_map[pdl_val]
                    repl_vals.append(rw_terminal_values[idx])

            _ = pdl.ReplaceOp(pdl_op, repl_values=repl_vals)

    rewrite_op = pdl.RewriteOp(pdl_ops[-1].results[0], body=Region(rewrite_block))
    pattern_block.add_op(rewrite_op)
    pattern_op = pdl.PatternOp(benefit=1, sym_name="temp", body=Region(pattern_block))
    mod.body.block.add_op(pattern_op)

    ctx = Context()
    _ = QuantumParser(ctx, "")

    PassPipeline((ApplyPDLPass(),)).apply(ctx, mod)
    mod.body.block.erase_op(pattern_op)


def pattern_match(func: QJIT = None, patterns: dict[Callable, Callable] = {}):
    """Apply pattern matching to q QJIT-ed workflow."""
    if func is None:
        return partial(pattern_match, patterns=patterns)

    @wraps(func)
    def wrapper(*args, **kwargs):
        mod = xdsl_from_qjit(func)(*args, **kwargs)

        for pattern, rewrite in patterns.items():
            p_nargs = len(signature(pattern).parameters)
            r_nargs = len(signature(rewrite).parameters)
            if p_nargs != r_nargs:
                raise ValueError("Pattern and match must have the same number of qubits as inputs")

            _rewrite_mod(mod, pattern, rewrite, p_nargs)

        return mod

    return wrapper
