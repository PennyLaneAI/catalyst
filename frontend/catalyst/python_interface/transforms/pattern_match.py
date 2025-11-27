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

import tempfile
from collections.abc import Callable, Sequence
from functools import partial, wraps
from inspect import getsource, signature

import pennylane as qml
from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import builtin, func, pdl
from xdsl.ir import Block, Operation, Region
from xdsl.passes import ModulePass
from xdsl.traits import SymbolTable
from xdsl.transforms.apply_pdl import ApplyPDLPass

from catalyst.jax_primitives import decomposition_rule
from catalyst.jit import QJIT
from catalyst.python_interface import QuantumParser
from catalyst.python_interface.conversion import xdsl_from_qjit
from catalyst.python_interface.dialects import quantum
from catalyst.python_interface.pass_api import compiler_transform


class PatternMatchPass(ModulePass):
    """Module pass to apply Python-level pattern-matching to xDSL modules."""

    name: str = "pattern-match"
    _patterns: dict[Callable, Callable]
    _pdl_patterns: tuple[pdl.PatternOp, ...]

    def __init__(self, patterns: dict[Callable, Callable] | None = None):
        self._patterns = patterns or {}
        self._pdl_patterns = ()

    def apply(self, ctx: Context, op: builtin.ModuleOp):
        """Apply the provided patterns to the input module."""

        # Load default dialects
        for dialect in QuantumParser.default_dialects:
            if ctx.get_optional_dialect(dialect.name) is None:
                ctx.load_dialect(dialect)

        if self._patterns and not self._pdl_patterns:
            for pat, rw in self._patterns.items():
                pat_fn, rw_fn = self._pattern_to_xdsl(pat, rw)
                pattern_op = self._create_pdl_pattern(pat_fn, rw_fn)
                self._pdl_patterns += (pattern_op,)

        pdl_pass = ApplyPDLPass()
        for pattern_op in self._pdl_patterns:
            op.body.block.add_op(pattern_op)
            pdl_pass.apply(ctx, op)
            op.body.block.erase_op(pattern_op)

    def _pattern_to_xdsl(self, py_pattern: Callable, py_rewrite: Callable) -> func.FuncOp:
        """Create xDSL ``func.FuncOp``\ s from Python pattern and rewrite functions."""

        n_args = len(signature(py_pattern).parameters)
        if len(signature(py_rewrite).parameters) != n_args:
            raise ValueError("Search and rewrite patterns must have the same number of arguments.")

        # Rename functions so that their names in the xDSL module are known
        pattern_name = "__pattern"
        rewrite_name = "__rewrite"
        py_pattern.__name__ = pattern_name
        py_rewrite.__name__ = rewrite_name
        args = range(n_args)

        # Lower the functions and extract them from the IR
        @xdsl_from_qjit
        @qml.qjit
        @qml.qnode(qml.device("null.qubit", wires=n_args))
        def mod_fn():
            decomposition_rule(py_pattern, is_qreg=False, num_params=0)(*args)
            decomposition_rule(py_rewrite, is_qreg=False, num_params=0)(*args)

            return qml.state()

        mod = mod_fn()
        pattern_fn: func.FuncOp = SymbolTable.lookup_symbol(mod, pattern_name)
        rewrite_fn: func.FuncOp = SymbolTable.lookup_symbol(mod, rewrite_name)

        # Erase return op from the func bodies. These do not need to be matched.
        # This will make the funcs invalid, but that is fine since they will be
        # discarded after the PDL patterns are created.
        pattern_fn.body.block.erase_op(pattern_fn.body.block.last_op)
        rewrite_fn.body.block.erase_op(rewrite_fn.body.block.last_op)

        return pattern_fn, rewrite_fn

    def _create_pdl_pattern(self, pattern_fn: func.FuncOp, rewrite_fn: func.FuncOp):
        """Use xDSL functions containing patterns to match and rewrite to create a PDL PatternOp."""
        pdl_ops: list[pdl.OperationOp] = []
        starting_values: list[pdl.ValueType] = []
        cur_values: list[pdl.ValueType] = []
        value_map: dict[quantum.QubitSSAValue, int] = {}
        pdl_value_map: dict[pdl.ValueType, int] = {}
        type_map: dict[quantum.QubitSSAValue, pdl.TypeType] = {}

        # Create block containing the pattern we want to match. This will be the main
        # body of the pdl.PatternOp we're building.
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
                pdl_op = self._create_pdl_operation(
                    op=op,
                    cur_values=cur_values,
                    value_map=value_map,
                    pdl_value_map=pdl_value_map,
                    type_map=type_map,
                )
                pdl_ops.append(pdl_op)

        # Create block containing the rewrite pattern. This will be the block inside
        # a pdl.RewriteOp, which will be the terminator op of the pdl.PatternOp that
        # we're building.
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
                pdl_op = self._create_pdl_operation(
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
        return pattern_op

    def _create_pdl_operation(
        self,
        op: Operation,
        cur_values: list[pdl.ValueType],
        value_map: dict[quantum.QubitSSAValue, int],
        pdl_value_map: dict[pdl.ValueType, int],
        type_map: dict[quantum.QubitSSAValue, pdl.TypeType],
    ):
        """Create a pdl.OperationOp corresponding to the input xDSL operation. This method must be
        called from within an ``ImplicitBuilder`` context."""
        # Create operations corresponding to the operation attributes and properties.
        attr_vals = []
        attr_names = []
        for attr_name, attr in (op.properties | op.attributes).items():
            attr_names.append(builtin.StringAttr(attr_name))
            attr_vals.append(pdl.AttributeOp(attr))

        # Find values corresponding to the operands. For now, assume all operands and results
        # are qubits.
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

        # A pdl.OperationOp returns a pdl.OperationType. To use the values corresponding to
        # the results of the original operation, we must use pdl.ResultOp to extract them.
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

    @classmethod
    def create_from_serialized_options(cls, *_, **options):
        """Create a pass instance using serialized patterns."""
        paths = options["pattern_paths"]
        py_patterns = _patterns_from_paths(paths)
        pass_instance = cls(patterns=py_patterns)
        return pass_instance


pattern_match = compiler_transform(PatternMatchPass)


@pattern_match.custom_serialize_options
def _(*, patterns: dict[Callable, Callable] = {}):
    paths = []

    for pat, rw in patterns.items():
        # Create source for pattern and rewrite functions
        cur_program = _create_pattern_source(pat, rw)

        # Create tempfile with search pattern
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as cur_file:
            cur_file.write(cur_program)
            paths.append(cur_file.name)

    valued_options = {"pattern_paths": paths}
    return (), valued_options


def _create_pattern_source(pattern: Callable, rewrite: Callable) -> str:
    """Create a program represented as a string that encodes the ``pattern`` and ``rewrite``
    functions."""
    return ""


def _patterns_from_paths(paths: Sequence[str]) -> dict[Callable, Callable]:
    """Create pattern and rewrite functions using source files specified by ``paths``."""
    return {}
