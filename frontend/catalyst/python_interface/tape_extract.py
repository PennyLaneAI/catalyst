# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for extracting a tape from MLIR."""


from typing import Optional, Tuple, Any

from pennylane.tape import QuantumScript

from catalyst.python_interface.conversion import parse_generic_to_xdsl_module
from catalyst.python_interface.inspection.collector import QMLCollector
from catalyst.python_interface.inspection.xdsl_conversion import count_static_loop_iterations
from catalyst.python_interface.conversion import xdsl_from_qjit


class DialectSerializationError(Exception):
    """Raised when the quantum-dialect IR cannot be serialized to a QuantumScript."""


def tape_from_xdsl(
    module, *, shots: int | None = None, unroll: bool = True, max_unroll_iterations: int = 100
) -> QuantumScript:
    """Convert an xDSL ModuleOp (quantum dialect) -> flat QuantumScript."""
    module = _resolve_innermost_module(module)

    if unroll:
        _unroll_static_regions(module, max_iterations=max_unroll_iterations)

    _reject_residual_regions(module)

    try:
        ops, meas = QMLCollector(module).collect()
    except NotImplementedError as e:
        raise DialectSerializationError(f"Module contains un-serializable operations: {e}") from e

    return QuantumScript(ops, meas, shots=shots)


def tape_from_qjit(workflow, *args, shots: int | None = None, unroll: bool = True, **kwargs) -> QuantumScript:
    """Convert a @qml.qjit-compiled QNode -> drawable QuantumScript."""
    module = xdsl_from_qjit(workflow)(*args, **kwargs)
    return tape_from_xdsl(module, shots=shots, unroll=unroll)


def tape_from_mlir(mlir_str: str, *, shots: int | None = None, unroll: bool = True) -> QuantumScript:
    """Convert a generic MLIR string (quantum dialect) -> drawable QuantumScript."""
    module = parse_generic_to_xdsl_module(mlir_str)
    return tape_from_xdsl(module, shots=shots, unroll=unroll)


# ============================================================================
# Internal IR Manipulation Helpers
# ============================================================================

def _has_quantum_gates(module) -> bool:
    from xdsl.dialects.func import FuncOp
    for op in module.body.ops:
        if isinstance(op, FuncOp) and any(
            o.name.startswith("quantum.custom")
            for block in op.body.blocks for o in block.ops
        ):
            return True
    return False

def _resolve_innermost_module(module):
    """Find the ModuleOp that contains FuncOps with quantum gate operations."""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.dialects.func import FuncOp

    # Check nested modules first
    for op in module.body.ops:
        if isinstance(op, ModuleOp):
            result = _resolve_innermost_module(op)
            if result and _has_quantum_gates(result):
                return result

    if _has_quantum_gates(module):
        return module

    # Fallback: check if module (or nested) has *any* FuncOps
    for op in module.body.ops:
        if isinstance(op, ModuleOp):
            result = _resolve_innermost_module(op)
            if result and any(isinstance(o, FuncOp) for o in result.body.ops):
                return result

    return module if any(isinstance(op, FuncOp) for op in module.body.ops) else module


def _unroll_static_regions(module, max_iterations: int = 100) -> None:
    from xdsl.dialects import scf
    from xdsl.dialects.func import FuncOp

    for _ in range(max_iterations):
        changed = False
        for func_op in (op for op in module.body.ops if isinstance(op, FuncOp)):
            for block in func_op.body.blocks:
                for op in list(block.ops):
                    if isinstance(op, scf.ForOp):
                        try:
                            n_iters = count_static_loop_iterations(op)
                        except Exception:
                            continue  # Dynamic bounds skipped here

                        if n_iters == 0:
                            for res, init in zip(op.results, op.iter_args):
                                res.replace_all_uses_with(init)
                            op.detach(); op.erase()
                        else:
                            _inline_for_loop(op, n_iters, block)
                        changed = True; break

                    elif isinstance(op, scf.IfOp) and _try_resolve_constant_if(op):
                        changed = True; break
        if not changed:
            break


def _inline_for_loop(for_op, n_iters: int, parent_block) -> None:
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.rewriter import InsertPoint, Rewriter

    body_block = for_op.body.block
    lb_val = _extract_int_constant(for_op.lb)
    step_val = _extract_int_constant(for_op.step)

    # Fallback to Catalyst's resolve if basic extraction fails
    if lb_val is None or step_val is None:
        lb_val = int(resolve_constant_params(for_op.lb)) if lb_val is None else lb_val
        step_val = int(resolve_constant_params(for_op.step)) if step_val is None else step_val

    if lb_val is None or step_val is None:
        return

    rewriter = Rewriter()
    current_iter_vals = list(for_op.iter_args)

    for i in range(n_iters):
        iv_const = arith.ConstantOp(IntegerAttr(lb_val + i * step_val, body_block.args[0].type))
        rewriter.insert_op(iv_const, InsertPoint.before(for_op))

        body_ops = list(body_block.ops)
        terminator = body_ops[-1] if body_ops else None

        value_map = {body_block.args[0]: iv_const.result}
        value_map.update(dict(zip(body_block.args[1:], current_iter_vals)))

        for body_op in body_ops[:-1] if terminator else body_ops:
            cloned = body_op.clone()
            cloned.operands = tuple(value_map.get(op, op) for op in cloned.operands)
            value_map.update(dict(zip(body_op.results, cloned.results)))
            rewriter.insert_op(cloned, InsertPoint.before(for_op))

        if terminator:
            current_iter_vals = [value_map.get(op, op) for op in terminator.operands]

    for res, final_val in zip(for_op.results, current_iter_vals):
        res.replace_all_uses_with(final_val)

    for_op.detach()
    for_op.erase()


def _try_resolve_constant_if(if_op) -> bool:
    from xdsl.rewriter import InsertPoint, Rewriter

    # Try simple extraction first
    cond_val = _extract_int_constant(if_op.cond)

    # Fallback to Catalyst's robust constant resolution for JAX 0D tensors
    if cond_val is None:
        resolved = resolve_constant_params(if_op.cond)
        # Ensure it is a valid scalar value we can cast to bool
        if isinstance(resolved, (int, float, bool)):
            cond_val = bool(resolved)

    if cond_val is None:
        return False

    taken_region = if_op.true_region if cond_val else if_op.false_region
    if not taken_region.blocks:
        if_op.detach(); if_op.erase()
        return True

    taken_block = taken_region.blocks[0]
    ops_list = list(taken_block.ops)
    terminator = ops_list[-1] if ops_list else None

    rewriter, value_map = Rewriter(), {}

    for body_op in ops_list[:-1] if terminator else ops_list:
        cloned = body_op.clone()
        cloned.operands = tuple(value_map.get(op, op) for op in cloned.operands)
        value_map.update(dict(zip(body_op.results, cloned.results)))
        rewriter.insert_op(cloned, InsertPoint.before(if_op))

    if terminator:
        for res, op in zip(if_op.results, terminator.operands):
            res.replace_all_uses_with(value_map.get(op, op))

    if_op.detach()
    if_op.erase()
    return True


def _extract_int_constant(ssa_val) -> Optional[int]:
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.ir import Block

    owner = ssa_val.owner
    if not owner or isinstance(owner, Block):
        return None
    if getattr(owner, "name", "") == "arith.constant" and isinstance(owner.value, IntegerAttr):
        return owner.value.value.data
    if hasattr(owner, "value") and isinstance(owner.value, IntegerAttr):
        return owner.value.value.data
    return None


def _reject_residual_regions(module) -> None:
    from xdsl.dialects import scf
    from xdsl.dialects.func import FuncOp

    for func_op in (o for o in module.body.ops if isinstance(o, FuncOp)):
        for block in func_op.body.blocks:
            for op in block.ops:
                if isinstance(op, scf.ForOp):
                    raise DialectSerializationError("Residual scf.for with dynamic bounds after unrolling.")
                if isinstance(op, scf.WhileOp):
                    raise DialectSerializationError("scf.while cannot be statically unrolled.")
                if isinstance(op, scf.IfOp):
                    raise DialectSerializationError("Residual scf.if with dynamic condition.")
                if len(op.regions) > 0 and op.name != "func.func":
                    raise DialectSerializationError(f"Residual operation with regions: {op.name}")

