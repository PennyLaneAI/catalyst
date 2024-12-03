# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains some helper functions for translating JAX
primitives to MLIR"""

import copy

import pennylane as qml
from jax._src import core, util
from jax._src.lib.mlir import ir
from jax.interpreters import mlir
from jaxlib.mlir.dialects.builtin import ModuleOp
from jaxlib.mlir.dialects.func import CallOp
from mlir_quantum.dialects._transform_ops_gen import NamedSequenceOp, YieldOp
from mlir_quantum.dialects.catalyst import LaunchKernelOp


def get_or_create_funcop(ctx, callable_, call_jaxpr):
    """Get funcOp from cache, or create it from scratch"""
    if func_op := ctx.module_context.cached_primitive_lowerings.get(callable_):
        return func_op
    func_op = lower_callable_to_funcop(ctx, callable_, call_jaxpr)
    ctx.module_context.cached_primitive_lowerings[callable_] = func_op
    return func_op


def get_or_create_qnode_funcop(ctx, callable_, call_jaxpr):
    """A wrapper around lower_qnode_to_funcop that will cache the FuncOp.

    Args:
      ctx: LoweringRuleContext
      callable_: qml.Qnode
      call_jaxpr: jaxpr representing callable_
    Returns:
      FuncOp
    """
    if func_op := ctx.module_context.cached_primitive_lowerings.get(callable_):
        return func_op
    func_op = lower_qnode_to_funcop(ctx, callable_, call_jaxpr)
    ctx.module_context.cached_primitive_lowerings[callable_] = func_op
    return func_op


def get_symbolref(ctx, func_op):
    """Get symbolref by deciding whether to constructo a symbolref or flatsymbolref"""
    is_call_same_module = ctx.module_context.module.operation == func_op.parent
    if is_call_same_module:
        return ir.FlatSymbolRefAttr.get(func_op.name.value)
    parent = func_op.parent
    parent_name = parent.operation.attributes["sym_name"].value
    child_name = func_op.name.value
    return ir.SymbolRefAttr.get([parent_name, child_name])


def create_call_op(ctx, func_op, *args):
    """Create a func::CallOp from JAXPR."""
    output_types = list(map(mlir.aval_to_ir_types, ctx.avals_out))
    flat_output_types = util.flatten(output_types)
    mlir_args = mlir.flatten_lowering_ir_args(args)
    symbol_ref = get_symbolref(ctx, func_op)
    is_call_same_module = ctx.module_context.module.operation == func_op.parent
    constructor = CallOp if is_call_same_module else LaunchKernelOp
    return constructor(flat_output_types, symbol_ref, mlir_args)


def create_module_op(ctx, name):
    """Create a module with name name"""

    symbol_table = ctx.module_context.symbol_table
    parent = ctx.module_context.module
    with ir.InsertionPoint(parent.body):
        module = ModuleOp()
        symbol_attr = ir._symbolNameAttr(name, ctx.module_context.context)
        module.operation.attributes["sym_name"] = symbol_attr
        symbol_table.insert(module)

    return module


def lower_callable(ctx, callable_, call_jaxpr):
    """Lowers _callable to MLIR.

    If callable_ is a qnode, then we will first create a module, then
    create a FuncOp corresponding to call_jaxpr. Otherwise, a FuncOp
    will be created in the current module. This function might
    add more than one FuncOps. This depends on the contents of call_jaxpr.

    Args:
      ctx: LoweringRuleContext
      callable_: python function
      call_jaxpr: jaxpr representing callable_
    Returns:
      FuncOp
    """
    if not isinstance(callable_, qml.QNode):
        return get_or_create_funcop(ctx, callable_, call_jaxpr)

    return get_or_create_qnode_funcop(ctx, callable_, call_jaxpr)


def lower_callable_to_funcop(ctx, callable_, call_jaxpr):
    """Lower callable to either a FuncOp"""
    if isinstance(call_jaxpr, core.Jaxpr):
        call_jaxpr = core.ClosedJaxpr(call_jaxpr, ())

    kwargs = {}
    kwargs["ctx"] = ctx.module_context
    kwargs["name"] = callable_.__name__
    kwargs["jaxpr"] = call_jaxpr
    kwargs["effects"] = []
    kwargs["name_stack"] = ctx.name_stack
    func_op = mlir.lower_jaxpr_to_fun(**kwargs)

    if isinstance(callable_, qml.QNode):
        func_op.attributes["qnode"] = ir.UnitAttr.get()
        # "best", the default option in PennyLane, chooses backprop on the device
        # if supported and parameter-shift otherwise. Emulating the same behaviour
        # would require generating code to query the device.
        # For simplicity, Catalyst instead defaults to parameter-shift.
        diff_method = (
            "parameter-shift" if callable_.diff_method == "best" else str(callable_.diff_method)
        )
        func_op.attributes["diff_method"] = ir.StringAttr.get(diff_method)

    return func_op


class NestedModule:
    """Context manager for the nested module"""

    def __init__(self, ctx, name):
        self.ctx = ctx
        self.moduleOp = create_module_op(ctx, name)
        self.old_module_context = ctx.module_context

    def __enter__(self):
        self.ctx.module_context = copy.copy(self.ctx.module_context)
        self.ctx.module_context.module = self.moduleOp
        self.ctx.module_context.cached_primitive_lowerings = {}
        return self.moduleOp

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ctx.module_context = self.old_module_context


def lower_qnode_to_funcop(ctx, callable_, call_jaxpr):
    """Lowers callable_ to MLIR.

    Will create ModuleOp and then lower the callable_ to a
    FuncOp inside the ModuleOp. The ModuleOp may have more
    than one FuncOp. This depends on the contents of call_jaxpr.

    Args:
      ctx: LoweringRuleContext
      callable_: qml.Qnode
      call_jaxpr: jaxpr representing callable_
    Returns:
      FuncOp
    """
    assert isinstance(callable_, qml.QNode), "This function expects qnodes"

    name = "module_" + callable_.__name__
    # pylint: disable-next=no-member
    with NestedModule(ctx, name) as module, ir.InsertionPoint(module.regions[0].blocks[0]) as ip:
        transform_named_sequence_lowering(ctx)
        ctx.module_context.ip = ip
        func_op = get_or_create_funcop(ctx, callable_, call_jaxpr)
        func_op.sym_visibility = ir.StringAttr.get("public")

    return func_op


def transform_named_sequence_lowering(jax_ctx: mlir.LoweringRuleContext):
    transform_mod_type = ir.OpaqueType.get("transform", 'op<"builtin.module">')
    module = jax_ctx.module_context.module

    # We wish to generate the transformer module, and place it in the top-level module
    # The transformer module must be marked with the "transform.with_named_sequence" attribute
    # The transformer module has a single block, and the block contains the
    # "transform.named_sequence @__transform_main" operation

    with ir.InsertionPoint(module.body):
        transformer_module = ModuleOp()
        with_named_sequence_attr = ir.UnitAttr.get(jax_ctx.module_context.context)
        transformer_module.operation.attributes["transform.with_named_sequence"] = (
            with_named_sequence_attr
        )
        bb_transformer = transformer_module.body

    functype = ir.FunctionType.get(inputs=[transform_mod_type], results=[])
    functype_attr = ir.TypeAttr.get(functype)

    # Insert the transform.named_sequence op into the transformer module
    # Note that InsertionPoint(Block) inserts after the last operation but still inside the block.
    with ir.InsertionPoint(bb_transformer):
        named_sequence_op = NamedSequenceOp(
            sym_name="__transform_main",
            function_type=functype_attr,
        )

        # transform.named_sequence op is the "main function" of the transform dialect
        # and thus needs an entry block (which also should be its only block)
        # The argument of the block is the payload module
        bb_named_sequence = ir.Block.create_at_start(
            named_sequence_op.body, arg_types=[transform_mod_type]
        )

        # The transform.named_sequence needs a terminator called "transform.yield"
        with ir.InsertionPoint(bb_named_sequence):
            transform_yield_op = YieldOp(operands_=[])  # pylint: disable=unused-variable

    return named_sequence_op.results
