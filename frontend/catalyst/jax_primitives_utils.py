# Copyright 2024 Xanadu Quantum Technologies Inc.

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
import functools

import pennylane as qml
from jax._src import core, util
from jax._src.lib.mlir import ir
from jax.interpreters import mlir
from jaxlib.mlir.dialects.builtin import ModuleOp
from jaxlib.mlir.dialects.func import CallOp
from mlir_quantum.dialects._transform_ops_gen import ApplyRegisteredPassOp, NamedSequenceOp, YieldOp
from mlir_quantum.dialects.catalyst import LaunchKernelOp

from catalyst.jax_extras.lowering import get_mlir_attribute_from_pyval


def _only_single_expval(call_jaxpr: core.ClosedJaxpr) -> bool:
    found_expval = False
    for eqn in call_jaxpr.eqns:
        name = eqn.primitive.name
        if name in {"probs", "counts", "sample"}:
            return False
        elif name == "expval":
            if found_expval:
                return False
            found_expval = True
    return True


def _calculate_diff_method(qn: qml.QNode, call_jaxpr: core.ClosedJaxpr):
    diff_method = str(qn.diff_method)
    if diff_method != "best":
        return diff_method

    device_name = getattr(getattr(qn, "device", None), "name", None)

    if device_name and "lightning" in device_name and _only_single_expval(call_jaxpr):
        return "adjoint"
    return "parameter-shift"


def get_call_jaxpr(jaxpr):
    """Extracts the `call_jaxpr` from a JAXPR if it exists.""" ""
    for eqn in jaxpr.eqns:
        if eqn.params.get("call_jaxpr"):
            return eqn.params["call_jaxpr"]
    raise AssertionError("No call_jaxpr found in the JAXPR.")


def get_call_equation(jaxpr):
    """Extracts the equation which has a call_jaxpr."""
    for eqn in jaxpr.eqns:
        if eqn.params.get("call_jaxpr"):
            return eqn
    raise AssertionError("No call_jaxpr found in the JAXPR.")


def lower_jaxpr(ctx, jaxpr, metadata=None, fn=None):
    """Lowers a call primitive jaxpr, may be either func_p or quantum_kernel_p

    Args:
        ctx: LoweringRuleContext
        jaxpr: JAXPR to be lowered
        metadata: additional metadata to distinguish different FuncOps
        fn (Callable | None): the function the jaxpr corresponds to. Used for naming and caching.

    Returns:
        FuncOp
    """

    if fn is None or isinstance(fn, qml.QNode):
        equation = get_call_equation(jaxpr)
        call_jaxpr = equation.params["call_jaxpr"]
        pipeline = equation.params.get("pipeline")
        callable_ = equation.params.get("fn")
        if callable_ is None:
            callable_ = equation.params.get("qnode", None)
    else:
        call_jaxpr = jaxpr
        pipeline = ()
        callable_ = fn

    return lower_callable(ctx, callable_, call_jaxpr, pipeline=pipeline, metadata=metadata)


# pylint: disable=too-many-arguments, too-many-positional-arguments
def lower_callable(ctx, callable_, call_jaxpr, pipeline=(), metadata=None, public=False):
    """Lowers _callable to MLIR.

    If callable_ is a qnode, then we will first create a module, then
    create a FuncOp corresponding to call_jaxpr. Otherwise, a FuncOp
    will be created in the current module. This function might
    add more than one FuncOps. This depends on the contents of call_jaxpr.

    Args:
      ctx: LoweringRuleContext
      callable_: python function
      call_jaxpr: jaxpr representing callable_
      public: whether the visibility should be marked public

    Returns:
      FuncOp
    """
    if pipeline is None:
        pipeline = tuple()

    if isinstance(callable_, qml.QNode):
        return get_or_create_qnode_funcop(ctx, callable_, call_jaxpr, pipeline, metadata=metadata)
    return get_or_create_funcop(
        ctx, callable_, call_jaxpr, pipeline, metadata=metadata, public=public
    )


# pylint: disable=too-many-arguments, too-many-positional-arguments
def get_or_create_funcop(ctx, callable_, call_jaxpr, pipeline, metadata=None, public=False):
    """Get funcOp from cache, or create it from scratch

    Args:
        ctx: LoweringRuleContext
        callable_: python function
        call_jaxpr: jaxpr representing callable_
        metadata: additional metadata to distinguish different FuncOps
        public: whether the visibility should be marked public

    Returns:
        FuncOp
    """
    if metadata is None:
        metadata = tuple()
    key = (callable_, *metadata, *pipeline)
    if callable_ is not None:
        if func_op := get_cached(ctx, key):
            return func_op
    func_op = lower_callable_to_funcop(ctx, callable_, call_jaxpr, public=public)
    cache(ctx, key, func_op)
    return func_op


def lower_callable_to_funcop(ctx, callable_, call_jaxpr, public=False):
    """Lower callable to either a FuncOp

    Args:
        ctx: LoweringRuleContext
        callable_: python function
        call_jaxpr: jaxpr representing callable_
        public: whether the visibility should be marked public

    Returns:
        FuncOp
    """
    if isinstance(call_jaxpr, core.Jaxpr):
        call_jaxpr = core.ClosedJaxpr(call_jaxpr, ())

    kwargs = {}
    kwargs["ctx"] = ctx.module_context
    if isinstance(callable_, functools.partial):
        name = callable_.func.__name__ + ".partial"
    else:
        name = callable_.__name__

    kwargs["name"] = name
    kwargs["jaxpr"] = call_jaxpr
    kwargs["effects"] = []
    kwargs["main_function"] = False

    const_args = core.jaxpr_const_args(call_jaxpr.jaxpr)
    const_arg_avals = [core.shaped_abstractify(c) for c in const_args]
    num_const_args = len(const_arg_avals)

    kwargs["in_avals"] = const_arg_avals + call_jaxpr.in_avals
    kwargs["num_const_args"] = num_const_args

    func_op = mlir.lower_jaxpr_to_fun(**kwargs)
    if public:
        func_op.attributes["sym_visibility"] = ir.StringAttr.get("public")

    if isinstance(callable_, qml.QNode):
        func_op.attributes["qnode"] = ir.UnitAttr.get()

        diff_method = _calculate_diff_method(callable_, call_jaxpr)

        func_op.attributes["diff_method"] = ir.StringAttr.get(diff_method)

        # Register the decomposition gatesets to the QNode FuncOp
        # This will set a queue of gatesets that enables support for multiple
        # levels of decomposition in the MLIR decomposition pass
        if gateset := getattr(callable_, "decompose_gatesets", []):
            func_op.attributes["decompose_gatesets"] = get_mlir_attribute_from_pyval(gateset)

    # Extract the target gate and number of wires from decomposition rules
    # and set them as attributes on the FuncOp for use in the MLIR decomposition pass
    if target_gate := getattr(callable_, "target_gate", None):
        func_op.attributes["target_gate"] = get_mlir_attribute_from_pyval(target_gate)
    if num_wires := getattr(callable_, "num_wires", None):
        func_op.attributes["num_wires"] = get_mlir_attribute_from_pyval(num_wires)

    return func_op


def get_or_create_qnode_funcop(ctx, callable_, call_jaxpr, pipeline, metadata):
    """A wrapper around lower_qnode_to_funcop that will cache the FuncOp.

    Args:
      ctx: LoweringRuleContext
      callable_: qml.Qnode
      call_jaxpr: jaxpr representing callable_
    Returns:
      FuncOp
    """
    if metadata is None:
        metadata = tuple()
    if callable_.static_argnums:
        return lower_qnode_to_funcop(ctx, callable_, call_jaxpr, pipeline)
    key = (callable_, *metadata, *pipeline)
    if func_op := get_cached(ctx, key):
        return func_op
    func_op = lower_qnode_to_funcop(ctx, callable_, call_jaxpr, pipeline)
    cache(ctx, key, func_op)
    return func_op


def lower_qnode_to_funcop(ctx, callable_, call_jaxpr, pipeline):
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
        transform_named_sequence_lowering(ctx, pipeline)
        ctx.module_context.ip = ip
        func_op = get_or_create_funcop(ctx, callable_, call_jaxpr, pipeline)
        func_op.sym_visibility = ir.StringAttr.get("public")

    return func_op


def get_cached(ctx, key):
    """Looks for key in the cache"""
    return ctx.module_context.cached_primitive_lowerings.get(key)


def cache(ctx, key, val):
    """Caches value in cache with key"""
    ctx.module_context.cached_primitive_lowerings[key] = val


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
    mlir_args = mlir.flatten_ir_values(args)
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


def _lowered_options(args, kwargs):
    lowered_options = {}
    for arg in args:
        lowered_options[str(arg)] = get_mlir_attribute_from_pyval(True)
    for option, value in kwargs.items():
        mlir_option = str(option).replace("_", "-")
        lowered_options[mlir_option] = get_mlir_attribute_from_pyval(value)
    return lowered_options


def transform_named_sequence_lowering(jax_ctx: mlir.LoweringRuleContext, pipeline):
    """Generate a transform module embedded in the current module and schedule
    the transformations in pipeline"""

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

    # Track if we created any xDSL passes
    uses_xdsl_passes = False

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
            target = bb_named_sequence.arguments[0]
            for _pass in pipeline:
                if isinstance(_pass, qml.transforms.core.TransformContainer):
                    options = _lowered_options(_pass.args, _pass.kwargs)
                    name = _pass.pass_name
                else:
                    options = _pass.get_options()
                    name = _pass.name
                apply_registered_pass_op = ApplyRegisteredPassOp(
                    result=transform_mod_type,
                    target=target,
                    pass_name=name,
                    options=options,
                    dynamic_options={},
                )
                target = apply_registered_pass_op.result

                try:
                    # pylint: disable=import-outside-toplevel
                    from catalyst.python_interface.pass_api import is_xdsl_pass

                    # catalyst.python_interface.xdsl_universe collects all transforms in
                    # catalyst.python_interface.transforms, so importing from that file
                    # updates the global xDSL transforms registry.
                    from catalyst.python_interface.xdsl_universe import XDSL_UNIVERSE as _

                    if is_xdsl_pass(name):
                        uses_xdsl_passes = True
                        apply_registered_pass_op.operation.attributes["catalyst.xdsl_pass"] = (
                            ir.UnitAttr.get()
                        )
                except ModuleNotFoundError:
                    # If xDSL pass API is not available, do not set the attribute
                    pass

            transform_yield_op = YieldOp(operands_=[])  # pylint: disable=unused-variable

    # Set an attribute on the transformer module if we created any xDSL pass operations
    if uses_xdsl_passes:
        transformer_module.operation.attributes["catalyst.uses_xdsl_passes"] = ir.UnitAttr.get()

    return named_sequence_op.results
