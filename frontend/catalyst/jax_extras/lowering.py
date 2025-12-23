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
"""Jax extras module containing functions related to the StableHLO lowering"""

from __future__ import annotations

import logging
import textwrap

import jax
from jax._src import core
from jax._src.effects import ordered_effects as jax_ordered_effects
from jax._src.interpreters.mlir import _module_name_regex
from jax._src.interpreters.pxla import _jaxpr_replicas as jaxpr_replicas
from jax._src.sharding_impls import AxisEnv, ReplicaAxisContext
from jax.extend.core import ClosedJaxpr
from jax.interpreters.mlir import (
    AxisContext,
    LoweringParameters,
    ModuleContext,
    ir,
    lower_jaxpr_to_fun,
    lowerable_effects,
)
from jaxlib.mlir.dialects.builtin import ModuleOp
from jaxlib.mlir.dialects.func import FuncOp

import catalyst
from catalyst.logging import debug_logger
from catalyst.utils.exceptions import CompileError
from catalyst.utils.patching import Patcher

# pylint: disable=protected-access

__all__ = ("jaxpr_to_mlir", "custom_lower_jaxpr_to_module")

from catalyst.jax_extras.patches import (
    _no_clean_up_dead_vars,
    get_aval2,
    patched_multi_broadcast_in_dim,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@debug_logger
def jaxpr_to_mlir(jaxpr, func_name, arg_names):
    """Lower a Jaxpr into an MLIR module.

    Args:
        jaxpr(Jaxpr): Jaxpr code to lower
        func_name(str): function name
        arg_names(list[str]): list of argument names

    Returns:
        module: the MLIR module corresponding to ``func``
        context: the MLIR context corresponding
    """

    with Patcher(
        (jax._src.interpreters.partial_eval, "get_aval", get_aval2),
        (jax._src.core, "clean_up_dead_vars", _no_clean_up_dead_vars),
        (jax._src.interpreters.mlir, "multi_broadcast_in_dim", patched_multi_broadcast_in_dim),
    ):
        nrep = jaxpr_replicas(jaxpr)
        effects = jax_ordered_effects.filter_in(jaxpr.effects)
        axis_context = ReplicaAxisContext(AxisEnv(nrep, (), ()))
        module, context = custom_lower_jaxpr_to_module(
            func_name="jit_" + func_name,
            module_name=func_name,
            jaxpr=jaxpr,
            effects=effects,
            platform="cpu",
            axis_context=axis_context,
            arg_names=arg_names,
        )

    return module, context


# pylint: disable=too-many-arguments
@debug_logger
def custom_lower_jaxpr_to_module(
    func_name: str,
    module_name: str,
    jaxpr: ClosedJaxpr,
    effects,
    platform: str,
    axis_context: AxisContext,
    replicated_args=None,
    arg_names=None,
    arg_shardings=None,
    result_shardings=None,
):
    """Lowers a top-level jaxpr to an MHLO module.

    Handles the quirks of the argument/return value passing conventions of the
    runtime.

    This function has been modified from its original form in the JAX project at
    https://github.com/google/jax/blob/c4d590b1b640cc9fcfdbe91bf3fe34c47bcde917/jax/interpreters/mlir.py#L625version
    released under the Apache License, Version 2.0, with the following copyright notice:

    Copyright 2021 The JAX Authors.
    """

    if any(lowerable_effects.filter_not_in(jaxpr.effects)):  # pragma: no cover
        raise ValueError(f"Cannot lower jaxpr with effects: {jaxpr.effects}")

    assert platform == "cpu"
    assert arg_shardings is None
    assert result_shardings is None

    # MHLO channels need to start at 1
    channel_iter = 1
    # Create a keepalives list that will be mutated during the lowering.
    keepalives = []
    host_callbacks = []
    custom_lowering_rules = catalyst.jax_primitives.CUSTOM_LOWERING_RULES
    lowering_params = LoweringParameters(override_lowering_rules=custom_lowering_rules)
    ctx = ModuleContext(
        backend=None,
        platforms=[platform],
        axis_context=axis_context,
        keepalives=keepalives,
        channel_iterator=channel_iter,
        host_callbacks=host_callbacks,
        lowering_parameters=lowering_params,
    )
    ctx.context.allow_unregistered_dialects = True
    with ctx.context, ir.Location.unknown(ctx.context):
        # register_dialect()
        # Remove module name characters that XLA would alter. This ensures that
        # XLA computation preserves the module name.
        module_name = _module_name_regex.sub("_", module_name)
        ctx.module.operation.attributes["sym_name"] = ir.StringAttr.get(module_name)

        const_args = core.jaxpr_const_args(jaxpr.jaxpr)
        const_arg_avals = [core.shaped_abstractify(c) for c in const_args]
        num_const_args = len(const_arg_avals)
        in_avals = const_arg_avals + jaxpr.in_avals

        # Use main_function=False to preserve the function name (e.g., "jit_func")
        # instead of renaming it to "main"
        lower_jaxpr_to_fun(
            ctx,
            func_name,
            jaxpr,
            effects,
            num_const_args=num_const_args,
            in_avals=in_avals,
            main_function=False,
            replicated_args=replicated_args,
            arg_names=arg_names,
            arg_shardings=arg_shardings,
            result_shardings=result_shardings,
        )

        # Set the entry point function visibility to public and other functions to internal
        worklist = [*ctx.module.body.operations]
        while worklist:
            op = worklist.pop()
            func_name = str(op.name)
            is_entry_point = func_name.startswith('"jit_')

            if is_entry_point:
                # Keep entry point functions public
                op.attributes["sym_visibility"] = ir.StringAttr.get("public")
                continue
            if isinstance(op, FuncOp):
                # Set non-entry functions to internal linkage
                op.attributes["llvm.linkage"] = ir.Attribute.parse("#llvm.linkage<internal>")
            if isinstance(op, ModuleOp):
                worklist += [*op.body.operations]

    return ctx.module, ctx.context


def get_mlir_attribute_from_pyval(value):
    """
    Given a value of any type, construct an mlir attribute of corresponding type.

    We set up the context and location outside because recursive calls to this function
    will segfault if multiple `Context()`s are instantiated.
    """

    attr = None
    match value:
        case bool():
            attr = ir.BoolAttr.get(value)

        case int():
            if -9223372036854775808 <= value < 0:  # 2**63
                attr = ir.IntegerAttr.get(ir.IntegerType.get_signed(64), value)
            elif 0 <= value < 18446744073709551616:  # = 2**64
                attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
            else:
                raise CompileError(
                    textwrap.dedent(
                        """
                    Large interger attributes currently not supported in MLIR,
                    see https://github.com/llvm/llvm-project/issues/128072
                    """
                    )
                )

        case float():
            attr = ir.FloatAttr.get(ir.F64Type.get(), value)

        case str():
            attr = ir.StringAttr.get(value)

        case list() | tuple():
            element_attrs = [get_mlir_attribute_from_pyval(elem) for elem in value]
            attr = ir.ArrayAttr.get(element_attrs)

        case dict():
            named_attrs = {}
            for k, v in value.items():
                if not isinstance(k, str):
                    raise CompileError(
                        f"Dictionary keys for MLIR DictionaryAttr must be strings, got: {type(k)}"
                    )
                named_attrs[k] = get_mlir_attribute_from_pyval(v)
            attr = ir.DictAttr.get(named_attrs)

        case _:
            raise CompileError(f"Cannot convert Python type {type(value)} to an MLIR attribute.")

    return attr
