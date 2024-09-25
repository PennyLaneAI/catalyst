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
""" Jax extras module containing functions related to the StableHLO lowering """

from __future__ import annotations

import logging

import jax
from jax._src.dispatch import jaxpr_replicas
from jax._src.effects import ordered_effects as jax_ordered_effects
from jax._src.interpreters.mlir import _module_name_regex
from jax._src.lax.lax import xla
from jax._src.sharding_impls import ReplicaAxisContext
from jax._src.source_info_util import new_name_stack
from jax._src.util import wrap_name
from jax.core import ClosedJaxpr
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
from catalyst.utils.patching import Patcher

# pylint: disable=protected-access

__all__ = ("jaxpr_to_mlir", "custom_lower_jaxpr_to_module")

from catalyst.jax_extras.patches import _no_clean_up_dead_vars, get_aval2

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@debug_logger
def jaxpr_to_mlir(func_name, jaxpr):
    """Lower a Jaxpr into an MLIR module.

    Args:
        func_name(str): function name
        jaxpr(Jaxpr): Jaxpr code to lower

    Returns:
        module: the MLIR module corresponding to ``func``
        context: the MLIR context corresponding
    """

    with Patcher(
        (jax._src.interpreters.partial_eval, "get_aval", get_aval2),
        (jax._src.core, "clean_up_dead_vars", _no_clean_up_dead_vars),
    ):
        nrep = jaxpr_replicas(jaxpr)
        effects = jax_ordered_effects.filter_in(jaxpr.effects)
        axis_context = ReplicaAxisContext(xla.AxisEnv(nrep, (), ()))
        name_stack = new_name_stack(wrap_name("ok", "jit"))
        module, context = custom_lower_jaxpr_to_module(
            func_name="jit_" + func_name,
            module_name=func_name,
            jaxpr=jaxpr,
            effects=effects,
            platform="cpu",
            axis_context=axis_context,
            name_stack=name_stack,
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
    name_stack,
    replicated_args=None,
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
        backend_or_name=None,
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
        lower_jaxpr_to_fun(
            ctx,
            func_name,
            jaxpr,
            effects,
            public=True,
            create_tokens=True,
            replace_tokens_with_dummy=True,
            replicated_args=replicated_args,
            arg_shardings=arg_shardings,
            result_shardings=result_shardings,
            name_stack=name_stack,
        )

        worklist = [*ctx.module.body.operations]
        while worklist:
            op = worklist.pop()
            func_name = str(op.name)
            is_entry_point = func_name.startswith('"jit_')
            if is_entry_point:
                continue
            if isinstance(op, FuncOp):
                op.attributes["llvm.linkage"] = ir.Attribute.parse("#llvm.linkage<internal>")
            if isinstance(op, ModuleOp):
                worklist += [*op.body.operations]

    return ctx.module, ctx.context
