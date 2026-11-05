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


"""Compiler entry points for decomposition-rule preparation and compilation.

``rule_preparation`` produces batches consumed by QNode capture or by the standalone
MLIR helpers in ``rule_compilation``. This module coordinates the standalone path
and preserves existing helper imports and callback names, including the entry point
loaded by the C++ compiler.
"""

from jax._src.lib.mlir import ir

from catalyst.decomposition.capture_session import OpDecompRequest
from catalyst.decomposition.graph_op_id import GraphOpID
from catalyst.decomposition.modifier_state import ModifierState
from catalyst.decomposition.rule_compilation import (
    build_rule_module,
    get_rule_strings_from_module,
    get_rules_from_module,
    materialize_decomp_target_spec,
    materialize_rule_strings,
)
from catalyst.decomposition.rule_preparation import (
    CollectedRule,
    DecompRuleSpec,
    DecompTargetSpec,
    RuleTraceInputs,
    _adjoint_folds_to_base,
    _append_decomp_target_spec,
    _resources_have_measurement,
    _rule_allocates_work_wires,
    _rule_is_eligible,
    build_base_op,
    build_decomp_rule_variants,
    build_decomp_target_spec,
    build_registered_decomp_target_spec,
    collect_resources_for_op,
    collect_symbolic_resources,
    flatten_hybrid_args,
    ordered_kwarg_names,
    prepare_dummy_rule_arguments,
    prepare_dynamic_op_kwargs,
    rule_call_operands,
    symbolic_arguments,
    symbolic_op_name,
    unpack_rule_operands,
    walk_reachable_decomp_rule_sets,
)
from catalyst.decomposition.type_utils import SerializedGraphOpID

__all__ = (
    "CollectedRule",
    "DecompRuleSpec",
    "DecompTargetSpec",
    "GraphOpID",
    "RuleTraceInputs",
    "build_base_op",
    "build_decomp_rule_variants",
    "build_decomp_target_spec",
    "build_registered_decomp_target_spec",
    "build_rule_module",
    "collect_resources_for_op",
    "collect_symbolic_resources",
    "compile_decomposition_rules",
    "compile_decomposition_rules_wrapper",
    "compile_reachable_decomposition_rules_wrapper",
    "compile_registered_symbolic_rules",
    "flatten_hybrid_args",
    "get_rule_strings_from_module",
    "get_rules_from_module",
    "materialize_decomp_target_spec",
    "materialize_reachable_rule_strings",
    "ordered_kwarg_names",
    "prepare_dummy_rule_arguments",
    "prepare_dynamic_op_kwargs",
    "rule_call_operands",
    "symbolic_arguments",
    "symbolic_op_name",
    "unpack_rule_operands",
    "walk_reachable_decomp_rule_sets",
)


def compile_decomposition_rules(
    op_name: str,
    op_id: SerializedGraphOpID,
    dynamic_shape,
    wire_lens,
    static_data,
    extra_data=None,
    is_custom_op=False,
    modifiers=ModifierState(),
) -> ir.Operation:
    """Compile decomposition rules into the legacy standalone MLIR module."""

    target_spec = build_decomp_target_spec(
        op_name,
        op_id,
        dynamic_shape,
        wire_lens,
        static_data,
        extra_data=extra_data,
        is_custom_op=is_custom_op,
        modifiers=modifiers,
    )
    return materialize_decomp_target_spec(target_spec)


def compile_registered_symbolic_rules(
    op_name: str,
    op_id: SerializedGraphOpID,
    dynamic_shape,
    wire_lens,
    static_data,
    extra_data=None,
    is_custom_op=False,
    op_cls=None,
    *,
    modifiers,
) -> ir.Operation | None:
    """Compile registered symbolic rules into the legacy standalone MLIR module."""

    target_spec = build_registered_decomp_target_spec(
        op_name,
        op_id,
        dynamic_shape,
        wire_lens,
        static_data,
        extra_data=extra_data,
        is_custom_op=is_custom_op,
        op_cls=op_cls,
        modifiers=modifiers,
    )
    if target_spec is None:
        return None
    return materialize_decomp_target_spec(target_spec)


def compile_decomposition_rules_wrapper(
    op_name: str,
    op_id: str,
    dynamic_shape,
    wire_lens,
    static_data,
    extra_data=None,
    is_custom_op=False,
) -> str:
    """Return a string MLIR module containing the decomposition rules for an operator instance.

    Args:
        op_name (str): the operator's name, as PennyLane registers its rules
        op_id (str): the operator's graphOpId
        dynamic_shape (dict): dynamic argument names to their MLIR types
        wire_lens (dict): wire argument names to their lengths
        static_data (dict): compiler-static argument names to their values
        extra_data (dict): argument values the graphOpId identifies by UID instead of spelling
        is_custom_op (bool): whether the operator lowers to ``qref.custom``

    Returns:
        str: the module holding the rules
    """
    return str(
        compile_decomposition_rules(
            op_name,
            SerializedGraphOpID(op_id),
            dynamic_shape,
            wire_lens,
            static_data,
            extra_data=extra_data,
            is_custom_op=is_custom_op,
        )
    )


def compile_reachable_decomposition_rules_wrapper(
    op_name: str,
    op_id: str,
    dynamic_shape,
    wire_lens,
    static_data,
    extra_data=None,
    is_custom_op=False,
) -> str:
    """Return an MLIR module with the full reachable decomposition-rule closure for an operator.

    This is the entry point for the compiler's *on-demand* rule loader (``loadPythonDecomps`` ->
    ``pythonRuleLowering``). The callback currently receives the op's *base* name
    (``getOperatorName()``) together with its full graphOpId (``getGraphOpId()``). Two things matter
    here:

    * **Modifier ids.** For a plain op the name and id agree (``"S"`` / ``"S{...}"``). For an
      op-level modifier the graphOpId is name-wrapped while the name stays the base (``"S"``). We
      recover the base id and modifier state so discovery starts from that variant.
    * **The whole closure, not just this op's direct rules.** The loader does not recurse into a
      rule's resource ops, so it needs every rule reachable from this op down to the gate set in one
      shot. For ``Adjoint(S)`` that includes ``Adjoint(S) -> Adjoint(PhaseShift)`` *and*
      ``Adjoint(PhaseShift) -> PhaseShift``; returning only the first would leave the solver unable
      to complete a path. :func:`materialize_reachable_rule_strings` builds that closure
      (base + adjoint-registered + distributed-adjoint rules, transitively) and each returned func
      keeps its own ``target_gate``, which is how the loader registers them.

    Args:
        op_name (str): the operator's *base* name
        op_id (str): the operator's full graphOpId, modifiers included
        dynamic_shape (dict): dynamic argument names to their MLIR types
        wire_lens (dict): wire argument names to their lengths
        static_data (dict): compiler-static argument names to their values
        extra_data (dict): argument values the graphOpId identifies by UID instead of spelling
        is_custom_op (bool): whether the operator lowers to ``qref.custom``

    Returns:
        str: a module holding the whole reachable rule closure
    """
    base_id, modifiers = ModifierState.from_graph_op_id(op_name, SerializedGraphOpID(op_id))

    rule_strings = materialize_reachable_rule_strings(
        op_name=op_name,
        op_id=base_id,
        dynamic_shape=dynamic_shape,
        wire_lens=wire_lens,
        static_data=static_data,
        extra_data=extra_data,
        is_custom_op=is_custom_op,
        modifiers=modifiers,
    )
    # Wrap the rule funcs in a module: the compiler parses this string with
    # `parseSourceString<ModuleOp>`, which requires a single top-level op.
    return "module {\n" + "\n".join(rule_strings) + "\n}"


def materialize_reachable_rule_strings(
    op_name: str,
    op_id: SerializedGraphOpID,
    dynamic_shape,
    wire_lens,
    static_data,
    extra_data=None,
    is_custom_op=False,
    modifiers=ModifierState(),
    op_cls=None,
):
    """Materialize reachable rules as MLIR strings for the compiler's on-demand callback.

    This is the string-producing compatibility boundary. Eager capture consumes DecompTargetSpecs
    directly.

    Args:
        op_name (str): the operator's name, as PennyLane registers its rules
        op_id (SerializedGraphOpID): the operator's graphOpId
        dynamic_shape (dict): dynamic argument names to their MLIR types
        wire_lens (dict): wire argument names to their lengths
        static_data (dict): compiler-static argument names to their values
        extra_data (dict): argument values the graphOpId identifies by UID instead of spelling
        is_custom_op (bool): whether the operator lowers to ``qref.custom``
        modifiers (ModifierState): the modifiers on the operator instance being decomposed
        op_cls (type): the operator's class; classes of the ops met along the way are taken from
            the resources themselves

    Returns:
        list[str]: the rules, as MLIR strings
    """
    request = OpDecompRequest(
        base_id=op_id,
        op_name=op_name,
        op_cls=op_cls,
        dynamic_shape=dynamic_shape,
        wire_lens=wire_lens,
        static_data=static_data,
        extra_data=extra_data or {},
        is_custom_op=is_custom_op,
        modifier_state=modifiers,
    )
    target_specs = walk_reachable_decomp_rule_sets([(request, {request.modifier_state})])
    return materialize_rule_strings(target_specs)
