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


"""Compile prepared decomposition-rule batches into standalone MLIR.

This module consumes ``DecompTargetSpec`` batches from rule preparation and owns
qjit tracing, MLIR attributes, inlining, and rule-function extraction. The compiler
entry points coordinate discovery and preparation before calling these helpers.
"""

import warnings

import pennylane as qp
from jax._src.lib.mlir import ir

from catalyst.compiler import _quantum_opt
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.decomposition.rule_preparation import (
    DecompTargetSpec,
    RuleTraceInputs,
    rule_call_operands,
)
from catalyst.decomposition.type_utils import SerializedGraphOpID
from catalyst.jax_extras.lowering import get_mlir_attribute_from_pyval
from catalyst.utils.exceptions import CompileError

# Ops that make a decomposition body non-invertible
_NON_INVERTIBLE_MARKERS = (
    "qref.measure",
    "quantum.measure",
    "measure_in_basis",
    ".ppm",  # pbc.ppm / pbc.ref.ppm / pbc.select.ppm
)


def get_rule_strings_from_module(module: ir.Module) -> list[str]:
    """Extract the decomposition rules held by a module as MLIR strings.

    Every FuncOp carrying a ``target_gate`` attribute is a decomposition rule.

    Args:
        module (ir.Module): the module a rule-compiling qjit produced

    Returns:
        list[str]: one string per rule, with the ``__builtin_`` prefix added to its name
    """
    raw_funcOps = []

    def find_condition(op):
        if op.name == "func.func":
            if "target_gate" in op.attributes:
                raw_funcOps.append(op)
                return ir.WalkResult.SKIP
        return ir.WalkResult.ADVANCE

    module.operation.walk(find_condition)

    # If we simply rename the rule func op in the original module (from the qjit that compiles the
    # rule), the call op to the rule subroutine from the main qjit function will complain that its
    # callee doesn't exist.
    # We have to do a clone, and rename the clone.
    # And to clone safely, we must set the insertion point to a separate sandbox module
    funcOps = []
    ctx = module.context
    with ctx, ir.Location.unknown(ctx):
        sandbox_module = ir.Module.create()
        with ir.InsertionPoint(sandbox_module.body):
            for op in raw_funcOps:
                clone = op.clone()

                old_attr = clone.attributes["sym_name"]
                clean_name = old_attr.value.strip('"')

                if not clean_name.startswith("__builtin_"):
                    clone.attributes["sym_name"] = ir.StringAttr.get(
                        "__builtin_" + clean_name, context=ctx
                    )

                funcOps.append(str(clone))

    return funcOps


def get_rules_from_module(module: ir.Module) -> str:
    """
    Parse and modify decomposition rules from a ModuleOp.

    Args:
        module (ir.Module): an MLIR module object; every FuncOp carrying a `target_gate`
            attribute is extracted as a decomposition rule.

    Returns:
        str: The string representation of any decomposition rules from `module`, pre-pending the
             `__builtin_` prefix to their names.
    """
    funcOps = get_rule_strings_from_module(module)
    return "\n".join(str(funcOp) for funcOp in funcOps) if funcOps else ""


def materialize_decomp_target_spec(target_spec: DecompTargetSpec) -> ir.Operation:
    """Materialize one target specification through the legacy standalone qjit path."""

    return build_rule_module(
        [qp.capture.subroutine(rule.prepared_rule_fn) for rule in target_spec.rules],
        qp.device("null.qubit", wires=target_spec.num_qreg_qubits),
        target_spec.trace_inputs,
        {rule.prepared_rule_fn.__name__: rule for rule in target_spec.rules},
        target_spec.target_id,
    )


def build_rule_module(
    subroutines,
    device,
    trace_inputs: RuleTraceInputs,
    name_to_rule_specs,
    target_id: SerializedGraphOpID,
) -> ir.Operation:
    """Trace ``subroutines`` into a module of standalone decomposition-rule functions.

    Args:
        subroutines (list): the rule bodies, as captured subroutines
        device (Device): the device to trace them on, sized for the operator's wires
        trace_inputs (RuleTraceInputs): shared dummy arguments and operand layout for the subroutines
        name_to_rule_specs (dict): generated function name to its rule specification
        target_id (SerializedGraphOpID): the graphOpId of the gate the rules decompose

    Returns:
        ir.Operation: the module holding the rules

    Raises:
        CompileError: if the rules could not be traced
    """

    operands = rule_call_operands(trace_inputs)

    @qp.qjit(target="mlir", capture=True, collect_decomp_rules=False)
    @qp.qnode(device=device)
    def circuit():
        for subroutine in subroutines:
            subroutine(*operands)

    module = circuit.mlir_module
    if module is None:
        raise CompileError(
            f"Failed to generate an MLIR module while compiling decomposition rules for {target_id}"
        )

    def update_funcop_attributes(op):
        """Update the decomposition rule attributes if op is a decomposition rule.

        For use with module.walk

        This function updates the following attributes:
            - Adds the `target_gate` attribute.
            - Adds the `resources` attribute.
            - Sets the visibility to public (so the inliner does not remove them)
        """
        if op.name == "func.func":
            func_name = ir.StringAttr(op.attributes["sym_name"]).value
            if rule := name_to_rule_specs.get(func_name):
                op.attributes["resources"] = get_mlir_attribute_from_pyval(
                    {"operations": rule.resource_counts_by_id}
                )
                op.attributes["target_gate"] = ir.StringAttr.get(target_id)
                op.attributes["sym_visibility"] = ir.StringAttr.get("public")
                op.attributes["frontend_name"] = ir.StringAttr.get(rule.frontend_name)

        return ir.WalkResult.ADVANCE

    with module.context, ir.Location.unknown():
        module.operation.walk(update_funcop_attributes)

    # Inline to avoid helper functions. We want all decomp rule functions to be standalone
    # Generic printing needed when parsing --quantum-opt string output back to jax IR ModuleOps
    # since Catalyst's python bindings never export the Catalyst dialects
    # Before inlining we need to remove the qnode function, since that has a call to the compiled
    # rule subroutine
    qnode_func_erasure_worklist = []

    def remove_qnode_func(op):
        if op.name == "catalyst.launch_kernel":
            qnode_func_erasure_worklist.append(op.parent)
            return ir.WalkResult.ADVANCE
        if op.name == "func.func" and "quantum.node" in op.attributes:
            qnode_func_erasure_worklist.append(op)
            return ir.WalkResult.ADVANCE
        return ir.WalkResult.ADVANCE

    with module.context, ir.Location.unknown():
        module.operation.walk(remove_qnode_func)

    for qnode_func in qnode_func_erasure_worklist:
        qnode_func.erase()

    inlined = _quantum_opt(
        "--inline=inlining-threshold=4294967295",  # Use uint max to indicate always inline
        "--mlir-print-op-generic",
        stdin=str(module),
    )

    inlined_module = ir.Operation.parse(inlined, context=module.context)

    def re_privatize_rules(op):
        if op.name == "func.func" and "target_gate" in op.attributes:
            op.attributes["sym_visibility"] = ir.StringAttr.get("private")
        return ir.WalkResult.ADVANCE

    with inlined_module.context, ir.Location.unknown():
        inlined_module.operation.walk(re_privatize_rules)

    return inlined_module


def materialize_rule_strings(target_specs: list[DecompTargetSpec]) -> list[str]:
    """Compile prepared batches and extract their standalone rule functions.

    Empty batches are skipped, compilation failures are reported per target, and
    noninvertible bodies are excluded when the target requires invertible rules.
    """
    rules = []
    for target_spec in target_specs:
        if not target_spec.rules:
            continue
        try:
            rule_strings = get_rule_strings_from_module(materialize_decomp_target_spec(target_spec))
            if target_spec.reject_noninvertible:
                rule_strings = [
                    rule
                    for rule in rule_strings
                    if not any(marker in rule for marker in _NON_INVERTIBLE_MARKERS)
                ]
            rules.extend(rule_strings)
        except Exception as exc:  # pylint: disable=broad-except
            warnings.warn(
                f"Failed to materialize decomposition rules for {target_spec.target_id}: {exc}",
                category=RuleLoweringWarning,
            )
    return rules
