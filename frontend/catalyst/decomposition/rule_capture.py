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
"""Discover, capture, and bind decomposition-rule definitions for a supplied interpreter."""

from __future__ import annotations

import warnings
from copy import copy
from typing import TYPE_CHECKING

import jax
from jax.core import take_current_trace

import catalyst
from catalyst.decomposition.capture_session import DecompositionScope, RuleIdentity, TracedRule
from catalyst.decomposition.precompiled_rule_loading import load_precompiled_rule_identities
from catalyst.decomposition.rule_arguments import rule_call_operands
from catalyst.decomposition.rule_discovery import walk_reachable_decomp_rule_sets
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.decomposition.rule_preparation import DecompTargetSpec
from catalyst.from_plxpr.qref_jax_primitives import QrefQreg
from catalyst.jax_primitives import decomp_definition_p
from catalyst.utils.exceptions import CompileError

if TYPE_CHECKING:
    from catalyst.from_plxpr.qfunc_interpreter import PLxPRToQuantumJaxprInterpreter


def _convert_decomp_target_spec(
    interpreter: PLxPRToQuantumJaxprInterpreter,
    target_spec: DecompTargetSpec,
    precompiled_rule_identities: frozenset[RuleIdentity] = frozenset(),
    estimate_decomps: bool = False,
) -> list[TracedRule]:
    """Convert rules into Catalyst JAXPRs, retaining only their signatures if requested."""
    with take_current_trace():
        operands = rule_call_operands(target_spec.trace_inputs)
    out = []
    for rule in target_spec.rules:
        if (
            RuleIdentity(
                target_spec.target_id,
                rule.frontend_name,
                tuple(sorted(rule.resource_counts_by_id.items())),
            )
            in precompiled_rule_identities
        ):
            continue

        prepared_rule_fn = rule.prepared_rule_fn
        if not prepared_rule_fn.__name__.startswith("__builtin_"):
            prepared_rule_fn.__name__ = "__builtin_" + prepared_rule_fn.__name__

        with take_current_trace():
            if estimate_decomps:
                converted = jax.make_jaxpr(
                    lambda *args: catalyst.debug_assert(
                        False, "rule without implementation for estimation purposes"
                    )
                )(QrefQreg(), *operands)
            else:
                try:
                    plxpr = jax.make_jaxpr(prepared_rule_fn)(*operands)
                except Exception as exc:  # pylint: disable=broad-except
                    warnings.warn(
                        f"Failed to capture the {rule.frontend_name} decomposition rule for "
                        f"{target_spec.target_id}: {exc}",
                        category=RuleLoweringWarning,
                    )
                    continue
                if target_spec.reject_noninvertible and _contains_noninvertible_equation(plxpr):
                    continue

                def wrapper(global_qreg, *args):
                    converter = copy(interpreter)
                    converter.init_qreg = global_qreg
                    converter.decomposition_scope = None
                    converter.subroutine_cache = {}
                    converter(plxpr, *args)

                try:
                    converted = jax.make_jaxpr(wrapper)(QrefQreg(), *plxpr.in_avals)
                except Exception as exc:  # pylint: disable=broad-except
                    if isinstance(exc, CompileError):
                        raise
                    warnings.warn(
                        f"Failed to convert the {rule.frontend_name} decomposition rule for "
                        f"{target_spec.target_id}: {exc}",
                        category=RuleLoweringWarning,
                    )
                    continue
        out.append(
            TracedRule(
                prepared_rule_fn=prepared_rule_fn,
                closed_jaxpr=converted,
                target_gate=target_spec.target_id,
                resources=rule.resource_counts_by_id,
                frontend_name=rule.frontend_name,
            )
        )
    return out


def _capture_scope_rules(
    interpreter: PLxPRToQuantumJaxprInterpreter,
    scope: DecompositionScope,
    precompiled_rule_identities: frozenset[RuleIdentity] | None = None,
) -> None:
    """Populate one scope, loading cached identities unless the caller supplies them."""
    if scope.estimate_decomps:
        # Emit every declaration, including rules whose implementations are precompiled.
        precompiled_rule_identities = frozenset()
    elif precompiled_rule_identities is None:
        # default to empty-set - precompiled is an optimization, not required
        precompiled_rule_identities = frozenset()
        try:
            precompiled_rule_identities = load_precompiled_rule_identities()
        except Exception as e:
            warnings.warn(f"Failed to load precompiled rules: {e}")
    for target_spec in walk_reachable_decomp_rule_sets(list(scope.roots.values())):
        for traced_rule in _convert_decomp_target_spec(
            interpreter,
            target_spec,
            precompiled_rule_identities,
            estimate_decomps=scope.estimate_decomps,
        ):
            scope.record_definition(traced_rule)


def capture_and_bind_kernel_rules(
    interpreter: PLxPRToQuantumJaxprInterpreter,
    precompiled_rule_identities: frozenset[RuleIdentity] | None = None,
) -> None:
    """Capture reachable rules and bind definitions, optionally using supplied cache identities."""
    scope = interpreter.decomposition_scope
    if scope is None:
        return

    with take_current_trace():
        _capture_scope_rules(interpreter, scope, precompiled_rule_identities)

    for traced_rule in scope.definitions.values():
        decomp_definition_p.bind(
            pyfun=traced_rule.prepared_rule_fn,
            func_jaxpr=traced_rule.closed_jaxpr,
            target_gate=traced_rule.target_gate,
            resources=tuple(sorted(traced_rule.resources.items())),
            frontend_name=traced_rule.frontend_name,
            num_wires=None,
            estimate_decomps=scope.estimate_decomps,
        )


def _any_equation(jaxpr, predicate):
    """Return whether any equation in a PLxPR or its nested regions satisfies ``predicate``."""

    def is_region(value):
        return hasattr(value, "jaxpr") or hasattr(value, "eqns")

    for eqn in getattr(jaxpr, "jaxpr", jaxpr).eqns:
        if predicate(eqn):
            return True
        for value in eqn.params.values():
            nested = value if isinstance(value, (tuple, list)) else (value,)
            if any(is_region(item) and _any_equation(item, predicate) for item in nested):
                return True
    return False


def _contains_noninvertible_equation(jaxpr):
    """Return whether a PLxPR contains a measurement-like primitive."""

    return _any_equation(
        jaxpr,
        lambda eqn: "measure" in eqn.primitive.name or eqn.primitive.name.endswith("ppm"),
    )
