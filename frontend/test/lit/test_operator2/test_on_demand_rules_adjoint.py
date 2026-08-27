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

"""
Test the compiler's on-demand decomposition-rule entry point for adjoint rules.
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable = line-too-long,unused-argument,missing-function-docstring

import pennylane as qp
from operator2_dummy_gates import NoParams, SingleParam
from pennylane.typing import Float, Wire

from catalyst.decomposition.decomposition_rules import (
    compile_reachable_decomposition_rules_wrapper,
)


def _base_rule():
    def base_resource_fn(reg):
        return {SingleParam(x=Float, reg=Wire[2]): 1}

    @qp.register_resources(base_resource_fn)
    def base_rule(reg):
        SingleParam(x=0.1, reg=reg[0:2])

    return base_rule


def _adj_rule():
    def adj_resource_fn(reg):
        return {SingleParam(x=Float, reg=Wire[2]): 2}

    @qp.register_resources(adj_resource_fn)
    def adj_rule(reg):
        SingleParam(x=0.2, reg=reg[0:2])
        SingleParam(x=0.3, reg=reg[0:2])

    return adj_rule


def test_on_demand_adjoint_id_routes_to_adjoint_rules():
    """Requesting the rules for an ``Adjoint(NoParams)`` id yields the reachable closure with the
    adjoint rules correctly routed: the base rule of NoParams, the rule registered on
    Adjoint(NoParams), and the synthesized (adjointed base) rule.
    Crucially, the adjoint target is never attached to the un-adjointed base body."""
    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, _base_rule())
        qp.add_decomps("Adjoint(NoParams)", _adj_rule())

        out = compile_reachable_decomposition_rules_wrapper(
            "NoParams", "Adjoint(NoParams){}{reg:2}{}", {}, {"reg": 2}, {}, is_custom_op=False
        )

        print(out)

    # CHECK: module {
    # CHECK-DAG: func.func private @"__builtin_base_rule_NoParams{}{reg:2}{}"{{.*}}"SingleParam{{.*}}target_gate = "NoParams{}{reg:2}{}"
    # CHECK-DAG: func.func private @"__builtin_adj_rule_Adjoint(NoParams){}{reg:2}{}"{{.*}}"SingleParam{{.*}} = 2 : i64{{.*}}target_gate = "Adjoint(NoParams){}{reg:2}{}"
    # CHECK-DAG: func.func private @"__builtin_base_rule_Adjoint(NoParams){}{reg:2}{}"{{.*}}"Adjoint(SingleParam{{.*}}target_gate = "Adjoint(NoParams){}{reg:2}{}"
    # CHECK: qref.adjoint


test_on_demand_adjoint_id_routes_to_adjoint_rules()
