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

"""Compatibility imports for decomposition-rule helpers and compiler callbacks.

Capture uses ``rule_arguments`` and ``rule_discovery`` directly. The compiler still
loads its on-demand callback from this module; its implementation lives in
``on_demand_rules``. Precompilation uses its own driver.
"""

from catalyst.decomposition.graph_op_id import GraphOpID
from catalyst.decomposition.on_demand_rules import (
    compile_decomposition_rules,
    compile_decomposition_rules_wrapper,
    compile_reachable_decomposition_rules_wrapper,
    compile_registered_symbolic_rules,
    materialize_reachable_rule_strings,
)
from catalyst.decomposition.rule_arguments import (
    HybridArgumentSpec,
    HybridLeafSource,
    RuleTraceInputs,
    build_base_op,
    flatten_hybrid_args,
    ordered_kwarg_names,
    prepare_dummy_rule_arguments,
    prepare_dynamic_op_kwargs,
    rule_call_operands,
    symbolic_arguments,
    unpack_rule_operands,
)
from catalyst.decomposition.rule_compilation import (
    build_rule_module,
    get_rule_strings_from_module,
    get_rules_from_module,
    materialize_decomp_target_spec,
)
from catalyst.decomposition.rule_discovery import walk_reachable_decomp_rule_sets
from catalyst.decomposition.rule_preparation import (
    CollectedRule,
    DecompRuleSpec,
    DecompTargetSpec,
    _adjoint_folds_to_base,
    _append_decomp_target_spec,
    _resources_have_measurement,
    _rule_allocates_work_wires,
    _rule_is_compatible,
    build_decomp_rule_variants,
    build_decomp_target_spec,
    build_registered_decomp_target_spec,
    collect_resources_for_op,
    collect_symbolic_resources,
    symbolic_op_name,
)

__all__ = (
    "CollectedRule",
    "DecompRuleSpec",
    "DecompTargetSpec",
    "GraphOpID",
    "HybridArgumentSpec",
    "HybridLeafSource",
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
