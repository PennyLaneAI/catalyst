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
"""Catalyst additions to the Jax library"""

from catalyst.jax_extras.lowering import custom_lower_jaxpr_to_module, jaxpr_to_mlir
from catalyst.jax_extras.patches import (
    _gather_shape_rule_dynamic,
    _no_clean_up_dead_vars,
    get_aval2,
)
from catalyst.jax_extras.tracing import (
    ClosedJaxpr,
    ConcreteArray,
    DBIdx,
    DShapedArray,
    DynamicJaxprTrace,
    DynamicJaxprTracer,
    DynshapePrimitive,
    ExpansionStrategy,
    InDBIdx,
    InputSignature,
    Jaxpr,
    JaxprEqn,
    OutDBIdx,
    OutputSignature,
    PyTreeDef,
    PyTreeRegistry,
    ShapedArray,
    ShapeDtypeStruct,
    _abstractify,
    _extract_implicit_args,
    _initial_style_jaxpr,
    _input_type_to_tracers,
    collapse,
    cond_expansion_strategy,
    convert_constvars_jaxpr,
    convert_element_type,
    deduce_avals,
    deduce_signatures,
    eval_jaxpr,
    expand_args,
    expand_results,
    find_top_trace,
    for_loop_expansion_strategy,
    get_implicit_and_explicit_flat_args,
    infer_lambda_input_type,
    infer_output_type_jaxpr,
    infer_output_type_python,
    input_type_to_tracers,
    jaxpr_pad_consts,
    make_jaxpr2,
    make_jaxpr_effects,
    new_dynamic_main2,
    new_inner_tracer,
    output_type_to_tracers,
    sort_eqns,
    trace_to_jaxpr,
    transient_jax_config,
    tree_flatten,
    tree_structure,
    tree_unflatten,
    treedef_is_leaf,
    unzip2,
    while_loop_expansion_strategy,
    wrap_init,
)
