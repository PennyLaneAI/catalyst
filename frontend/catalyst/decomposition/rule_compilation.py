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

"""Compile reachable decomposition rules through the shared scope tracer."""

import pennylane as qp
from jax._src.lib.mlir import ir

from catalyst.decomposition.capture_session import DecompositionScope, RuleIdentity
from catalyst.decomposition.rule_capture import capture_and_bind_kernel_rules
from catalyst.from_plxpr.qfunc_interpreter import PLxPRToQuantumJaxprInterpreter
from catalyst.from_plxpr.qref_jax_primitives import QrefQreg
from catalyst.utils.exceptions import CompileError


def _capture_rule_module(
    scope: DecompositionScope,
    *,
    precompiled_rule_identities: frozenset[RuleIdentity] = frozenset(),
) -> ir.Module:
    """Capture reachable rules into one module, skipping supplied available identities."""
    interpreter = PLxPRToQuantumJaxprInterpreter(
        qp.device("null.qubit", wires=1),
        None,
        QrefQreg(),
        {},
        decomposition_scope=scope,
    )

    # use target="mlir" and .mlir_module to skip compiler invocation
    @qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
    def rules_module():
        capture_and_bind_kernel_rules(
            interpreter, precompiled_rule_identities=precompiled_rule_identities
        )

    module = rules_module.mlir_module
    if module is None:
        raise CompileError("Failed to generate the precompiled decomposition-rule module")
    return module
