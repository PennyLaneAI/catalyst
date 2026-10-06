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

"""Utilities for AOT compiling PennyLane's decomposition rules to MLIR Bytecode."""

import json
from pathlib import Path

import pennylane as qp
from jax._src.lib.mlir import ir

from catalyst.compiler import _quantum_opt
from catalyst.decomposition.capture_session import DecompositionScope, OpDecompRequest
from catalyst.decomposition.decomposition_cache import (
    get_bytecode_hash,
    get_rule_entry,
)
from catalyst.from_plxpr.qfunc_interpreter import (
    PLxPRToQuantumJaxprInterpreter,
    capture_and_bind_kernel_rules,
)
from catalyst.from_plxpr.qref_jax_primitives import QrefQreg
from catalyst.utils.runtime_environment import BYTECODE_FILE_PATH, get_bytecode_manifest_path

PRECOMPILATION_MODIFIERS = (
    (False, 0),
    (True, 0),
)


def _capture_rule_module(scope: DecompositionScope) -> ir.Operation:
    """Capture every reachable rule in scope into one MLIR module."""
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
        capture_and_bind_kernel_rules(interpreter, precompiled_rule_identities=frozenset())

    return rules_module.mlir_module


def precompile_decomp_rules(_decomp_file_path: str | Path = BYTECODE_FILE_PATH) -> None:
    """Compile PennyLane built-in decomposition rules to MLIR Bytecode.

    Args:
        decomp_file_path (Path): path to compile rules to.
    """
    Path(_decomp_file_path).parent.mkdir(parents=True, exist_ok=True)
    _decomp_file_path = Path(_decomp_file_path)

    scope = DecompositionScope()

    for abstract_ops in qp.decomposition.signature_registry().values():
        for op in abstract_ops:
            for modifier_state in PRECOMPILATION_MODIFIERS:
                scope.record_root(OpDecompRequest.from_operation(op, modifier_state))

    rule_module = _capture_rule_module(scope)

    bytecode = _quantum_opt(
        "--emit-bytecode",
        "--canonicalize",  # TODO: do we need these passes anymore?
        "--convert-to-value-semantics",
        "--canonicalize",
        "--register-decomp-rule-resource",
        stdin=str(rule_module).encode("utf-8"),
        text=None,
        stderr_return=True,
    )

    with open(_decomp_file_path, "wb") as bytecode_file:
        bytecode_file.write(bytecode)

    with open(
        get_bytecode_manifest_path(_decomp_file_path), "w", encoding="utf-8"
    ) as manifest_file:
        manifest = {
            "bytecode_hash": get_bytecode_hash(bytecode),
            "precompiled_rules": [get_rule_entry(rule) for rule in scope.definitions],
        }

        json.dump(manifest, manifest_file)


if __name__ == "__main__":  # pragma: no cover
    precompile_decomp_rules()
