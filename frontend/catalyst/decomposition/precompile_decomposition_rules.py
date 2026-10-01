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

from pathlib import Path

import pennylane as qp
from jax._src.lib.mlir import ir

from catalyst.compiler import _quantum_opt
from catalyst.decomposition.capture_session import DecompositionScope, OpDecompRequest
from catalyst.decomposition.decomposition_rules import (
    materialize_decomp_rule_strings,
    walk_reachable_decomp_rule_sets,
)
from catalyst.utils.runtime_environment import BYTECODE_FILE_PATH

PRECOMPILED_MODIFIERS = (
    (False, 0),
    (True, 0),
)


def precompile_decomp_rules(decomp_file_path: str = BYTECODE_FILE_PATH) -> None:
    """Compile PennyLane built-in decomposition rules to MLIR Bytecode.

    Args:
        decomp_file_path (Path): path to compile rules to.
    """
    Path(decomp_file_path).parent.mkdir(parents=True, exist_ok=True)

    # newline to ensure emptystring is never passed
    bytecode_lib = "\n"

    scope = DecompositionScope()

    for abstract_ops in qp.decomposition.signature_registry().values():
        for op in abstract_ops:
            for modifier_context in PRECOMPILED_MODIFIERS:
                scope.record_root(OpDecompRequest.from_operation(op, modifier_context))

    with ir.Context():
        target_specs = walk_reachable_decomp_rule_sets(list(scope.roots.values()))
        bytecode_lib += "\n".join(materialize_decomp_rule_strings(target_specs))

    bytecode = _quantum_opt(
        "--emit-bytecode",
        "--canonicalize",
        "--convert-to-value-semantics",
        "--canonicalize",
        "--register-decomp-rule-resource",
        stdin=bytecode_lib.encode("utf-8"),
        text=None,
    )

    with open(decomp_file_path, "wb") as bytecode_file:
        bytecode_file.write(bytecode)


if __name__ == "__main__":  # pragma: no cover
    precompile_decomp_rules()
