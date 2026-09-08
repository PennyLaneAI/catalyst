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

"""Helpers for the ``catalyst.runtime_artifacts`` module attribute.

The attribute records shared-library paths that a compiled program needs at link time. The write
side is used by JAX primitive lowering, when emitting a ``catalyst.custom_call`` that targets a
local (in-process) external symbol exported by library. The read side is used by the compiler driver
to add those libraries to the link command.
"""

from jax._src.lib.mlir import ir

RUNTIME_ARTIFACTS_ATTR = "catalyst.runtime_artifacts"


def record_runtime_artifact(module_op, artifact_path):
    """Append ``artifact_path`` to the module's ``catalyst.runtime_artifacts`` attribute, once."""
    attrs = module_op.attributes
    existing = (
        [ir.StringAttr(a).value for a in attrs[RUNTIME_ARTIFACTS_ATTR]]
        if RUNTIME_ARTIFACTS_ATTR in attrs
        else []
    )
    if artifact_path in existing:
        return
    existing.append(artifact_path)
    attrs[RUNTIME_ARTIFACTS_ATTR] = ir.ArrayAttr.get([ir.StringAttr.get(p) for p in existing])


def collect_runtime_artifacts(mlir_module, compile_options):
    """Aggregate every ``catalyst.runtime_artifacts`` path into ``compile_options``.

    Walks the module and all nested modules, collecting the artifact paths recorded on each, so the
    linker receives the full set. The result is stored on ``compile_options.runtime_artifacts``.
    """
    seen = []

    def _walk(op):
        attrs = op.attributes
        if RUNTIME_ARTIFACTS_ATTR in attrs:
            for string_attr in attrs[RUNTIME_ARTIFACTS_ATTR]:
                path = ir.StringAttr(string_attr).value
                if path not in seen:
                    seen.append(path)
        for region in op.regions:
            for block in region:
                for child_op in block:
                    _walk(child_op)

    _walk(mlir_module.operation)
    compile_options.runtime_artifacts = tuple(seen)
