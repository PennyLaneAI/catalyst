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

The attribute records shared-library paths that compiled programs need at link time. The write
side is used by JAX primitive lowering (when emitting a ``catalyst.custom_call`` that targets an
external symbol); the read side is used by the compiler driver to populate the link command.
"""

import re

from jax._src.lib.mlir import ir

RUNTIME_ARTIFACTS_ATTR = "catalyst.runtime_artifacts"


def record_runtime_artifact(module_op, artifact_path):
    """Append `artifact_path` to the module's `catalyst.runtime_artifacts` attr."""
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
    """Walk all nested modules and collect artifact paths into compile_options.

    Looks for catalyst.runtime_artifacts ArrayAttr on all nested modules and aggregates them so
    that the linker receives the full set of artifacts.
    """
    seen = set()

    def _walk(op):
        attrs = op.attributes
        if RUNTIME_ARTIFACTS_ATTR in attrs:
            for string_attr in attrs[RUNTIME_ARTIFACTS_ATTR]:
                seen.add(ir.StringAttr(string_attr).value)
        for region in op.regions:
            for block in region:
                for child_op in block:
                    _walk(child_op)

    _walk(mlir_module.operation)
    compile_options.runtime_artifacts = tuple(seen)


_RUNTIME_ARTIFACTS_TEXT_RE = re.compile(
    rf"{re.escape(RUNTIME_ARTIFACTS_ATTR)}\s*=\s*\[([^\]]*)\]"
)
_RUNTIME_ARTIFACTS_PATH_RE = re.compile(r'"([^"]*)"')


def collect_runtime_artifacts_from_text(ir_text, compile_options):
    """Text-mode sibling of `collect_runtime_artifacts`.

    Used by entry points that only have raw MLIR text (e.g. `compile_mlir`), where parsing into
    an `ir.Module` would require registering every Catalyst/StableHLO/etc. dialect on a fresh
    context. Scans the text for all `catalyst.runtime_artifacts = [...]` occurrences (handling
    nested modules) and populates `compile_options.runtime_artifacts`.
    """
    seen = set()
    for match in _RUNTIME_ARTIFACTS_TEXT_RE.finditer(ir_text):
        seen.update(_RUNTIME_ARTIFACTS_PATH_RE.findall(match.group(1)))
    compile_options.runtime_artifacts = tuple(seen)
