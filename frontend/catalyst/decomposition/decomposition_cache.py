# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for loading decomposition rules from the precompilation cache."""

import hashlib
import json
from pathlib import Path

from catalyst.decomposition.capture_session import RuleIdentity
from catalyst.utils.runtime_environment import BYTECODE_FILE_PATH, get_bytecode_manifest_path


def get_bytecode_hash(bytecode: bytes) -> str:
    """Return the sha256 hash of the bytecode content."""
    return hashlib.sha256(bytecode).hexdigest()


def get_rule_entry(identity: RuleIdentity) -> dict:
    """Serialize a rule identity for the precompilation manifest."""
    return {
        "target_gate": identity.target_gate,
        "frontend_name": identity.frontend_name,
        "resources": dict(identity.resources),
    }


def _validate_rule_entry(entry) -> RuleIdentity:
    """Validate and canonicalize one rule manifest entry."""
    if not isinstance(entry, dict) or set(entry) != {
        "target_gate",
        "frontend_name",
        "resources",
    }:
        raise ValueError("Malformed precompiled rule entry.")

    target_gate = entry["target_gate"]
    frontend_name = entry["frontend_name"]
    resources = entry["resources"]

    if not isinstance(target_gate, str) or not isinstance(frontend_name, str):
        raise ValueError("Malformed precompiled rule.")

    if not isinstance(resources, dict) or any(
        not isinstance(gate, str) or not isinstance(count, int) for gate, count in resources.items()
    ):
        raise ValueError("Malformed precompiled rule resources.")

    return RuleIdentity(target_gate, frontend_name, resources)


def _validate_manifest(manifest, bytecode: bytes) -> frozenset[RuleIdentity]:
    """Validate a manifest and return its canonical rule identities."""
    if not isinstance(manifest, dict) or set(manifest) != {
        "bytecode_hash",
        "precompiled_rules",
    }:
        raise ValueError("Malformed decomposition cache manifest.")

    if not isinstance(manifest["bytecode_hash"], str) or manifest[
        "bytecode_hash"
    ] != get_bytecode_hash(bytecode):
        raise ValueError(
            "Decomposition cache hash mismatch. The bytecode version is likely out of date."
        )

    entries = manifest["precompiled_rules"]
    if not isinstance(entries, list):
        raise ValueError("Malformed precompiled rule list")
    return frozenset(_validate_rule_entry(entry) for entry in entries)


def load_precompiled_rule_identities(
    _bytecode_path: str | Path | None = None,
) -> frozenset[RuleIdentity]:
    """Validated and load precompiled rules from the precompiled cache."""
    _bytecode_path = Path(BYTECODE_FILE_PATH if _bytecode_path is None else _bytecode_path)

    bytecode = _bytecode_path.read_bytes()
    with get_bytecode_manifest_path(_bytecode_path).open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    return _validate_manifest(manifest, bytecode)
