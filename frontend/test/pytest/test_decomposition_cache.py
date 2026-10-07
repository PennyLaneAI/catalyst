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

"""Tests for precompiled decomposition-rule cache metadata."""

import json

import pytest

from catalyst.decomposition import decomposition_cache
from catalyst.decomposition.capture_session import RuleIdentity
from catalyst.decomposition.decomposition_cache import (
    get_bytecode_hash,
    get_rule_entry,
    load_precompiled_rule_identities,
)
from catalyst.utils.runtime_environment import get_bytecode_manifest_path

BYTECODE = b"bytecode"
RULE_ENTRY = {
    "target_gate": "Target",
    "frontend_name": "rule",
    "resources": {"Z": 2, "A": 1},
}


def _get_test_manifest(entries=(RULE_ENTRY,)):
    """Return a valid cache manifest."""
    return {
        "bytecode_hash": get_bytecode_hash(BYTECODE),
        "precompiled_rules": list(entries),
    }


def _write_test_cache(bytecode_path, manifest=None):
    """Write a temporary bytecode file and its manifest for testing."""
    bytecode_path.write_bytes(BYTECODE)
    get_bytecode_manifest_path(bytecode_path).write_text(
        json.dumps(_get_test_manifest() if manifest is None else manifest),
        encoding="utf-8",
    )


def test_load_valid_default_cache(monkeypatch, tmp_path):
    """The default cache path loads canonical, immutable rule identities."""
    bytecode_path = tmp_path / "rules.mlirbc"
    _write_test_cache(bytecode_path)

    # mock the default file location to the tmp file
    monkeypatch.setattr(decomposition_cache, "BYTECODE_FILE_PATH", str(bytecode_path))

    # load from the mocked cache
    identities = load_precompiled_rule_identities()

    assert identities == frozenset({RuleIdentity("Target", "rule", {"A": 1, "Z": 2})})


def test_manifest_entry_canonicalizes_resources():
    """Manifest entries use the same canonical resource ordering as rule identities."""
    identity = RuleIdentity("Target", "rule", {"Z": 2, "A": 1})
    entry = get_rule_entry(identity)

    assert identity.resources == (("A", 1), ("Z", 2))
    assert entry == RULE_ENTRY
    assert list(entry["resources"]) == ["A", "Z"]


@pytest.mark.parametrize("missing_file", ["bytecode", "manifest"])
def test_missing_cache_file_raises(tmp_path, missing_file):
    """A missing cache or missing manifest raises an error on loading."""
    bytecode_path = tmp_path / "rules.mlirbc"
    if missing_file == "manifest":
        bytecode_path.write_bytes(BYTECODE)
    else:
        get_bytecode_manifest_path(bytecode_path).write_text(
            json.dumps(_get_test_manifest()), encoding="utf-8"
        )

    with pytest.raises(FileNotFoundError):
        load_precompiled_rule_identities(bytecode_path)


def test_malformed_manifest(tmp_path):
    """Malformed JSON raises a parsing error."""
    bytecode_path = tmp_path / "rules.mlirbc"
    bytecode_path.write_bytes(BYTECODE)
    get_bytecode_manifest_path(bytecode_path).write_text("{", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_precompiled_rule_identities(bytecode_path)


@pytest.mark.parametrize(
    "manifest",
    [
        {},
        {"bytecode_hash": 1, "precompiled_rules": []},
        {"bytecode_hash": "wrong", "precompiled_rules": []},
        {"bytecode_hash": get_bytecode_hash(BYTECODE), "precompiled_rules": {}},
        {
            "bytecode_hash": get_bytecode_hash(BYTECODE),
            "precompiled_rules": [],
            "unexpected": None,
        },
    ],
)
def test_invalid_manifest_raises(tmp_path, manifest):
    """Malformed manifest fields and hash mismatches raise validation errors."""
    bytecode_path = tmp_path / "rules.mlirbc"
    _write_test_cache(bytecode_path, manifest)

    with pytest.raises(ValueError):
        load_precompiled_rule_identities(bytecode_path)


@pytest.mark.parametrize(
    "entry",
    [
        None,
        {"target_gate": "Target", "frontend_name": "rule"},
        {"target_gate": 1, "frontend_name": "rule", "resources": {}},
        {"target_gate": "Target", "frontend_name": 1, "resources": {}},
        {"target_gate": "Target", "frontend_name": "rule", "resources": []},
        {"target_gate": "Target", "frontend_name": "rule", "resources": {"A": "1"}},
        {"target_gate": "Target", "frontend_name": "rule", "resources": {"A": True}},
    ],
)
def test_malformed_rule_entry_raises(tmp_path, entry):
    """A malformed rule entry raises a validation error."""
    bytecode_path = tmp_path / "rules.mlirbc"
    _write_test_cache(bytecode_path, _get_test_manifest([entry]))

    with pytest.raises(ValueError):
        load_precompiled_rule_identities(bytecode_path)
