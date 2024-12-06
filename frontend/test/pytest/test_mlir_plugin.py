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
"""Tests for standalone plugin"""

from pathlib import Path

import pennylane as qml

from catalyst.passes import apply_pass, apply_pass_plugin
from catalyst.utils.runtime_environment import get_bin_path

plugin_path = get_bin_path("cli", "CATALYST_BIN_DIR") + "/../StandalonePlugin.so"
plugin = Path(plugin_path)


def test_standalone_plugin():
    """Generate MLIR for the standalone plugin. Do not execute code.
    The code execution test is in the lit test. See that test
    for more information as to why that is the case."""

    @apply_pass("standalone-switch-bar-foo")
    @qml.qnode(qml.device("lightning.qubit", wires=0))
    def qnode():
        return qml.state()

    @qml.qjit(pass_plugins={plugin}, dialect_plugins={plugin}, target="mlir")
    def module():
        return qnode()

    # It would be nice if we were able to convine lit tests with
    # pytest
    assert "standalone-switch-bar-foo" in module.mlir


def test_standalone_plugin_no_preregistration():
    """Generate MLIR for the standalone plugin, no need to register the
    plugin ahead of time in the qjit decorator"""

    @apply_pass_plugin(plugin, "standalone-switch-bar-foo")
    @qml.qnode(qml.device("lightning.qubit", wires=0))
    def qnode():
        return qml.state()

    @qml.qjit(target="mlir")
    def module():
        return qnode()

    # It would be nice if we were able to convine lit tests with
    # pytest
    assert "standalone-switch-bar-foo" in module.mlir
