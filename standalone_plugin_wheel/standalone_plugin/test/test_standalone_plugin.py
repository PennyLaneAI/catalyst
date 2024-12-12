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
"""Tests for standalone plugin."""

import pennylane as qml
import pytest
from standalone_plugin import SwitchBarToFoo


def test_standalone_plugin():
    """Generate MLIR for the standalone plugin"""

    @SwitchBarToFoo
    @qml.qnode(qml.device("lightning.qubit", wires=0))
    def qnode():
        return qml.state()

    @qml.qjit(target="mlir", keep_intermediate=True)
    def module():
        return qnode()

    assert "standalone-switch-bar-foo" in module.mlir


if __name__ == "__main__":
    pytest.main(["-x", __file__])
