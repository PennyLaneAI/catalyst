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
"""Tests for standalone plugin.

The Standalone plugin may be found here:
https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone
"""

import pennylane as qp
import pytest

have_standalone_plugin = True

from standalone_plugin import SwitchBarToFoo, getStandalonePluginAbsolutePath


@pytest.mark.parametrize("capture", (True, False))
def test_pass_automatically_adds_to_required_plugins(capture):
    """Test that applying the pass from the plugin automatically adds
    to the list of required plugins."""

    @qp.qjit(capture=capture)
    @SwitchBarToFoo
    @qp.qnode(qp.device("null.qubit", wires=1))
    def c():
        return qp.expval(qp.Z(0))

    assert c.compile_options.pass_plugins == {getStandalonePluginAbsolutePath()}
    assert c.compile_options.dialect_plugins == {getStandalonePluginAbsolutePath()}

    assert 'transform.apply_registered_pass "standalone-switch-bar-foo"' in c.mlir


if __name__ == "__main__":
    pytest.main(["-x", __file__])
