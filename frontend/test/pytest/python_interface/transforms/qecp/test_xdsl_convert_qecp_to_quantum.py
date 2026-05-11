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

"""Tests for the convert-qecp-to-quantum xDSL dialect-conversion pass."""

import pytest

from catalyst.python_interface.transforms.qecp.convert_qecp_to_quantum import (
    ConvertQecPhysicalToQuantumPass,
    convert_qecp_to_quantum_pass,
)

pytestmark = pytest.mark.xdsl


class TestConvertQecPhysicalToQuantumPass:
    """Unit tests for ConvertQecPhysicalToQuantumPass."""

    def test_pass_name(self):
        """The module pass reports a stable pipeline name."""
        assert ConvertQecPhysicalToQuantumPass.name == "convert-qecp-to-quantum"

    def test_compiler_transform_wrapper(self):
        """The compiler_transform wrapper is defined and callable."""
        assert callable(convert_qecp_to_quantum_pass)
