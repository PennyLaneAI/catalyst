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

"""Tests for the Open Quantum Design (OQD) trapped-ion quantum computer device.
"""

import pennylane as qml
import pytest


class TestOQDDevice:
    """Test the OQD device python layer for Catalyst."""

    def test_device_initialization(self):
        """Test that the OQD device is correctly initialized.

        The test checks that the backend, ion, shots, and wires parameters are correctly set.
        """
        device = qml.device("oqd.default", backend="default", shots=1024, wires=8)

        assert device.backend == "default"
        assert device.shots == qml.measurements.Shots(1024)
        assert device.wires == qml.wires.Wires(range(0, 8))


if __name__ == "__main__":
    pytest.main(["-x", __file__])
