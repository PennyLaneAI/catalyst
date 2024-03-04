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
"""Test for the OQC device.
"""
import pytest

import pennylane as qml
from catalyst.oqc import OQCDevice

class TestOQC:

    def test_authenticate(self):
        """Test the authentification"""
        credentials = {'url': 'abc', 'email': '@', 'password': '123' }
        device = OQCDevice(backend="lucy", shots=1000, wires=1, credentials=credentials)

        @qml.qnode(device=device)
        def circuit():
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliZ(wires=0))


if __name__ == "__main__":
    pytest.main(["-x", __file__])