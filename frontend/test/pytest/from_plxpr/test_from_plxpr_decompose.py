# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests the ``decompose`` transform with the new Catalyst graph-based decomposition system."""
from functools import partial

import numpy as np
import pennylane as qml
import pytest

pytestmark = pytest.mark.usefixtures("disable_capture")


class TestCatalystDecompose:
    """Tests the decompose transform with graph enabled."""

    def test_simple_decompose(self):
        """Test catalyst decompose that can be toggled with qml.transforms.decompose"""

        @partial(qml.transforms.decompose, gateset={"RX", "RZ"})
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit():
            qml.RX(0.5, wires=0)
            return qml.expval(qml.PauliZ(0))
