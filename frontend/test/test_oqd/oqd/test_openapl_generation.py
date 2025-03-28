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

"""Test the OpenAPL generation."""

import json
import os

import numpy as np
import pennylane as qml
import pytest

from catalyst import qjit
from catalyst.third_party.oqd import OQDDevice, OQDDevicePipeline


class TestOpenAPL:
    """Test that the OQD device correctly generates an OpenAPL program."""

    test_path = os.path.dirname(__file__)
    toml_path = os.path.join(test_path, "calibration_data/")
    oqd_pipelines = OQDDevicePipeline(
        toml_path + "device.toml", toml_path + "qubit.toml", toml_path + "gate.toml"
    )

    output_f = "__openapl__output.json"

    def test_RX_gate(self):
        """Test OpenAPL generation for a circuit with a single RX Gate."""
        oqd_dev = OQDDevice(backend="default", shots=4, wires=1)

        @qjit(pipelines=self.oqd_pipelines)
        @qml.qnode(oqd_dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.counts()

        circuit(np.pi / 2)

        expected_f = os.path.join(self.test_path, "test_single_RX.json")
        with open(self.output_f, "r", encoding="utf-8") as f:
            catalyst_json = json.load(f)

        with open(
            expected_f,
            "r",
            encoding="utf-8",
        ) as f:
            expected_json = json.load(f)

        assert sorted(catalyst_json.items()) == sorted(expected_json.items())
        os.remove(self.output_f)

    def test_CNOT_gate(self):
        """Test OpenAPL generation for a circuit with a single CNOT circuit."""
        oqd_dev = OQDDevice(backend="default", shots=4, wires=2)

        @qjit(pipelines=self.oqd_pipelines)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.counts()

        circuit()

        with open(self.output_f, "r", encoding="utf-8") as f:
            catalyst_json = json.load(f)

        expected_f = os.path.join(self.test_path, "test_single_CNOT.json")
        with open(expected_f, "r", encoding="utf-8") as f:
            expected_json = json.load(f)

        assert sorted(catalyst_json.items()) == sorted(expected_json.items())
        os.remove("__openapl__output.json")


if __name__ == "__main__":
    pytest.main(["-x", __file__])
