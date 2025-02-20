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

import pennylane as qml
import pytest

from catalyst import qjit
from catalyst.third_party.oqd import OQDDevice


class TestOpenAPL:
    """Test that the OQD device correctly generates an OpenAPL program."""

    oqd_pipelines = [
        (
            "device-agnoistic-pipeline",
            [
                "enforce-runtime-invariants-pipeline",
                "hlo-lowering-pipeline",
                "quantum-compilation-pipeline",
                "bufferization-pipeline",
            ],
        ),
        (
            "oqd_pipeline",
            [
                "func.func(ions-decomposition)",
                "func.func(quantum-to-ion{"
                + "device-toml-loc="
                + "frontend/test/test_oqd/oqd/calibration_data/device.toml "
                + "qubit-toml-loc="
                + "frontend/test/test_oqd/oqd/calibration_data/qubit.toml "
                + "gate-to-pulse-toml-loc="
                + "frontend/test/test_oqd/oqd/calibration_data/gate.toml"
                + "})",
                "convert-ion-to-llvm",
            ],
        ),
        (
            "llvm-dialect-lowering-pipeline",
            [
                "llvm-dialect-lowering-pipeline",
            ],
        ),
    ]

    output_f = "__openapl__output.json"

    def test_RX_gate(self):
        """Test OpenAPL generation for a circuit with a single RX Gate."""
        oqd_dev = OQDDevice(backend="default", shots=4, wires=1)

        @qjit(pipelines=self.oqd_pipelines)
        @qml.qnode(oqd_dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.counts()

        circuit(1.5708)

        expected_f = "frontend/test/test_oqd/oqd/test_single_RX.json"
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

        expected_f = "frontend/test/test_oqd/oqd/test_single_CNOT.json"
        with open(expected_f, "r", encoding="utf-8") as f:
            expected_json = json.load(f)

        assert sorted(catalyst_json.items()) == sorted(expected_json.items())
        os.remove("__openapl__output.json")


if __name__ == "__main__":
    pytest.main(["-x", __file__])
