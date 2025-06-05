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

MODULE_TEST_PATH = os.path.dirname(__file__)
OQD_PIPELINES = OQDDevicePipeline(
    os.path.join(MODULE_TEST_PATH, "calibration_data/device.toml"),
    os.path.join(MODULE_TEST_PATH, "calibration_data/qubit.toml"),
    os.path.join(MODULE_TEST_PATH, "calibration_data/gate.toml"),
)


def profile_openapl(file_path):
    """Parses an OpenAPL JSON file and extracts statistics."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    num_parallel_protocols = 0
    num_beams = 0
    num_transitions = 0
    num_levels = 0
    num_ions = 0
    if "system" in data and "ions" in data["system"]:
        for ion in data["system"]["ions"]:
            if "levels" in ion:
                for _ in ion["levels"]:
                    num_levels += 1
            if "transitions" in ion:
                for _ in ion["transitions"]:
                    num_transitions += 1
            num_ions += 1
    if "protocol" in data and "sequence" in data["protocol"]:
        for item in data["protocol"]["sequence"]:
            if item.get("class_") == "ParallelProtocol":
                num_parallel_protocols += 1
                if "sequence" in item:
                    for sub_item in item["sequence"]:
                        if "beam" in sub_item and sub_item["beam"].get("class_") == "Beam":
                            num_beams += 1
    stats = {
        "num_parallel_protocols": num_parallel_protocols,
        "num_beams": num_beams,
        "num_transitions": num_transitions,
        "num_levels": num_levels,
        "num_ions": num_ions,
    }
    return stats


def verify_json(correct_file_name, expected_file_name):
    """Verify the two JSON files are identical."""
    if not os.path.exists(correct_file_name):
        raise FileNotFoundError(f"File {correct_file_name} does not exist")
    if not os.path.exists(expected_file_name):
        raise FileNotFoundError(f"File {expected_file_name} does not exist")

    with open(correct_file_name, "r", encoding="utf-8") as f:
        correct_json = json.load(f)
    with open(expected_file_name, "r", encoding="utf-8") as f:
        expected_json = json.load(f)

    return correct_json == expected_json


class TestTargetGates:
    """Test OQD device OpenAPL generation for target gates ({'RX', 'RY', 'MS'})."""

    def test_RX_gate(self, tmp_openapl_file_name):
        """Test OpenAPL generation for a circuit with a single RX Gate."""
        oqd_dev = OQDDevice(
            backend="default", shots=4, wires=1, openapl_file_name=tmp_openapl_file_name
        )

        @qjit(pipelines=OQD_PIPELINES)
        @qml.qnode(oqd_dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.counts(wires=0)

        circuit(np.pi / 2)

        expected_f = os.path.join(MODULE_TEST_PATH, "test_single_RX.json")
        assert verify_json(expected_f, oqd_dev.openapl_file_name)

    def test_RY_gate(self, tmp_openapl_file_name):
        """Test OpenAPL generation for a circuit with a single RY Gate."""
        oqd_dev = OQDDevice(
            backend="default", shots=4, wires=1, openapl_file_name=tmp_openapl_file_name
        )

        @qjit(pipelines=OQD_PIPELINES)
        @qml.qnode(oqd_dev)
        def circuit(x):
            qml.RY(x, wires=0)
            return qml.counts(wires=0)

        circuit(np.pi / 2)

        stats = profile_openapl(oqd_dev.openapl_file_name)
        assert stats["num_ions"] == 1
        assert stats["num_parallel_protocols"] == 1
        assert stats["num_beams"] == 2
        assert stats["num_transitions"] == 4
        assert stats["num_levels"] == 4
        with open(oqd_dev.openapl_file_name, "r", encoding="utf-8") as f:
            result_json = json.load(f)
        assert (
            result_json["protocol"]["sequence"][0]["sequence"][0]["beam"]["phase"]["value"]
            == 1.5707963267948966
        )


class TestChainedGates:
    """Test that the OQD device correctly generates an OpenAPL program for chained gates."""

    def test_RX_RY_gate(self, tmp_openapl_file_name):
        """Test OpenAPL generation for a circuit with a single RX and RY Gate."""
        oqd_dev = OQDDevice(
            backend="default", shots=4, wires=1, openapl_file_name=tmp_openapl_file_name
        )

        @qjit(pipelines=OQD_PIPELINES)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.RX(np.pi / 2, wires=0)
            qml.RY(np.pi / 2, wires=0)
            return qml.counts(wires=0)

        circuit()

        stats = profile_openapl(oqd_dev.openapl_file_name)
        assert stats["num_ions"] == 1
        assert stats["num_parallel_protocols"] == 2
        assert stats["num_beams"] == 4
        assert stats["num_transitions"] == 4
        assert stats["num_levels"] == 4


class TestDecomposableGates:
    """Test OQD device OpenAPL generation for gates decomposable into target gates."""

    def test_CNOT_gate(self, tmp_openapl_file_name):
        """Test OpenAPL generation for a circuit with a single CNOT circuit."""
        oqd_dev = OQDDevice(
            backend="default", shots=4, wires=2, openapl_file_name=tmp_openapl_file_name
        )

        @qjit(pipelines=OQD_PIPELINES)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.counts(wires=0)

        circuit()

        expected_f = os.path.join(MODULE_TEST_PATH, "test_single_CNOT.json")
        assert verify_json(expected_f, oqd_dev.openapl_file_name)

    def test_Hadamard_gate(self, tmp_openapl_file_name):
        """Test OpenAPL generation for a circuit with a single Hadamard gate."""
        oqd_dev = OQDDevice(
            backend="default", shots=4, wires=1, openapl_file_name=tmp_openapl_file_name
        )

        @qjit(pipelines=OQD_PIPELINES)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.counts(wires=0)

        circuit()

        stats = profile_openapl(oqd_dev.openapl_file_name)
        assert stats["num_ions"] == 1
        assert stats["num_parallel_protocols"] == 3
        assert stats["num_beams"] == 6
        assert stats["num_transitions"] == 4
        assert stats["num_levels"] == 4

    def test_PauliZ_gate(self, tmp_openapl_file_name):
        """Test OpenAPL generation for a circuit with a single PauliZ gate."""
        oqd_dev = OQDDevice(
            backend="default", shots=4, wires=1, openapl_file_name=tmp_openapl_file_name
        )

        @qjit(pipelines=OQD_PIPELINES)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.PauliZ(wires=0)
            return qml.counts(wires=0)

        circuit()

        stats = profile_openapl(oqd_dev.openapl_file_name)
        assert stats["num_ions"] == 1
        assert stats["num_parallel_protocols"] == 3
        assert stats["num_beams"] == 6
        assert stats["num_transitions"] == 4
        assert stats["num_levels"] == 4

    def test_PhaseShift_gate(self, tmp_openapl_file_name):
        """Test OpenAPL generation for a circuit with a single PhaseShift gate."""
        oqd_dev = OQDDevice(
            backend="default", shots=4, wires=1, openapl_file_name=tmp_openapl_file_name
        )

        @qjit(pipelines=OQD_PIPELINES)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.PhaseShift(np.pi / 4, wires=0)
            return qml.counts(wires=0)

        circuit()

        stats = profile_openapl(oqd_dev.openapl_file_name)
        assert stats["num_ions"] == 1
        assert stats["num_parallel_protocols"] == 3
        assert stats["num_beams"] == 6
        assert stats["num_transitions"] == 4
        assert stats["num_levels"] == 4

    def test_RZ_gate(self, tmp_openapl_file_name):
        """Test OpenAPL generation for a circuit with a single RZ gate."""
        oqd_dev = OQDDevice(
            backend="default", shots=4, wires=1, openapl_file_name=tmp_openapl_file_name
        )

        @qjit(pipelines=OQD_PIPELINES)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.RZ(np.pi / 4, wires=0)
            return qml.counts(wires=0)

        circuit()

        stats = profile_openapl(oqd_dev.openapl_file_name)
        assert stats["num_ions"] == 1
        assert stats["num_parallel_protocols"] == 3
        assert stats["num_beams"] == 6
        assert stats["num_transitions"] == 4
        assert stats["num_levels"] == 4

    def test_T_gate(self, tmp_openapl_file_name):
        """Test OpenAPL generation for a circuit with a single T gate."""
        oqd_dev = OQDDevice(
            backend="default", shots=4, wires=1, openapl_file_name=tmp_openapl_file_name
        )

        @qjit(pipelines=OQD_PIPELINES)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.T(wires=0)
            return qml.counts(wires=0)

        circuit()

        stats = profile_openapl(oqd_dev.openapl_file_name)
        assert stats["num_ions"] == 1
        assert stats["num_parallel_protocols"] == 3
        assert stats["num_beams"] == 6
        assert stats["num_transitions"] == 4
        assert stats["num_levels"] == 4

    def test_S_gate(self, tmp_openapl_file_name):
        """Test OpenAPL generation for a circuit with a single S gate."""
        oqd_dev = OQDDevice(
            backend="default", shots=4, wires=1, openapl_file_name=tmp_openapl_file_name
        )

        @qjit(pipelines=OQD_PIPELINES)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.S(wires=0)
            return qml.counts(wires=0)

        circuit()

        stats = profile_openapl(oqd_dev.openapl_file_name)
        assert stats["num_ions"] == 1
        assert stats["num_parallel_protocols"] == 3
        assert stats["num_beams"] == 6
        assert stats["num_transitions"] == 4
        assert stats["num_levels"] == 4


class TestComplexCircuits:
    """Test OQD device OpenAPL generation for more complex quantum circuits."""

    def test_2qubit_QFT(self, tmp_openapl_file_name):
        """Test OpenAPL generation for a 2-qubit Quantum Fourier Transform circuit."""
        wires = 2
        oqd_dev = OQDDevice(
            backend="default", shots=4, wires=wires, openapl_file_name=tmp_openapl_file_name
        )

        @qjit(pipelines=OQD_PIPELINES)
        @qml.qnode(oqd_dev)
        def circuit(basis_state):
            qml.BasisState(basis_state, wires=range(wires))
            qml.QFT(wires=range(wires))
            return qml.counts(wires=0)

        circuit(np.array([0, 1]))

        stats = profile_openapl(oqd_dev.openapl_file_name)
        assert stats["num_ions"] == 2
        assert stats["num_parallel_protocols"] == 54
        assert stats["num_beams"] == 128
        assert stats["num_transitions"] == 8
        assert stats["num_levels"] == 8


if __name__ == "__main__":
    pytest.main(["-x", __file__])
