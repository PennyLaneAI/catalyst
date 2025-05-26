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
from pennylane_ionq import ops

from catalyst import qjit
from catalyst.third_party.oqd import OQDDevice, OQDDevicePipeline


@pytest.fixture(scope="module")
def oqd_pipelines():
    """Get the OQD device pipelines."""
    test_path = os.path.dirname(__file__)
    toml_path = os.path.join(test_path, "calibration_data/")
    return OQDDevicePipeline(
        toml_path + "device.toml", toml_path + "qubit.toml", toml_path + "gate.toml"
    )


@pytest.fixture(scope="function")
def result_openapl_file():
    """Create a temporary OpenAPL file with the given content."""
    openapl_file = "__openapl__output.json"
    yield openapl_file
    # os.remove(openapl_file)


def verify_json(correct_file_name, expected_file_name):
    """Verify the two JSON files are identical."""
    with open(correct_file_name, "r", encoding="utf-8") as f:
        correct_json = json.load(f)

    with open(expected_file_name, "r", encoding="utf-8") as f:
        expected_json = json.load(f)

    return sorted(correct_json.items()) == sorted(expected_json.items())


class TestTargetGates:
    """Test OQD device OpenAPL generation for target gates ({'RX', 'RY', 'MS'})."""

    test_path = os.path.dirname(__file__)

    def test_RX_gate(self, oqd_pipelines, result_openapl_file):
        """Test OpenAPL generation for a circuit with a single RX Gate."""
        oqd_dev = OQDDevice(backend="default", shots=4, wires=1)

        @qjit(pipelines=oqd_pipelines)
        @qml.qnode(oqd_dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.counts(wires=0)

        circuit(np.pi / 2)

        expected_f = os.path.join(self.test_path, "test_single_RX.json")
        assert verify_json(expected_f, result_openapl_file)

    def test_RY_gate(self, oqd_pipelines, result_openapl_file):
        """Test OpenAPL generation for a circuit with a single RY Gate."""
        oqd_dev = OQDDevice(backend="default", shots=4, wires=1)

        @qjit(pipelines=oqd_pipelines)
        @qml.qnode(oqd_dev)
        def circuit(x):
            qml.RY(x, wires=0)
            return qml.counts(wires=0)

        circuit(np.pi / 2)

        expected_f = os.path.join(self.test_path, "test_single_RY.json")
        assert verify_json(expected_f, result_openapl_file)


class TestChainedGates:
    """Test that the OQD device correctly generates an OpenAPL program for chained gates."""

    test_path = os.path.dirname(__file__)

    def test_RX_RY_gate(self, oqd_pipelines, result_openapl_file):
        """Test OpenAPL generation for a circuit with a single RX and RY Gate."""
        oqd_dev = OQDDevice(backend="default", shots=4, wires=1)

        @qjit(pipelines=oqd_pipelines)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.RX(np.pi / 2, wires=0)
            qml.RY(np.pi / 2, wires=0)
            return qml.counts(wires=0)

        circuit()

        expected_f = os.path.join(self.test_path, "test_RX_RY.json")
        assert verify_json(expected_f, result_openapl_file)


class TestDecomposableGates:
    """Test OQD device OpenAPL generation for gates decomposable into target gates."""

    test_path = os.path.dirname(__file__)

    def test_CNOT_gate(self, oqd_pipelines, result_openapl_file):
        """Test OpenAPL generation for a circuit with a single CNOT circuit."""
        oqd_dev = OQDDevice(backend="default", shots=4, wires=2)

        @qjit(pipelines=oqd_pipelines)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.counts(wires=0)

        circuit()

        expected_f = os.path.join(self.test_path, "test_single_CNOT.json")
        assert verify_json(expected_f, result_openapl_file)

    def test_Hadamard_gate(self, oqd_pipelines, result_openapl_file):
        """Test OpenAPL generation for a circuit with a single Hadamard gate."""
        oqd_dev = OQDDevice(backend="default", shots=4, wires=1)

        @qjit(pipelines=oqd_pipelines)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.counts(wires=0)

        circuit()

        expected_f = os.path.join(self.test_path, "test_single_Hadamard.json")
        assert verify_json(expected_f, result_openapl_file)

    def test_PauliZ_gate(self, oqd_pipelines, result_openapl_file):
        """Test OpenAPL generation for a circuit with a single PauliZ gate."""
        oqd_dev = OQDDevice(backend="default", shots=4, wires=1)

        @qjit(pipelines=oqd_pipelines)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.PauliZ(wires=0)
            return qml.counts(wires=0)

        circuit()

        expected_f = os.path.join(self.test_path, "test_single_PauliZ.json")
        assert verify_json(expected_f, result_openapl_file)

    def test_PhaseShift_gate(self, oqd_pipelines, result_openapl_file):
        """Test OpenAPL generation for a circuit with a single PhaseShift gate."""
        oqd_dev = OQDDevice(backend="default", shots=4, wires=1)

        @qjit(pipelines=oqd_pipelines)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.PhaseShift(np.pi / 4, wires=0)
            return qml.counts(wires=0)

        circuit()

        expected_f = os.path.join(self.test_path, "test_single_PhaseShift.json")
        assert verify_json(expected_f, result_openapl_file)

    def test_RZ_gate(self, oqd_pipelines, result_openapl_file):
        """Test OpenAPL generation for a circuit with a single RZ gate."""
        oqd_dev = OQDDevice(backend="default", shots=4, wires=1)

        @qjit(pipelines=oqd_pipelines)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.RZ(np.pi / 4, wires=0)
            return qml.counts(wires=0)

        circuit()

        expected_f = os.path.join(self.test_path, "test_single_RZ.json")
        assert verify_json(expected_f, result_openapl_file)

    def test_T_gate(self, oqd_pipelines, result_openapl_file):
        """Test OpenAPL generation for a circuit with a single T gate."""
        oqd_dev = OQDDevice(backend="default", shots=4, wires=1)

        @qjit(pipelines=oqd_pipelines)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.T(wires=0)
            return qml.counts(wires=0)

        circuit()

        expected_f = os.path.join(self.test_path, "test_single_T.json")
        assert verify_json(expected_f, result_openapl_file)

    def test_S_gate(self, oqd_pipelines, result_openapl_file):
        """Test OpenAPL generation for a circuit with a single S gate."""
        oqd_dev = OQDDevice(backend="default", shots=4, wires=1)

        @qjit(pipelines=oqd_pipelines)
        @qml.qnode(oqd_dev)
        def circuit():
            qml.S(wires=0)
            return qml.counts(wires=0)

        circuit()

        expected_f = os.path.join(self.test_path, "test_single_S.json")
        assert verify_json(expected_f, result_openapl_file)


class TestComplexCircuits:
    """Test OQD device OpenAPL generation for more complex quantum circuits."""

    test_path = os.path.dirname(__file__)

    def test_2qubit_QFT(self, oqd_pipelines, result_openapl_file):
        """Test OpenAPL generation for a 2-qubit Quantum Fourier Transform circuit."""
        wires = 2
        oqd_dev = OQDDevice(backend="default", shots=4, wires=wires)

        @qjit(pipelines=oqd_pipelines)
        @qml.qnode(oqd_dev)
        def circuit(basis_state):
            qml.BasisState(basis_state, wires=range(wires))
            qml.QFT(wires=range(wires))
            return qml.counts(wires=0)

        circuit(np.array([0, 1]))

        expected_f = os.path.join(self.test_path, "test_2qubit_QFT.json")
        assert verify_json(expected_f, result_openapl_file)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
