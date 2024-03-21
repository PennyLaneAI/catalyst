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
"""Test for the device preprocessing.
"""
# pylint: disable=unused-argument
import pathlib

import numpy as np
import pennylane as qml
import pytest
from pennylane.devices import Device
from pennylane.devices.execution_config import DefaultExecutionConfig, ExecutionConfig
from pennylane.transforms import split_non_commuting
from pennylane.transforms.core import TransformProgram

from catalyst import CompileError, ctrl
from catalyst.compiler import get_lib_path
from catalyst.preprocess import decompose_ops_to_unitary, measurements_from_counts


class DummyDevice(Device):
    """A dummy device from the device API."""

    config = pathlib.Path(__file__).parent.parent.parent.parent.joinpath(
        "runtime/tests/third_party/dummy_device.toml"
    )

    def __init__(self, wires, shots=1024):
        print(pathlib.Path(__file__).parent.parent.parent.parent)
        super().__init__(wires=wires, shots=shots)

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """

        return "dummy.remote", get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/libdummy_device.so"

    def execute(self, circuits, execution_config):
        """Execution."""
        return circuits, execution_config

    def preprocess(self, execution_config: ExecutionConfig = DefaultExecutionConfig):
        """Preprocessing."""
        transform_program = TransformProgram()
        transform_program.add_transform(split_non_commuting)
        return transform_program, execution_config


class TestPreprocess:
    """Test the preprocessing transforms implemented in Catalyst."""

    @pytest.mark.skipif(
        not pathlib.Path(
            get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/libdummy_device.so"
        ).is_file(),
        reason="lib_dummydevice.so was not found.",
    )
    def test_decompose_integration(self):
        """Test the decompose transform as part of the Catalyst pipeline."""
        dev = DummyDevice(wires=4)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(theta: float):
            qml.SingleExcitationPlus(theta, wires=[0, 1])
            return qml.state()

        mlir = qml.qjit(circuit, target="mlir").mlir
        assert "PauliX" in mlir
        assert "CNOT" in mlir
        assert "ControlledPhaseShift" in mlir
        assert "SingleExcitationPlus" not in mlir

    def test_decompose_ops_to_unitary(self):
        """Test the decompose ops to unitary transform."""
        operations = [qml.CNOT(wires=[0, 1]), qml.RX(0.1, wires=0)]
        tape = qml.tape.QuantumScript(ops=operations)
        ops_to_decompose = ["CNOT"]

        tapes, _ = decompose_ops_to_unitary(tape, ops_to_decompose)
        decomposed_ops = tapes[0].operations
        assert isinstance(decomposed_ops[0], qml.QubitUnitary)
        assert isinstance(decomposed_ops[1], qml.RX)

    @pytest.mark.skipif(
        not pathlib.Path(
            get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/libdummy_device.so"
        ).is_file(),
        reason="lib_dummydevice.so was not found.",
    )
    def test_decompose_ops_to_unitary_integration(self):
        """Test the decompose ops to unitary transform as part of the Catalyst pipeline."""
        dev = DummyDevice(wires=4)

        @qml.qjit
        @qml.qnode(dev)
        def circuit():
            qml.BlockEncode(np.array([[1, 1, 1], [0, 1, 0]]), wires=[0, 1, 2])
            return qml.state()

        mlir = qml.qjit(circuit, target="mlir").mlir
        assert "quantum.unitary" in mlir
        assert "BlockEncode" not in mlir

    @pytest.mark.skipif(
        not pathlib.Path(
            get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/libdummy_device.so"
        ).is_file(),
        reason="lib_dummydevice.so was not found.",
    )
    def test_no_matrix(self):
        """Test that controlling an operation without a matrix method raises an error."""
        dev = DummyDevice(wires=4)

        class OpWithNoMatrix(qml.operation.Operation):
            """Op without matrix."""

            num_wires = qml.operation.AnyWires

            def matrix(self, wire_order=None):
                """Matrix is overriden."""
                raise NotImplementedError()

        @qml.qnode(dev)
        def f():
            ctrl(OpWithNoMatrix(wires=[0, 1]), control=[2, 3])
            return qml.probs()

        with pytest.raises(CompileError, match="could not be decomposed, it might be unsupported"):
            qml.qjit(f, target="jaxpr")

    @pytest.mark.skipif(
        not pathlib.Path(
            get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/libdummy_device.so"
        ).is_file(),
        reason="lib_dummydevice.so was not found.",
    )
    def test_measurement_from_counts_integration_multiple_measurements(self):
        """Test the measurment from counts transform as part of the Catalyst pipeline."""
        dev = DummyDevice(wires=4, shots=1000)

        @qml.qjit
        @measurements_from_counts
        @qml.qnode(dev)
        def circuit(theta: float):
            qml.X(0)
            qml.X(1)
            qml.X(2)
            qml.X(3)
            return (
                qml.expval(qml.PauliX(wires=0) @ qml.PauliX(wires=1)),
                qml.var(qml.PauliX(wires=0) @ qml.PauliX(wires=2)),
                qml.counts(qml.PauliX(wires=0) @ qml.PauliX(wires=1) @ qml.PauliX(wires=2)),
            )

        mlir = qml.qjit(circuit, target="mlir").mlir
        assert "expval" not in mlir
        assert "var" not in mlir
        assert "counts" in mlir

    @pytest.mark.skipif(
        not pathlib.Path(
            get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/libdummy_device.so"
        ).is_file(),
        reason="lib_dummydevice.so was not found.",
    )
    def test_measurement_from_counts_integration_single_measurement(self):
        """Test the measurment from counts transform with a single measurements as part of 
        the Catalyst pipeline."""
        dev = DummyDevice(wires=4, shots=1000)

        @qml.qjit
        @measurements_from_counts
        @qml.qnode(dev)
        def circuit(theta: float):
            qml.X(0)
            qml.X(1)
            qml.X(2)
            qml.X(3)
            return qml.expval(qml.PauliX(wires=0) @ qml.PauliX(wires=1))

        mlir = qml.qjit(circuit, target="mlir").mlir
        assert "expval" not in mlir
        assert "counts" in mlir


class TestTransform:
    """Test the transforms implemented in Catalyst."""

    def test_measurements_from_counts(self):
        """Test the transfom measurements_from_counts."""
        device = qml.device("lightning.qubit", wires=4, shots=1000)

        @qml.qjit
        @measurements_from_counts
        @qml.qnode(device=device)
        def circuit(a: float):
            qml.X(0)
            qml.X(1)
            qml.X(2)
            qml.X(3)
            return (
                qml.expval(qml.PauliX(wires=0) @ qml.PauliX(wires=1)),
                qml.var(qml.PauliX(wires=0) @ qml.PauliX(wires=2)),
                qml.probs(wires=[3]),
                qml.counts(qml.PauliX(wires=0) @ qml.PauliX(wires=1) @ qml.PauliX(wires=2)),
            )

        res = circuit(0.2)
        results = res[0]

        assert isinstance(results, tuple)
        assert len(results) == 4

        expval = results[0]
        var = results[1]
        probs = results[2]
        counts = results[3]

        assert expval.shape == ()
        assert var.shape == ()
        assert probs.shape == (2,)
        assert isinstance(counts, tuple)
        assert len(counts) == 2
        assert counts[0].shape == (8,)
        assert counts[1].shape == (8,)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
