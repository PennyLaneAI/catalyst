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
from catalyst.preprocess import decompose_ops_to_unitary


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
            num_wires = qml.operation.AnyWires

            def matrix(self):
                raise NotImplementedError()

        @qml.qnode(dev)
        def f():
            ctrl(OpWithNoMatrix(wires=[0, 1]), control=[2, 3])
            return qml.probs()

        with pytest.raises(CompileError, match="could not be decomposed, it might be unsupported"):
            qml.qjit(f, target="jaxpr")


if __name__ == "__main__":
    pytest.main(["-x", __file__])
