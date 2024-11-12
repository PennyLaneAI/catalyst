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
"""Test for the device API.
"""
import pennylane as qml
import pytest
from pennylane.devices import NullQubit

from catalyst import qjit
from catalyst.device import QJITDevice, get_device_capabilities, qjit_device
from catalyst.tracing.contexts import EvaluationContext, EvaluationMode

# pylint:disable = protected-access,attribute-defined-outside-init


def test_qjit_device():
    """Test the qjit device from a device using the new api."""
    device = NullQubit(wires=10, shots=2032)

    # Create qjit device
    device_qjit = QJITDevice(device)

    # Check attributes of the new device
    assert device_qjit.shots == qml.measurements.Shots(2032)
    assert device_qjit.wires == qml.wires.Wires(range(0, 10))

    # Check the preprocess of the new device
    with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
        transform_program, _ = device_qjit.preprocess(ctx)
    assert transform_program
    assert len(transform_program) == 3
    assert transform_program[-2]._transform.__name__ == "verify_operations"
    assert transform_program[-1]._transform.__name__ == "validate_measurements"

    # TODO: readd when we do not discard device preprocessing
    # t = transform_program[0].transform.__name__
    # assert t == "split_non_commuting"

    t = transform_program[0].transform.__name__
    assert t == "catalyst_decompose"

    # Check that the device cannot execute tapes
    with pytest.raises(RuntimeError, match="QJIT devices cannot execute tapes"):
        device_qjit.execute(10, 2)


def test_qjit_device_no_wires():
    """Test the qjit device from a device using the new api without wires set."""
    device = NullQubit(shots=2032)

    with pytest.raises(
        AttributeError, match="Catalyst does not support device instances without set wires."
    ):
        # Create qjit device
        QJITDevice(device)


@pytest.mark.parametrize(
    "wires",
    (
        qml.wires.Wires(["a", "b"]),
        qml.wires.Wires([0, 2, 4]),
        qml.wires.Wires([1, 2, 3]),
    ),
)
def test_qjit_device_invalid_wires(wires):
    """Test the qjit device from a device using the new api without wires set."""
    device = NullQubit(shots=2032)
    device._wires = wires

    with pytest.raises(
        AttributeError, match="Catalyst requires continuous integer wire labels starting at 0"
    ):
        # Create qjit device
        QJITDevice(device)


@pytest.mark.parametrize("shots", [2048, None])
def test_qjit_device_measurements(shots, mocker):
    """Test that the list of measurements that are supported is correctly
    updated based on shots provided to the device"""

    spy = mocker.spy(qjit_device, "get_device_capabilities")

    dev = qml.device("lightning.qubit", wires=2, shots=shots)
    state_measurements = {"StateMP"}
    finite_shot_measurements = {"CountsMP", "SampleMP"}

    capabilities = get_device_capabilities(dev)
    all_measurements = set(capabilities.measurement_processes)

    assert state_measurements.issubset(all_measurements)
    assert finite_shot_measurements.issubset(all_measurements)

    dev_capabilities = get_device_capabilities(dev)
    expected_measurements = dev_capabilities.measurement_processes

    if shots is None:
        # state measurements are present in expected_measurements, finite shot measurements are not
        assert state_measurements.issubset(expected_measurements)
        assert finite_shot_measurements.intersection(expected_measurements) == set()
    else:
        # finite shot measurements are present in expected_measurements, state measurements are not
        assert finite_shot_measurements.issubset(expected_measurements)
        assert state_measurements.intersection(expected_measurements) == set()

    spy = mocker.spy(qjit_device, "get_qjit_device_capabilities")

    @qjit
    @qml.qnode(dev)
    def circuit():
        qml.X(0)
        return qml.expval(qml.PauliZ(0))

    circuit()

    assert spy.spy_return.measurement_processes == expected_measurements


def test_simple_circuit():
    """Test that a circuit with the new device API is compiling to MLIR."""
    dev = NullQubit(wires=2, shots=2048)

    @qjit(target="mlir")
    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(wires=0))

    assert circuit.mlir


if __name__ == "__main__":
    pytest.main(["-x", __file__])
