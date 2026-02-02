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
"""Test for the device API."""
import pennylane as qml
import pytest
from pennylane.devices import NullQubit
from pennylane.devices.capabilities import DeviceCapabilities, ExecutionCondition

from catalyst import qjit
from catalyst.device import QJITDevice, get_device_capabilities, qjit_device
from catalyst.tracing.contexts import EvaluationContext, EvaluationMode

# pylint:disable = protected-access,attribute-defined-outside-init


def test_qjit_device():
    """Test the qjit device from a device using the new api."""

    # Create qjit device
    device = NullQubit(wires=10)
    device_qjit = QJITDevice(device)

    # Check attributes of the new device
    # Since shots are not used in the new API, we expect None
    assert device_qjit.shots == qml.measurements.Shots(None)
    assert device_qjit.wires == qml.wires.Wires(range(0, 10))

    # Check the preprocess of the new device
    with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
        transform_program, _ = device_qjit.preprocess(ctx)
    assert transform_program
    assert len(transform_program) == 3
    assert transform_program[-2].transform.__name__ == "verify_operations"
    assert transform_program[-1].transform.__name__ == "validate_measurements"

    # TODO: readd when we do not discard device preprocessing
    # t = transform_program[0].transform.__name__
    # assert t == "split_non_commuting"

    t = transform_program[0].transform.__name__
    assert t == "catalyst_decompose"

    # Check that the device cannot execute tapes
    with pytest.raises(RuntimeError, match="QJIT devices cannot execute tapes"):
        device_qjit.execute(10, 2)


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
    device = NullQubit()
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

    dev = qml.device("lightning.qubit", wires=2)
    state_measurements = {"StateMP"}
    finite_shot_measurements = {"CountsMP", "SampleMP"}

    dev_capabilities = get_device_capabilities(dev, shots)
    expected_measurements = dev_capabilities.measurement_processes

    if shots is None:
        # state measurements are present in expected_measurements, finite shot measurements are not
        assert state_measurements.issubset(expected_measurements)
        assert finite_shot_measurements.intersection(expected_measurements) == set()
    else:
        # finite shot measurements are present in expected_measurements, state measurements are not
        assert finite_shot_measurements.issubset(expected_measurements)
        assert state_measurements.intersection(expected_measurements) == set()

    spy = mocker.spy(qjit_device, "filter_device_capabilities_with_shots")

    @qjit
    @qml.set_shots(shots)
    @qml.qnode(dev)
    def circuit():
        qml.X(0)
        return qml.expval(qml.PauliZ(0))

    circuit()

    assert spy.spy_return.measurement_processes == expected_measurements


@pytest.mark.parametrize(
    "MPs, requires_shots",
    [
        (
            {
                "StateMP": [ExecutionCondition.ANALYTIC_MODE_ONLY],
                "CountsMP": [ExecutionCondition.FINITE_SHOTS_ONLY],
            },
            False,
        ),
        (
            {
                "SampleMP": [
                    ExecutionCondition.FINITE_SHOTS_ONLY,
                    ExecutionCondition.TERMS_MUST_COMMUTE,
                ],
                "CountsMP": [ExecutionCondition.FINITE_SHOTS_ONLY],
            },
            True,
        ),
    ],
)
def test_device_requires_shots(MPs, requires_shots):
    """Test that shots requirement is properly inferred from capabilities"""
    # Construct a mock DeviceCapabilities object.
    # Don't care about non MP capabilities, so just use default values.
    caps = DeviceCapabilities(measurement_processes=MPs)
    assert qjit_device._requires_shots(caps) == requires_shots


def test_simple_circuit():
    """Test that a circuit with the new device API is compiling to MLIR."""
    dev = NullQubit(wires=2)

    @qjit(target="mlir")
    @qml.set_shots(shots=2048)
    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(wires=0))

    assert circuit.mlir


def test_track_resources():
    """Test that resource tracking settings get passed to the device."""
    dev = NullQubit(wires=2)
    assert "track_resources" in QJITDevice.extract_backend_info(dev).kwargs
    assert QJITDevice.extract_backend_info(dev).kwargs["track_resources"] is False

    dev = NullQubit(wires=2, track_resources=True)
    assert "track_resources" in QJITDevice.extract_backend_info(dev).kwargs
    assert QJITDevice.extract_backend_info(dev).kwargs["track_resources"] is True
    assert "resources_filename" not in QJITDevice.extract_backend_info(dev).kwargs
    assert "compute_depth" not in QJITDevice.extract_backend_info(dev).kwargs

    dev = NullQubit(
        wires=2, track_resources=True, resources_filename="my_resources.txt", compute_depth=True
    )
    assert "track_resources" in QJITDevice.extract_backend_info(dev).kwargs
    assert QJITDevice.extract_backend_info(dev).kwargs["track_resources"] is True
    assert "resources_filename" in QJITDevice.extract_backend_info(dev).kwargs
    assert QJITDevice.extract_backend_info(dev).kwargs["resources_filename"] == "my_resources.txt"
    assert "compute_depth" in QJITDevice.extract_backend_info(dev).kwargs
    assert QJITDevice.extract_backend_info(dev).kwargs["compute_depth"] is True


if __name__ == "__main__":
    pytest.main(["-x", __file__])
