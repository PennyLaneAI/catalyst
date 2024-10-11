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
import os
import pathlib

# pylint: disable=unused-argument
import platform
import tempfile
from functools import partial
from unittest.mock import Mock, patch

import numpy as np
import pennylane as qml
import pytest
from pennylane.devices import Device
from pennylane.transforms import split_non_commuting, split_to_single_terms

from catalyst.compiler import get_lib_path
from catalyst.device import QJITDevice, get_device_capabilities, get_device_toml_config
from catalyst.device.decomposition import (
    measurements_from_counts,
    measurements_from_samples,
)
from catalyst.tracing.contexts import EvaluationContext, EvaluationMode
from catalyst.utils.toml import OperationProperties
from frontend.test.conftest import CONFIG_CUSTOM_DEVICE

# pylint: disable=attribute-defined-outside-init


class CustomDevice(Device):
    """A Custom Device following the new API."""

    config = CONFIG_CUSTOM_DEVICE

    def __init__(self, wires, shots=1024):
        print(pathlib.Path(__file__).parent.parent.parent.parent)
        super().__init__(wires=wires, shots=shots)
        dummy_capabilities = get_device_capabilities(self)
        dummy_capabilities.native_ops.pop("BlockEncode")
        dummy_capabilities.to_matrix_ops["BlockEncode"] = OperationProperties(False, False, False)
        self.capabilities = dummy_capabilities

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """
        system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
        # Borrowing the NullQubit library:
        lib_path = (
            get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_null_qubit" + system_extension
        )
        return "CustomQubit", lib_path

    def execute(self, circuits, execution_config):
        """Execution."""
        return circuits, execution_config


class CustomDeviceLimitedMPs(Device):
    """A Custom Device from the device API without wires."""

    config = CONFIG_CUSTOM_DEVICE

    def __init__(self, wires, shots=1024, allow_counts=False, allow_samples=False):
        self.allow_samples = allow_samples
        self.allow_counts = allow_counts

        super().__init__(wires=wires, shots=shots)

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """

        system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
        # Borrowing the NullQubit library:
        lib_path = (
            get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_null_qubit" + system_extension
        )
        return "CustomDevice", lib_path

    def execute(self, circuits, execution_config):
        """Execution."""
        return circuits, execution_config

    def __enter__(self, *args, **kwargs):
        toml_file_path = self.config
        with open(toml_file_path, mode="r", encoding="UTF-8") as f:
            toml_contents = f.readlines()

        updated_toml_contents = []
        for line in toml_contents:
            if "Expval" in line:
                continue
            if "Var" in line:
                continue
            if "Probs" in line:
                continue
            if "Sample" in line and not self.allow_samples:
                continue
            if "Counts" in line and not self.allow_counts:
                continue

            updated_toml_contents.append(line)

        self.toml_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        self.toml_file.writelines(updated_toml_contents)
        self.toml_file.close()  # close for now without deleting

        self.config = self.toml_file.name
        return self

    def __exit__(self, *args, **kwargs):
        os.unlink(self.toml_file.name)
        self.config = None


class TestMeasurementTransforms:
    """Tests for transforms modifying measurements"""

    def test_measurements_from_counts_multiple_measurements(self):
        """Test the transforms for measurements_from_counts to other measurement types
        as part of the Catalyst pipeline."""

        dev = qml.device("lightning.qubit", wires=4, shots=5000)

        @qml.qnode(dev)
        def basic_circuit(theta: float):
            qml.RY(theta, 0)
            qml.RY(theta / 2, 1)
            qml.RY(2 * theta, 2)
            qml.RY(theta, 3)
            return (
                qml.expval(qml.PauliX(wires=0) @ qml.PauliX(wires=1)),
                qml.var(qml.PauliX(wires=2)),
                qml.counts(qml.PauliX(wires=0) @ qml.PauliX(wires=1) @ qml.PauliX(wires=2)),
                qml.probs(wires=[3]),
            )

        transformed_circuit = measurements_from_counts(basic_circuit, dev.wires)

        mlir = qml.qjit(transformed_circuit, target="mlir").mlir
        assert "expval" not in mlir
        assert "quantum.var" not in mlir
        assert "counts" in mlir

        theta = 1.9
        expval_res, var_res, counts_res, probs_res = qml.qjit(transformed_circuit, seed=37)(theta)

        expval_expected = np.sin(theta) * np.sin(theta / 2)
        var_expected = 1 - np.sin(2 * theta) ** 2
        counts_expected = basic_circuit(theta)[2]
        probs_expected = [np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2]

        assert np.isclose(expval_res, expval_expected, atol=0.05)
        assert np.isclose(var_res, var_expected, atol=0.05)
        assert np.allclose(probs_res, probs_expected, atol=0.05)

        # counts comparison by converting catalyst format to PL style eigvals dict
        basis_states, counts = counts_res
        num_excitations_per_state = [
            sum(int(i) for i in format(int(state), "01b")) for state in basis_states
        ]
        eigvals = [(-1) ** i for i in num_excitations_per_state]
        eigval_counts_res = {
            -1.0: sum(count for count, eigval in zip(counts, eigvals) if eigval == -1),
            1.0: sum(count for count, eigval in zip(counts, eigvals) if eigval == 1),
        }

        # +/- 200 shots is pretty reasonable with 5000 shots total
        assert np.isclose(eigval_counts_res[-1], counts_expected[-1], atol=200)
        assert np.isclose(eigval_counts_res[1], counts_expected[1], atol=200)

    def test_measurements_from_samples_multiple_measurements(self):
        """Test the transform measurements_from_samples with multiple measurement types
        as part of the Catalyst pipeline."""

        dev = qml.device("lightning.qubit", wires=4, shots=5000)

        @qml.qnode(dev)
        def basic_circuit(theta: float):
            qml.RY(theta, 0)
            qml.RY(theta / 2, 1)
            qml.RY(2 * theta, 2)
            qml.RY(theta, 3)
            return (
                qml.expval(qml.PauliX(wires=0) @ qml.PauliX(wires=1)),
                qml.var(qml.PauliX(wires=2)),
                qml.sample(qml.PauliX(wires=0) @ qml.PauliX(wires=1) @ qml.PauliX(wires=2)),
                qml.probs(wires=[3]),
            )

        transformed_circuit = measurements_from_samples(basic_circuit, dev.wires)

        mlir = qml.qjit(transformed_circuit, target="mlir").mlir
        assert "expval" not in mlir
        assert "quantum.var" not in mlir
        assert "sample" in mlir

        theta = 1.9

        expval_res, var_res, sample_res, probs_res = qml.qjit(transformed_circuit, seed=37)(theta)

        expval_expected = np.sin(theta) * np.sin(theta / 2)
        var_expected = 1 - np.sin(2 * theta) ** 2
        sample_expected = basic_circuit(theta)[2]
        probs_expected = [np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2]

        assert np.isclose(expval_res, expval_expected, atol=0.05)
        assert np.isclose(var_res, var_expected, atol=0.05)
        assert np.allclose(probs_res, probs_expected, atol=0.05)

        # sample comparison
        assert np.isclose(np.mean(sample_res), np.mean(sample_expected), atol=0.05)
        assert len(sample_res) == len(sample_expected)
        assert set(np.array(sample_res)) == set(sample_expected)

    @pytest.mark.parametrize(
        "unsupported_measurement, measurement_transform, target_measurement",
        [
            ("Sample", measurements_from_counts, "counts"),
            ("Counts", measurements_from_samples, "sample"),
            (None, measurements_from_samples, "sample"),
        ],
    )
    def test_measurement_from_readout_integration_if_no_observables_supported(
        self, unsupported_measurement, measurement_transform, target_measurement
    ):
        """Test that for devices without observable support,  measurment_from_samples transform
        is applied as part of the Catalyst pipeline if the device only supports sample, and
        measurement_from_counts transform is applied if the device only supports counts. If
        both are supported, sample takes precedence."""

        dev = qml.device("lightning.qubit", wires=4, shots=100)

        config = get_device_toml_config(dev)
        config["operators"]["observables"] = {}
        if unsupported_measurement:
            del config["measurement_processes"][unsupported_measurement]

        with patch("catalyst.device.qjit_device.get_device_toml_config", Mock(return_value=config)):
            # transform is added to transform program
            qjit_dev = QJITDevice(dev)

            with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
                transform_program, _ = qjit_dev.preprocess(ctx)

            assert split_non_commuting in transform_program
            assert measurement_transform in transform_program

            # MLIR only contains target measurement
            @qml.qjit
            @qml.qnode(dev)
            def circuit(theta: float):
                qml.X(0)
                qml.X(1)
                qml.X(2)
                qml.X(3)
                return (
                    qml.expval(qml.PauliX(wires=0) @ qml.PauliX(wires=1)),
                    qml.var(qml.PauliX(wires=0) @ qml.PauliX(wires=2)),
                    qml.probs(wires=[3, 4]),
                )

            mlir = qml.qjit(circuit, target="mlir").mlir

        assert "expval" not in mlir
        assert "quantum.var" not in mlir
        assert "probs" not in mlir
        assert target_measurement in mlir

    @pytest.mark.parametrize(
        "device_measurements, measurement_transform, target_measurement",
        [
            (["counts"], measurements_from_counts, "counts"),
            (["sample"], measurements_from_samples, "sample"),
            (["counts", "sample"], measurements_from_samples, "sample"),
        ],
    )
    def test_measurement_from_readout_if_only_readout_measurements_supported(
        self, device_measurements, measurement_transform, target_measurement
    ):
        """Test the measurment_from_samples transform is applied as part of the Catalyst pipeline
        if the device only supports sample, and measurement_from_counts transform is applied if
        the device only supports counts. If both are supported, sample takes precedence."""

        allow_sample = "sample" in device_measurements
        allow_counts = "counts" in device_measurements

        with CustomDeviceLimitedMPs(
            wires=4, shots=1000, allow_counts=allow_counts, allow_samples=allow_sample
        ) as dev:

            # transform is added to transform program
            qjit_dev = QJITDevice(dev)

            with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
                transform_program, _ = qjit_dev.preprocess(ctx)

            assert split_non_commuting in transform_program
            assert measurement_transform in transform_program

            # MLIR only contains target measurement
            @qml.qjit
            @qml.qnode(dev)
            def circuit(theta: float):
                qml.X(0)
                qml.X(1)
                qml.X(2)
                qml.X(3)
                return (
                    qml.expval(qml.PauliX(wires=0) @ qml.PauliX(wires=1)),
                    qml.var(qml.PauliX(wires=0) @ qml.PauliX(wires=2)),
                    qml.probs(wires=[3, 4]),
                )

            mlir = qml.qjit(circuit, target="mlir").mlir

        assert "expval" not in mlir
        assert "quantum.var" not in mlir
        assert "probs" not in mlir
        assert target_measurement in mlir

    def test_error_is_raised_if_no_observables_and_no_samples_or_counts(self, mocker):
        """Test that for a device that doesn't support observables, if counts
        and sample are also both unsupported, an error is raised."""

        # no shots - samples/counts unsupported
        dev = qml.device("lightning.qubit", wires=3)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.X(0)), qml.var(qml.Y(1))

        # modify config to indicate no observables supported
        config = get_device_toml_config(dev)
        config["operators"]["observables"] = {}

        with patch("catalyst.device.qjit_device.get_device_toml_config", Mock(return_value=config)):
            with pytest.raises(
                RuntimeError, match="The device does not support observables or sample/counts"
            ):
                qml.qjit(circuit)()

    # pylint: disable=unnecessary-lambda
    @pytest.mark.parametrize(
        "measurement",
        [
            lambda: qml.counts(),
            lambda: qml.counts(wires=[2]),
            lambda: qml.counts(wires=[2, 3]),
            lambda: qml.counts(qml.Y(1)),
        ],
    )
    def test_measurement_from_counts_with_counts_measurement(self, measurement):
        """Test the measurment_from_counts transform with a single counts measurement as part of
        the Catalyst pipeline."""

        dev = qml.device("lightning.qubit", wires=4, shots=3000)

        @qml.qnode(dev)
        def circuit(theta: float):
            qml.RX(theta, 0)
            qml.RX(theta / 2, 1)
            qml.RX(theta / 3, 2)
            return measurement()

        theta = 2.5
        counts_expected = circuit(theta)
        res = qml.qjit(measurements_from_counts(circuit, dev.wires), seed=37)(theta)

        # counts comparison by converting catalyst format to PL style eigvals dict
        basis_states, counts = res

        if measurement().obs:
            num_excitations_per_state = [
                sum(int(i) for i in format(int(state), "01b")) for state in basis_states
            ]
            eigvals = [(-1) ** i for i in num_excitations_per_state]
            eigval_counts_res = {
                -1.0: sum(count for count, eigval in zip(counts, eigvals) if eigval == -1),
                1.0: sum(count for count, eigval in zip(counts, eigvals) if eigval == 1),
            }

            # +/- 100 shots is pretty reasonable with 3000 shots total
            assert np.isclose(eigval_counts_res[-1], counts_expected[-1], atol=100)
            assert np.isclose(eigval_counts_res[1], counts_expected[1], atol=100)

        else:
            num_wires = len(measurement().wires) if measurement().wires else len(dev.wires)
            basis_states = [format(int(state), "01b").zfill(num_wires) for state in basis_states]
            counts = [int(c) for c in counts]
            counts_dict = dict((state, c) for (state, c) in zip(basis_states, counts) if c != 0)

            for res, expected_res in zip(counts_dict.items(), counts_expected.items()):
                assert res[0] == expected_res[0]
                assert np.isclose(res[1], expected_res[1], atol=100)

    # pylint: disable=unnecessary-lambda
    @pytest.mark.parametrize(
        "measurement",
        [
            lambda: qml.sample(),
            lambda: qml.sample(wires=[0]),
            lambda: qml.sample(wires=[1, 2]),
            lambda: qml.sample(qml.Y(1) @ qml.Y(0)),
        ],
    )
    def test_measurement_from_samples_with_sample_measurement(self, measurement):
        """Test the measurment_from_counts transform with a single counts measurement as part of
        the Catalyst pipeline."""

        dev = qml.device("lightning.qubit", wires=4, shots=3000)

        @qml.qnode(dev)
        def circuit(theta: float):
            qml.RX(theta, 0)
            qml.RX(theta / 2, 1)
            return measurement()

        theta = 2.5
        res = qml.qjit(measurements_from_samples(circuit, dev.wires), seed=37)(theta)

        if len(measurement().wires) == 1:
            samples_expected = qml.qjit(circuit)(theta)
        else:
            samples_expected = circuit(theta)

        assert res.shape == samples_expected.shape
        assert np.allclose(np.mean(res, axis=0), np.mean(samples_expected, axis=0), atol=0.05)

    # pylint: disable=unnecessary-lambda
    @pytest.mark.parametrize(
        "input_measurement, expected_res",
        [
            (
                lambda: qml.expval(qml.PauliY(wires=0) @ qml.PauliY(wires=1)),
                lambda theta: np.sin(theta) * np.sin(theta / 2),
            ),
            (lambda: qml.var(qml.Y(wires=1)), lambda theta: 1 - np.sin(theta / 2) ** 2),
            (
                lambda: qml.probs(),
                lambda theta: np.outer(
                    np.outer(
                        [np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2],
                        [np.cos(theta / 4) ** 2, np.sin(theta / 4) ** 2],
                    ),
                    [1, 0, 0, 0],
                ).flatten(),
            ),
            (
                lambda: qml.probs(wires=[1]),
                lambda theta: [np.cos(theta / 4) ** 2, np.sin(theta / 4) ** 2],
            ),
        ],
    )
    @pytest.mark.parametrize("shots", [3000, (3000, 4000), (3000, 3500, 4000)])
    def test_measurement_from_samples_single_measurement_analytic(
        self,
        input_measurement,
        expected_res,
        shots,
    ):
        """Test the measurement_from_samples transform with a single measurements as part of the
        Catalyst pipeline, for measurements whose outcome can be directly compared to an expected
        analytic result."""

        dev = qml.device("lightning.qubit", wires=4, shots=shots)

        @qml.qjit(seed=37)
        @partial(measurements_from_samples, device_wires=dev.wires)
        @qml.qnode(dev)
        def circuit(theta: float):
            qml.RX(theta, 0)
            qml.RX(theta / 2, 1)
            return input_measurement()

        mlir = qml.qjit(circuit, target="mlir").mlir
        assert "expval" not in mlir
        assert "sample" in mlir

        theta = 2.5
        res = circuit(theta)

        if len(dev.shots.shot_vector) != 1:
            assert len(res) == len(dev.shots.shot_vector)

        assert np.allclose(res, expected_res(theta), atol=0.05)

    # pylint: disable=unnecessary-lambda
    @pytest.mark.parametrize(
        "input_measurement, expected_res",
        [
            (
                lambda: qml.expval(qml.PauliY(wires=0) @ qml.PauliY(wires=1)),
                lambda theta: np.sin(theta) * np.sin(theta / 2),
            ),
            (lambda: qml.var(qml.Y(wires=1)), lambda theta: 1 - np.sin(theta / 2) ** 2),
            (
                lambda: qml.probs(),
                lambda theta: np.outer(
                    np.outer(
                        [np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2],
                        [np.cos(theta / 4) ** 2, np.sin(theta / 4) ** 2],
                    ),
                    [1, 0, 0, 0],
                ).flatten(),
            ),
            (
                lambda: qml.probs(wires=[1]),
                lambda theta: [np.cos(theta / 4) ** 2, np.sin(theta / 4) ** 2],
            ),
        ],
    )
    def test_measurement_from_counts_single_measurement_analytic(
        self, input_measurement, expected_res
    ):
        """Test the measurment_from_counts transform with a single measurements as part of the
        Catalyst pipeline, for measurements whose outcome can be directly compared to an expected
        analytic result."""

        dev = qml.device("lightning.qubit", wires=4, shots=3000)

        @qml.qjit
        @partial(measurements_from_counts, device_wires=dev.wires)
        @qml.qnode(dev)
        def circuit(theta: float):
            qml.RX(theta, 0)
            qml.RX(theta / 2, 1)
            return input_measurement()

        mlir = qml.qjit(circuit, target="mlir").mlir
        assert "expval" not in mlir
        assert "counts" in mlir

        theta = 2.5
        res = circuit(theta)

        if len(dev.shots.shot_vector) != 1:
            assert len(res) == len(dev.shots.shot_vector)

        assert np.allclose(res, expected_res(theta), atol=0.05)

    def test_measurement_from_counts_raises_not_implemented(self):
        """Test that an measurement not supported by the measurements_from_counts or
        measurements_from_samples transform raises a NotImplementedError"""

        dev = qml.device("lightning.qubit", wires=4, shots=1000)

        @partial(measurements_from_counts, device_wires=dev.wires)
        @qml.qnode(dev)
        def circuit(theta: float):
            qml.RX(theta, 0)
            return qml.sample()

        with pytest.raises(
            NotImplementedError, match="not implemented with measurements_from_counts"
        ):
            qml.qjit(circuit)

    def test_measurement_from_samples_raises_not_implemented(self):
        """Test that an measurement not supported by the measurements_from_counts or
        measurements_from_samples transform raises a NotImplementedError"""

        dev = qml.device("lightning.qubit", wires=4, shots=1000)

        @partial(measurements_from_samples, device_wires=dev.wires)
        @qml.qnode(dev)
        def circuit(theta: float):
            qml.RX(theta, 0)
            return qml.counts()

        with pytest.raises(
            NotImplementedError, match="not implemented with measurements_from_samples"
        ):
            qml.qjit(circuit)

    @pytest.mark.parametrize(
        "unsupported_obs",
        [
            ("PauliX",),
            ("PauliY",),
            ("Hadamard",),
            ("PauliX", "PauliY"),
            ("PauliX", "Hadamard"),
            ("PauliY", "Hadamard"),
            ("PauliX", "PauliY", "Hadamard"),
        ],
    )
    def test_diagonalize_measurements_added_to_transforms(self, unsupported_obs, mocker):
        """Test that the diagonalize_measurements transform is included in the TransformProgram
        as expected when we are not diagonalizing everything to counts or samples, but some of
        {X, Y, Z, H} are not supported."""

        dev = qml.device("lightning.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(theta: float):
            qml.RX(theta, 0)
            qml.RY(0.89, 1)
            return qml.expval(qml.X(0)), qml.var(qml.Y(1)), qml.expval(qml.Hadamard(2))

        expected_result = circuit(1.2)

        config = get_device_toml_config(dev)
        for obs in unsupported_obs:
            del config["operators"]["observables"][obs]

        spy = mocker.spy(QJITDevice, "preprocess")

        # mock TOML file output to indicate some observables are not supported
        with patch("catalyst.device.qjit_device.get_device_toml_config", Mock(return_value=config)):
            jitted_circuit = qml.qjit(circuit)

            transform_program, _ = spy.spy_return
            assert split_non_commuting in transform_program
            assert qml.transforms.diagonalize_measurements in transform_program

            assert len(jitted_circuit(1.2)) == len(expected_result) == 3
            assert np.allclose(jitted_circuit(1.2), expected_result)

    @pytest.mark.parametrize(
        "unsupported_obs",
        [
            ("PauliX",),
            ("PauliY",),
            ("Hadamard",),
            ("PauliX", "PauliY"),
            ("PauliX", "Hadamard"),
            ("PauliY", "Hadamard"),
            ("PauliX", "PauliY", "Hadamard"),
        ],
    )
    def test_diagonalize_measurements_applied_to_mlir(self, unsupported_obs, mocker):
        """Test that the diagonalize_measurements transform is applied or not as when
        we are not diagonalizing everything to counts or samples, but not all of
        {X, Y, Z, H} are supported."""

        dev = qml.device("lightning.qubit", wires=3)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.X(0)), qml.var(qml.Y(1)), qml.expval(qml.Hadamard(2))

        mlir = qml.qjit(circuit, target="mlir").mlir
        for obs in unsupported_obs:
            assert f"{obs}] : !quantum.obs" in mlir

        config = get_device_toml_config(dev)
        for obs in unsupported_obs:
            del config["operators"]["observables"][obs]

        # mock TOML file output to indicate some observables are not supported
        with patch("catalyst.device.qjit_device.get_device_toml_config", Mock(return_value=config)):
            mlir = qml.qjit(circuit, target="mlir").mlir

            for obs in unsupported_obs:
                assert f"{obs}] : !quantum.obs" not in mlir

    @pytest.mark.parametrize("non_commuting_flag", (True, False))
    def test_split_non_commuting_is_added_for_partial_diagonalization(
        self, non_commuting_flag, mocker
    ):
        """Test that the split_non_commuting transform is added to the transform program from
        preprocess when diagonalizing some observables, regarless of the non_commuting_observables
        flag"""

        dev = qml.device("lightning.qubit", wires=4, shots=1000)

        config = get_device_toml_config(dev)

        del config["operators"]["observables"]["Hadamard"]
        config["compilation"]["non_commuting_observables"] = non_commuting_flag

        with patch("catalyst.device.qjit_device.get_device_toml_config", Mock(return_value=config)):
            qjit_dev = QJITDevice(dev)

        # dev1 supports non-commuting observables and sum observables - no splitting
        assert qjit_dev.capabilities.non_commuting_observables_flag is non_commuting_flag

        # Check the preprocess
        with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
            transform_program, _ = qjit_dev.preprocess(ctx)

        assert split_non_commuting in transform_program

    @pytest.mark.parametrize("non_commuting_flag", (True, False))
    def test_split_non_commuting_is_added_for_full_diagonalization(
        self, non_commuting_flag, mocker
    ):
        """Test that the split_non_commuting transform is added to the transform program from
        preprocess when diagonalizing all observables, regarless of the non_commuting_observables
        flag"""

        dev = qml.device("lightning.qubit", wires=4, shots=1000)

        config = get_device_toml_config(dev)

        config["operators"]["observables"] = {}
        config["compilation"]["non_commuting_observables"] = non_commuting_flag

        with patch("catalyst.device.qjit_device.get_device_toml_config", Mock(return_value=config)):
            qjit_dev = QJITDevice(dev)

        # dev1 supports non-commuting observables and sum observables - no splitting
        assert qjit_dev.capabilities.non_commuting_observables_flag is non_commuting_flag

        # Check the preprocess
        with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
            transform_program, _ = qjit_dev.preprocess(ctx)

        assert split_non_commuting in transform_program

    def test_measurements_are_split(self, mocker):
        """Test that the split_to_single_terms or split_non_commuting transform
        are added to the transform program from preprocess as expected, based on the
        sum_observables_flag and the non_commuting_observables_flag"""

        dev = CustomDevice(wires=4, shots=1000)

        # dev1 supports non-commuting observables and sum observables - no splitting
        qjit_dev1 = QJITDevice(dev)
        assert "Sum" in qjit_dev1.capabilities.native_obs
        assert "Hamiltonian" in qjit_dev1.capabilities.native_obs
        assert qjit_dev1.capabilities.non_commuting_observables_flag is True

        # dev2 supports non-commuting observables but NOT sums - split_to_single_terms
        qjit_dev2 = QJITDevice(dev)
        del qjit_dev2.capabilities.native_obs["Sum"]
        del qjit_dev2.capabilities.native_obs["Hamiltonian"]

        # dev3 supports does not support non-commuting observables OR sums - split_non_commuting
        qjit_dev3 = QJITDevice(dev)
        del qjit_dev3.capabilities.native_obs["Sum"]
        del qjit_dev3.capabilities.native_obs["Hamiltonian"]
        qjit_dev3.capabilities.non_commuting_observables_flag = False

        # dev4 supports sums but NOT non-commuting observables - split_non_commuting
        qjit_dev4 = QJITDevice(dev)
        qjit_dev4.capabilities.non_commuting_observables_flag = False

        # Check the preprocess
        with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
            transform_program1, _ = qjit_dev1.preprocess(ctx)  # no splitting
            transform_program2, _ = qjit_dev2.preprocess(ctx)  # split_to_single_terms
            transform_program3, _ = qjit_dev3.preprocess(ctx)  # split_non_commuting
            transform_program4, _ = qjit_dev4.preprocess(ctx)  # split_non_commuting

        assert split_to_single_terms not in transform_program1
        assert split_non_commuting not in transform_program1

        assert split_to_single_terms in transform_program2
        assert split_non_commuting not in transform_program2

        assert split_non_commuting in transform_program3
        assert split_to_single_terms not in transform_program3

        assert split_non_commuting in transform_program4
        assert split_to_single_terms not in transform_program4

    @pytest.mark.parametrize(
        "observables",
        [
            (qml.X(0) @ qml.X(1), qml.Y(0)),  # distributed to separate tapes, but no sum splitting
            (qml.X(0) + qml.X(1), qml.Y(0)),  # split into 3 seperate terms and distributed
        ],
    )
    def test_split_non_commuting_execution(self, observables, mocker):
        """Test that the results of the execution for a tape with non-commuting observables is
        consistent (on a backend that does, in fact, support non-commuting observables) regardless
        of whether split_non_commuting is applied or not as expected"""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def unjitted_circuit(theta: float):
            qml.RX(theta, 0)
            qml.RY(0.89, 1)
            return [qml.expval(o) for o in observables]

        expected_result = unjitted_circuit(1.2)

        config = get_device_toml_config(dev)
        spy = mocker.spy(QJITDevice, "preprocess")

        # mock TOML file output to indicate non-commuting observables are supported
        config["compilation"]["non_commuting_observables"] = True
        with patch("catalyst.device.qjit_device.get_device_toml_config", Mock(return_value=config)):
            jitted_circuit = qml.qjit(unjitted_circuit)
            assert len(jitted_circuit(1.2)) == len(expected_result) == 2
            assert np.allclose(jitted_circuit(1.2), expected_result)

        transform_program, _ = spy.spy_return
        assert split_non_commuting not in transform_program

        # mock TOML file output to indicate non-commuting observables are NOT supported
        config["compilation"]["non_commuting_observables"] = False
        with patch("catalyst.device.qjit_device.get_device_toml_config", Mock(return_value=config)):
            jitted_circuit = qml.qjit(unjitted_circuit)
            assert len(jitted_circuit(1.2)) == len(expected_result) == 2
            assert np.allclose(jitted_circuit(1.2), unjitted_circuit(1.2))

        transform_program, _ = spy.spy_return
        assert split_non_commuting in transform_program

    def test_split_to_single_terms_execution(self, mocker):
        """Test that the results of the execution for a tape with multi-term observables is
        consistent (on a backend that does, in fact, support multi-term observables) regardless
        of whether split_to_single_terms is applied or not"""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def unjitted_circuit(theta: float):
            qml.RX(theta, 0)
            qml.RY(0.89, 1)
            return qml.expval(qml.X(0) + qml.X(1)), qml.expval(qml.Y(0))

        expected_result = unjitted_circuit(1.2)

        config = get_device_toml_config(dev)
        spy = mocker.spy(QJITDevice, "preprocess")

        # make sure non_commuting_observables_flag is True - otherwise we use
        # split_non_commuting instead of split_to_single_terms
        assert config["compilation"]["non_commuting_observables"] is True
        # make sure the testing device does in fact support sum observables
        assert "Sum" in config["operators"]["observables"]

        # test case where transform should not be applied
        jitted_circuit = qml.qjit(unjitted_circuit)
        assert len(jitted_circuit(1.2)) == len(expected_result) == 2
        assert np.allclose(jitted_circuit(1.2), expected_result)

        transform_program, _ = spy.spy_return
        assert split_to_single_terms not in transform_program

        # mock TOML file output to indicate non-commuting observables are NOT supported
        del config["operators"]["observables"]["Sum"]
        del config["operators"]["observables"]["Hamiltonian"]
        with patch("catalyst.device.qjit_device.get_device_toml_config", Mock(return_value=config)):
            jitted_circuit = qml.qjit(unjitted_circuit)
            assert len(jitted_circuit(1.2)) == len(expected_result) == 2
            assert np.allclose(jitted_circuit(1.2), unjitted_circuit(1.2))

        transform_program, _ = spy.spy_return
        assert split_to_single_terms in transform_program


class TestTransform:
    """Test the measurement transforms implemented in Catalyst."""

    def test_measurements_from_counts(self):
        """Test the transfom measurements_from_counts."""
        device = qml.device("lightning.qubit", wires=4, shots=1000)

        @qml.qjit
        @partial(measurements_from_counts, device_wires=device.wires)
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
        results = res

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
