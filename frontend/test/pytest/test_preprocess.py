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
from dataclasses import replace
from functools import partial
from os.path import join
from tempfile import TemporaryDirectory
from textwrap import dedent
from unittest.mock import Mock, patch

import numpy as np
import pennylane as qml
import pytest
from flaky import flaky
from pennylane.devices import Device
from pennylane.devices.execution_config import DefaultExecutionConfig, ExecutionConfig
from pennylane.measurements import CountsMP, SampleMP
from pennylane.tape import QuantumScript
from pennylane.transforms import split_non_commuting, split_to_single_terms
from pennylane.transforms.core import TransformProgram

from catalyst import CompileError, ctrl
from catalyst.api_extensions.control_flow import (
    Cond,
    ForLoop,
    WhileLoop,
    cond,
    for_loop,
    while_loop,
)
from catalyst.api_extensions.quantum_operators import HybridAdjoint, adjoint
from catalyst.compiler import get_lib_path
from catalyst.device import (
    QJITDeviceNewAPI,
    extract_backend_info,
    get_device_capabilities,
    get_device_toml_config,
)
from catalyst.device.decomposition import (
    catalyst_acceptance,
    catalyst_decompose,
    decompose_ops_to_unitary,
    measurements_from_counts,
    measurements_from_samples,
)
from catalyst.jax_tracer import HybridOpRegion
from catalyst.tracing.contexts import EvaluationContext, EvaluationMode
from catalyst.utils.toml import (
    DeviceCapabilities,
    OperationProperties,
    ProgramFeatures,
    TOMLDocument,
    load_device_capabilities,
    pennylane_operation_set,
    read_toml_file,
)


def get_test_config(config_text: str) -> TOMLDocument:
    """Parse test config into the TOMLDocument structure"""
    with TemporaryDirectory() as d:
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(config_text)
        config = read_toml_file(toml_file)
        return config


def get_test_device_capabilities(
    program_features: ProgramFeatures, config_text: str
) -> DeviceCapabilities:
    """Parse test config into the DeviceCapabilities structure"""
    config = get_test_config(config_text)
    device_capabilities = load_device_capabilities(config, program_features)
    return device_capabilities


class DummyDevice(Device):
    """A dummy device from the device API."""

    config = get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/backend/dummy_device.toml"

    def __init__(self, wires, shots=1024):
        print(pathlib.Path(__file__).parent.parent.parent.parent)
        super().__init__(wires=wires, shots=shots)
        program_features = ProgramFeatures(bool(shots))
        dummy_capabilities = get_device_capabilities(self, program_features)
        dummy_capabilities.native_ops.pop("BlockEncode")
        dummy_capabilities.to_matrix_ops["BlockEncode"] = OperationProperties(False, False, False)
        self.qjit_capabilities = dummy_capabilities

    @staticmethod
    def get_c_interface():
        """Returns a tuple consisting of the device name, and
        the location to the shared object with the C/C++ device implementation.
        """
        system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
        lib_path = get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_dummy" + system_extension
        return "dummy.remote", lib_path

    def execute(self, circuits, execution_config):
        """Execution."""
        return circuits, execution_config

    def preprocess(self, execution_config: ExecutionConfig = DefaultExecutionConfig):
        """Preprocessing."""
        transform_program = TransformProgram()
        transform_program.add_transform(split_non_commuting)
        return transform_program, execution_config


class DummyDeviceLimitedMPs(Device):
    """A dummy device from the device API without wires."""

    config = get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/backend/dummy_device.toml"

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
        lib_path = get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_dummy" + system_extension
        return "dummy.remote", lib_path

    def execute(self, circuits, execution_config):
        """Execution."""
        return circuits, execution_config

    def __enter__(self, *args, **kwargs):
        dummy_toml = self.config
        with open(dummy_toml, mode="r", encoding="UTF-8") as f:
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


class OtherHadamard(qml.Hadamard):
    """A version of the Hadamard operator that won't be recognized by the QJit device, and will
    need to be decomposed"""

    @property
    def name(self):
        """name"""
        return "OtherHadamard"


class OtherIsingXX(qml.IsingXX):
    """A version of the IsingXX operator that won't be recognized by the QJit device, and will
    need to be decomposed"""

    @property
    def name(self):
        """name"""
        return "OtherIsingXX"


class OtherRX(qml.RX):
    """A version of the RX operator that won't be recognized by the QJit device, and will need to
    be decomposed"""

    @property
    def name(self):
        """Name of the operator is UnknownOp"""
        return "OtherRX"

    def decomposition(self):
        """decomposes to normal RX"""
        return [qml.RX(*self.parameters, self.wires)]


class TestDecomposition:
    """Test the preprocessing transforms implemented in Catalyst."""

    def test_decompose_integration(self):
        """Test the decompose transform as part of the Catalyst pipeline."""
        dev = DummyDevice(wires=4, shots=None)

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

    def test_decompose_ops_to_unitary_integration(self):
        """Test the decompose ops to unitary transform as part of the Catalyst pipeline."""
        dev = DummyDevice(wires=4, shots=None)

        @qml.qjit
        @qml.qnode(dev)
        def circuit():
            qml.BlockEncode(np.array([[1, 1, 1], [0, 1, 0]]), wires=[0, 1, 2])
            return qml.state()

        mlir = qml.qjit(circuit, target="mlir").mlir
        assert "quantum.unitary" in mlir
        assert "BlockEncode" not in mlir

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

        with pytest.raises(CompileError, match="could not be decomposed, it might be unsupported."):
            qml.qjit(f, target="jaxpr")


class TestMeasurementTransforms:

    @flaky
    def test_measurements_from_counts_multiple_measurements(self):
        """Test the transforms for measurements_from_counts to other measurement types
        as part of the Catalyst pipeline."""

        dev = qml.device("lightning.qubit", wires=4, shots=3000)

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
        expval_res, var_res, counts_res, probs_res = qml.qjit(transformed_circuit)(theta)

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
            -1.0: sum([count for count, eigval in zip(counts, eigvals) if eigval == -1]),
            1.0: sum([count for count, eigval in zip(counts, eigvals) if eigval == 1]),
        }

        # +/- 100 shots is pretty reasonable with 3000 shots total
        assert np.isclose(eigval_counts_res[-1], counts_expected[-1], atol=100)
        assert np.isclose(eigval_counts_res[1], counts_expected[1], atol=100)

    def test_measurements_from_samples_multiple_measurements(self):
        """Test the transform measurements_from_samples with multiple measurement types
        as part of the Catalyst pipeline."""

        dev = qml.device("lightning.qubit", wires=4, shots=3000)

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

        expval_res, var_res, sample_res, probs_res = qml.qjit(transformed_circuit)(theta)

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
        res = qml.qjit(measurements_from_counts(circuit, dev.wires))(theta)

        # counts comparison by converting catalyst format to PL style eigvals dict
        basis_states, counts = res

        if measurement().obs:
            num_excitations_per_state = [
                sum(int(i) for i in format(int(state), "01b")) for state in basis_states
            ]
            eigvals = [(-1) ** i for i in num_excitations_per_state]
            eigval_counts_res = {
                -1.0: sum([count for count, eigval in zip(counts, eigvals) if eigval == -1]),
                1.0: sum([count for count, eigval in zip(counts, eigvals) if eigval == 1]),
            }

            # +/- 100 shots is pretty reasonable with 3000 shots total
            assert np.isclose(eigval_counts_res[-1], counts_expected[-1], atol=100)
            assert np.isclose(eigval_counts_res[1], counts_expected[1], atol=100)

        else:
            num_wires = len(measurement().wires) if measurement().wires else len(dev.wires)
            basis_states = [format(int(state), "01b").zfill(num_wires) for state in basis_states]
            counts = [int(c) for c in counts]
            counts_dict = dict([(state, c) for (state, c) in zip(basis_states, counts) if c != 0])

            for res, expected_res in zip(counts_dict.items(), counts_expected.items()):
                assert res[0] == expected_res[0]
                assert np.isclose(res[1], expected_res[1], atol=100)

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
        res = qml.qjit(measurements_from_samples(circuit, dev.wires))(theta)

        if len(measurement().wires) == 1:
            samples_expected = qml.qjit(circuit)(theta)
        else:
            samples_expected = circuit(theta)

        assert res.shape == samples_expected.shape
        assert np.allclose(np.mean(res, axis=0), np.mean(samples_expected, axis=0), atol=0.05)

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

        @qml.qjit
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

    def test_measurements_are_split(self, mocker):
        """Test that the split_to_single_terms or split_non_commuting transform
        are added to the transform program from preprocess as expected, based on the
        sum_observables_flag and the non_commuting_observables_flag"""

        dev = DummyDevice(wires=4, shots=1000)
        dev_capabilities = get_device_capabilities(dev, ProgramFeatures(bool(dev.shots)))

        # dev1 supports non-commuting observables and sum observables - no splitting
        assert "Sum" in dev_capabilities.native_obs
        assert "Hamiltonian" in dev_capabilities.native_obs
        assert dev_capabilities.non_commuting_observables_flag is True
        backend_info = extract_backend_info(dev, dev_capabilities)
        qjit_dev1 = QJITDeviceNewAPI(dev, dev_capabilities, backend_info)

        # dev2 supports non-commuting observables but NOT sums - split_to_single_terms
        del dev_capabilities.native_obs["Sum"]
        del dev_capabilities.native_obs["Hamiltonian"]
        backend_info = extract_backend_info(dev, dev_capabilities)
        qjit_dev2 = QJITDeviceNewAPI(dev, dev_capabilities, backend_info)

        # dev3 supports does not support non-commuting observables OR sums - split_non_commuting
        dev_capabilities = replace(dev_capabilities, non_commuting_observables_flag=False)
        backend_info = extract_backend_info(dev, dev_capabilities)
        qjit_dev3 = QJITDeviceNewAPI(dev, dev_capabilities, backend_info)

        # dev4 supports sums but NOT non-commuting observables - split_non_commuting
        dev_capabilities = replace(dev_capabilities, non_commuting_observables_flag=False)
        backend_info = extract_backend_info(dev, dev_capabilities)
        qjit_dev4 = QJITDeviceNewAPI(dev, dev_capabilities, backend_info)

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
        spy = mocker.spy(QJITDeviceNewAPI, "preprocess")

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
        spy = mocker.spy(QJITDeviceNewAPI, "preprocess")

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


# tapes and regions for generating HybridOps
tape1 = QuantumScript([qml.X(0), qml.Hadamard(1)])
tape2 = QuantumScript([qml.RY(1.23, 1), qml.Y(0), qml.Hadamard(2)])
region1 = HybridOpRegion([], tape1, [], [])
region2 = HybridOpRegion([], tape2, [], [])

# catalyst.pennylane_extensions.Adjoint:
#   Adjoint([], [], regions=[HybridOpRegion([], quantum_tape, [], [])])
adj_op = HybridAdjoint([], [], [region1])

# catalyst.pennylane_extensions.ForLoop:
#     ForLoop([], [], regions=[HybridOpRegion([], quantum_tape, [], [])])
forloop_op = ForLoop([], [], [region1])

# catalyst.pennylane_extensions.WhileLoop:
#     cond_region = HybridOpRegion([], None, [], [])
#     body_region = HybridOpRegion([], quantum_tape, [], [])
#     WhileLoop([], [], regions=[cond_region, body_region])
cond_region = HybridOpRegion([], None, [], [])
whileloop_op = WhileLoop([], [], regions=[cond_region, region1])

# catalyst.pennylane_extensions.Cond:
# Cond([], [], regions=[one Hybrid region per branch of the if-else tree])
cond_op = Cond([], [], regions=[region1, region2])

# each entry contains (initialized_op, op_class, num_regions)
HYBRID_OPS = [
    (adj_op, HybridAdjoint, 1),
    (forloop_op, ForLoop, 1),
    (whileloop_op, WhileLoop, 2),
    (cond_op, Cond, 2),
]

capabilities = get_test_device_capabilities(
    ProgramFeatures(False),
    dedent(
        """
        schema = 2
        [operators.gates.native]
        PauliX = { }
        PauliZ = { }
        RX = { }
        RY = { }
        RZ = { }
        CNOT = { }
        HybridAdjoint = { }
        ForLoop = { }
        WhileLoop = { }
        Cond = { }
        QubitUnitary = { }

        [operators.gates.matrix]
        S = { }
    """
    ),
)

expected_ops = pennylane_operation_set(capabilities.native_ops)


class TestPreprocessHybridOp:
    """Test that the operators on the tapes nested inside HybridOps are also decomposed"""

    @pytest.mark.parametrize("op, op_class, num_regions", HYBRID_OPS)
    def test_hybrid_op_decomposition(self, op, op_class, num_regions):
        """Tests that for a tape containing a HybridOp that contains unsupported
        Operators, the unsupported Operators are decomposed"""

        stopping_condition = partial(catalyst_acceptance, operations=expected_ops)

        # hack for unit test (since it doesn't create a full context)
        for region in op.regions:
            region.trace = None

        # create and decompose the tape
        tape = QuantumScript([op, qml.X(0), qml.Hadamard(3)])
        with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
            (new_tape,), _ = catalyst_decompose(tape, ctx, stopping_condition, capabilities)

        old_op = tape[0]
        new_op = new_tape[0]

        # the pre- and post decomposition HybridOps have the same type and number of regions
        assert isinstance(old_op, op_class)
        assert isinstance(new_op, op_class)
        assert len(new_op.regions) == len(old_op.regions) == num_regions

        # the HybridOp on the original tape is unmodified, i.e. continues to contain ops
        # not in `expected_ops`. The post-decomposition HybridOp tape does not
        for i in range(num_regions):
            if old_op.regions[i].quantum_tape:
                assert not np.all(
                    [op.name in expected_ops for op in old_op.regions[i].quantum_tape.operations]
                )
                assert np.all(
                    [op.name in expected_ops for op in new_op.regions[i].quantum_tape.operations]
                )

    @pytest.mark.parametrize("x, y", [(1.23, -0.4), (0.7, 0.25), (-1.51, 0.6)])
    def test_decomposition_of_adjoint_circuit(self, x, y):
        """Test that unsupported operators nested in Adjoint are decompsed
        and the resulting circuit has the expected result, obtained analytically"""

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(x: float, y: float):
            qml.RY(y, 0)
            adjoint(lambda: OtherRX(x, 0))()
            return qml.expval(qml.PauliZ(0))

        mlir = qml.qjit(circuit, target="mlir").mlir

        assert "quantum.adjoint" in mlir
        assert "RX" in mlir
        assert "RY" in mlir
        assert "OtherRX" not in mlir

        assert np.isclose(circuit(x, y), np.cos(-x) * np.cos(y))

    def test_decomposition_of_cond_circuit(self):
        """Test that unsupported operators nested in Cond are decompsed, and the
        resulting circuit has the expected result, obtained analytically"""

        dev = qml.device("lightning.qubit", wires=[0, 1])

        @qml.qjit
        @qml.qnode(dev)
        def circuit(phi: float):
            OtherHadamard(wires=0)

            # define a conditional ansatz
            @cond(phi > 1.4)
            def ansatz():
                OtherIsingXX(phi, wires=(0, 1))

            @ansatz.otherwise
            def ansatz():
                OtherIsingXX(2 * phi, wires=(0, 1))

            # apply the conditional ansatz
            ansatz()

            return qml.state()

        # mlir contains expected gate names, and not the unsupported gate names
        mlir = qml.qjit(circuit, target="mlir").mlir
        assert "RX" in mlir
        assert "CNOT" in mlir
        assert "PhaseShift" in mlir
        assert "OtherHadamard" not in mlir
        assert "OtherIsingXX" not in mlir

        # results are correct for cond is True (IsingXX angle is phi)
        phi = 1.6
        x1 = np.cos(phi / 2) / np.sqrt(2)
        x2 = -1j * np.sin(phi / 2) / np.sqrt(2)
        expected_res = np.array([x1, x2, x1, x2])
        assert np.allclose(expected_res, circuit(phi))

        # results are correct for cond is False (IsingXX angle is 2*phi)
        phi = 1.2
        x1 = np.cos(phi) / np.sqrt(2)
        x2 = -1j * np.sin(phi) / np.sqrt(2)
        expected_res = np.array([x1, x2, x1, x2])
        assert np.allclose(expected_res, circuit(phi))

    @pytest.mark.parametrize("reps, angle", [(3, 1.72), (5, 1.6), (10, 0.4)])
    def test_decomposition_of_forloop_circuit(self, reps, angle):
        """Test that unsupported operators nested in ForLoop are decompsed, and
        the resulting circuit has the expected result, obtained analytically"""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(n: int, x: float):
            OtherHadamard(wires=0)

            def loop_rx(i, phi):
                OtherIsingXX(phi, wires=(0, 1))
                # update the value of phi for the next iteration
                return phi / 2

            # apply the for loop
            for_loop(0, n, 1)(loop_rx)(x)

            return qml.state()

        def expected_res(n, x):
            """Analytic result for a loop with n reps and initial angle x"""
            phi = x * sum(1 / 2**i for i in range(0, n))

            x1 = np.cos(phi / 2) / np.sqrt(2)
            x2 = -1j * np.sin(phi / 2) / np.sqrt(2)

            return np.array([x1, x2, x1, x2])

        assert np.allclose(circuit(reps, angle), expected_res(reps, angle))

    @pytest.mark.parametrize("phi", [1.1, 1.6, 2.1])
    def test_decomposition_of_whileloop_circuit(self, phi):
        """Test that unsupported operators nested in WhileLoop are decompsed, and
        the resulting circuit has the expected result, obtained analytically"""

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(x: float):
            @while_loop(lambda x: x < 2.0)
            def loop_rx(x):
                # perform some work and update (some of) the arguments
                OtherRX(x, wires=0)
                return x**2

            # apply the while loop
            final_x = loop_rx(x)

            return qml.expval(qml.PauliY(0)), final_x

        res, final_phi = circuit(phi)

        total_angle = 0
        while phi < 2:
            total_angle += phi
            phi = phi**2

        expected_res = -np.sin(total_angle)

        assert np.isclose(res, expected_res)
        assert final_phi > 2.0

    def test_decomposition_of_nested_HybridOp(self):
        """Tests that HybridOps with HybridOps nested inside them are still decomposed correctly"""

        stopping_condition = partial(catalyst_acceptance, operations=expected_ops)

        # make a weird nested op
        adjoint_op = HybridAdjoint([], [], [region1])
        ops = [qml.RY(1.23, 1), adjoint_op, qml.Hadamard(2)]  # Hadamard will decompose
        adj_region = HybridOpRegion([], QuantumScript(ops), [], [])

        conditional_op = Cond([], [], regions=[adj_region, region2])
        ops = [conditional_op, qml.Y(1)]  # PauliY will decompose
        conditional_region = HybridOpRegion([], qml.tape.QuantumScript(ops), [], [])

        for_loop_op = ForLoop([], [], [conditional_region])
        ops = [for_loop_op, qml.X(0), qml.Hadamard(3)]  # Hadamard will decompose
        tape = qml.tape.QuantumScript(ops)

        # hack to avoid needing a full trace in unit test
        adjoint_op.regions[0].trace = None
        for region in conditional_op.regions:
            region.trace = None
        for_loop_op.regions[0].trace = None

        # do the decomposition and get the new tape
        with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
            (new_tape,), _ = catalyst_decompose(tape, ctx, stopping_condition, capabilities)

        # unsupported ops on the top-level tape have been decomposed (no more Hadamard)
        assert "Hadamard" not in [op.name for op in new_tape.operations]
        assert "RZ" in [op.name for op in new_tape.operations]

        # the first element on the top-level tape is a for-loop
        assert isinstance(new_tape[0], ForLoop)
        # unsupported op on it has been decomposed (no more PauliY)
        forloop_subtape = new_tape[0].regions[0].quantum_tape
        assert "PauliY" not in [op.name for op in forloop_subtape.operations]
        assert "RY" in [op.name for op in forloop_subtape.operations]

        # first op on the for-loop tape is a cond op
        assert isinstance(forloop_subtape[0], Cond)
        cond_subtapes = (
            forloop_subtape[0].regions[0].quantum_tape,
            forloop_subtape[0].regions[1].quantum_tape,
        )
        # unsupported ops in the subtape decomposed (original tapes contained Hadamard)
        for subtape in cond_subtapes:
            assert np.all([op.name in expected_ops for op in subtape.operations])
            assert "Hadamard" not in [op.name for op in subtape.operations]
            assert "RZ" in [op.name for op in subtape.operations]

        # the seconds element on the first subtape of the cond op is an adjoint
        assert isinstance(cond_subtapes[0][1], HybridAdjoint)
        # unsupported op on it has been decomposed (no more Hadamard)
        adj_subtape = cond_subtapes[0][1].regions[0].quantum_tape
        assert "Hadamard" not in [op.name for op in adj_subtape.operations]
        assert "RZ" in [op.name for op in adj_subtape.operations]

    def test_controlled_decomposes_to_unitary_listed(self):
        """Test that a PennyLane toml-listed operation is decomposed to a QubitUnitary"""

        stopping_condition = partial(catalyst_acceptance, operations=expected_ops)

        tape = qml.tape.QuantumScript([qml.PauliX(0), qml.S(0)])

        with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
            (new_tape,), _ = catalyst_decompose(tape, ctx, stopping_condition, capabilities)

        assert len(new_tape.operations) == 2
        assert isinstance(new_tape.operations[0], qml.PauliX)
        assert isinstance(new_tape.operations[1], qml.QubitUnitary)

    def test_controlled_decomposes_to_unitary_controlled(self):
        """Test that a PennyLane controlled operation is decomposed to a QubitUnitary"""

        stopping_condition = partial(catalyst_acceptance, operations=expected_ops)

        tape = qml.tape.QuantumScript([qml.ctrl(qml.RX(1.23, 0), 1)])

        with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
            (new_tape,), _ = catalyst_decompose(tape, ctx, stopping_condition, capabilities)

        assert len(new_tape.operations) == 1
        new_op = new_tape.operations[0]

        assert isinstance(new_op, qml.QubitUnitary)
        assert np.allclose(new_op.matrix(), tape.operations[0].matrix())

    def test_error_for_pennylane_midmeasure_decompose(self):
        """Test that an error is raised in decompose if a PennyLane mid-circuit measurement
        is encountered"""

        stopping_condition = partial(catalyst_acceptance, operations=expected_ops)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.23, wires=0)
            qml.measure(0)

        ops, measurements = qml.queuing.process_queue(q)
        tape = qml.tape.QuantumScript(ops, measurements)

        with pytest.raises(
            CompileError, match="Must use 'measure' from Catalyst instead of PennyLane."
        ):
            with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
                _ = catalyst_decompose(tape, ctx, stopping_condition, capabilities)

    def test_error_for_pennylane_midmeasure_decompose_nested(self):
        """Test that an error is raised in decompose if a PennyLane mid-circuit measurement
        is encountered"""

        stopping_condition = partial(catalyst_acceptance, operations=expected_ops)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.23, wires=0)
            qml.measure(0)

        ops, measurements = qml.queuing.process_queue(q)
        subtape = qml.tape.QuantumScript(ops, measurements)

        region = HybridOpRegion([], subtape, [], [])
        region.trace = None
        adjoint_op = HybridAdjoint([], [], [region])

        tape = qml.tape.QuantumScript([adjoint_op, qml.Y(1)], [])

        with pytest.raises(
            CompileError, match="Must use 'measure' from Catalyst instead of PennyLane."
        ):
            with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
                _ = catalyst_decompose(tape, ctx, stopping_condition, capabilities)

    def test_unsupported_op_with_no_decomposition_raises_error(self):
        """Test that an unsupported operator that doesn't provide a decomposition
        raises a CompileError"""

        # operations=[], all ops are unsupported
        stopping_condition = partial(catalyst_acceptance, operations=[])

        tape = qml.tape.QuantumScript([qml.Y(0)])

        with pytest.raises(
            CompileError,
            match="not supported with catalyst on this device and does not provide a decomposition",
        ):
            with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
                _ = catalyst_decompose(tape, ctx, stopping_condition, capabilities)

    def test_decompose_to_matrix_raises_error(self):
        """Test that _decompose_to_matrix raises a CompileError if the operator has no matrix"""

        # operations=[], all ops are unsupported
        stopping_condition = partial(catalyst_acceptance, operations=[])

        class NoMatrixMultiControlledX(qml.MultiControlledX):
            """A version of MulitControlledX with no matrix defined"""

            def matrix(self):
                """raise an error"""
                raise qml.operation.MatrixUndefinedError

        tape = qml.tape.QuantumScript([NoMatrixMultiControlledX(wires=[0, 1, 2, 3])])

        with pytest.raises(CompileError, match="could not be decomposed, it might be unsupported"):
            with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
                _ = catalyst_decompose(tape, ctx, stopping_condition, capabilities)


class TestDiagonalizationTransforms:
    """Test that the diagonalization transforms are included as expected in the QJIT device
    TransformProgram based on device capabilities"""

    @pytest.mark.parametrize(
        "device_measurements, measurement_transform, target_measurement",
        [
            (["counts"], measurements_from_counts, "counts"),
            (["sample"], measurements_from_samples, "sample"),
            (["counts", "sample"], measurements_from_samples, "sample"),
        ],
    )
    def test_measurement_from_readout_integration_multiple_measurements_device(
        self, device_measurements, measurement_transform, target_measurement
    ):
        """Test the measurment_from_samples transform is applied as part of the Catalyst pipeline if the
        device only supports sample, and measurement_from_counts transform is applied  if the device only
        supports counts. If both are supported, sample takes precedence."""

        allow_sample = "sample" in device_measurements
        allow_counts = "counts" in device_measurements

        with DummyDeviceLimitedMPs(
            wires=4, shots=1000, allow_counts=allow_counts, allow_samples=allow_sample
        ) as dev:

            config = get_device_toml_config(dev)
            config["operators"]["observables"] = {}

            with patch("catalyst.device.qjit_device.get_device_toml_config", Mock(return_value=config)):
                # transform is added to transform program
                dev_capabilities = get_device_capabilities(dev, ProgramFeatures(bool(dev.shots)))
                backend_info = extract_backend_info(dev, dev_capabilities)
                qjit_dev = QJITDeviceNewAPI(dev, dev_capabilities, backend_info)

                with EvaluationContext(EvaluationMode.QUANTUM_COMPILATION) as ctx:
                    transform_program, _ = qjit_dev.preprocess(ctx)

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
            with patch("catalyst.device.qjit_device.get_device_toml_config", Mock(return_value=config)):
                mlir = qml.qjit(circuit, target="mlir").mlir

            assert "expval" not in mlir
            assert "quantum.var" not in mlir
            assert "probs" not in mlir
            assert target_measurement in mlir

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
    def test_diagonalize_measurements_integration(self, unsupported_obs, mocker):
        """Test that the diagonalize_measurements transform is applied or not as when
        we are not diagonalizing everything to counts or samples, but not all of
        {X, Y, Z, H} are supported."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def unjitted_circuit(theta: float):
            qml.RX(theta, 0)
            qml.RY(0.89, 1)
            return qml.expval(qml.X(0)), qml.var(qml.Y(1))

        expected_result = unjitted_circuit(1.2)

        config = get_device_toml_config(dev)
        for obs in unsupported_obs:
            del config["operators"]["observables"][obs]

        spy = mocker.spy(QJITDeviceNewAPI, "preprocess")

        # mock TOML file output to indicate some observables are not supported
        with patch("catalyst.device.qjit_device.get_device_toml_config", Mock(return_value=config)):
            jitted_circuit = qml.qjit(unjitted_circuit)
            assert len(jitted_circuit(1.2)) == len(expected_result) == 2
            assert np.allclose(jitted_circuit(1.2), expected_result)

        transform_program, _ = spy.spy_return
        assert split_non_commuting in transform_program
        assert qml.transforms.diagonalize_measurements in transform_program


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


if __name__ == "__main__":
    pytest.main(["-x", __file__])
