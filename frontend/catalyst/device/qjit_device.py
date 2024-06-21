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

"""
This module contains device stubs for the old and new PennyLane device API, which facilitate
the application of decomposition and other device pre-processing routines.
"""
import logging
import os
import pathlib
import platform
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Set, Union

import pennylane as qml
from pennylane.measurements import MidMeasureMP
from pennylane.transforms import split_non_commuting
from pennylane.transforms.core import TransformProgram

from catalyst.device.decomposition import (
    catalyst_acceptance,
    catalyst_decompose,
    measurements_from_counts,
)
from catalyst.device.verification import (
    validate_observables,
    validate_observables_adjoint_diff,
    validate_observables_parameter_shift,
    verify_no_state_variance_returns,
    verify_operations,
)
from catalyst.logging import debug_logger, debug_logger_init
from catalyst.utils.exceptions import CompileError
from catalyst.utils.patching import Patcher
from catalyst.utils.runtime_environment import get_lib_path
from catalyst.utils.toml import (
    DeviceCapabilities,
    OperationProperties,
    ProgramFeatures,
    TOMLDocument,
    intersect_operations,
    load_device_capabilities,
    pennylane_operation_set,
    read_toml_file,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

RUNTIME_OPERATIONS = [
    "CNOT",
    "ControlledPhaseShift",
    "CRot",
    "CRX",
    "CRY",
    "CRZ",
    "CSWAP",
    "CY",
    "CZ",
    "Hadamard",
    "Identity",
    "IsingXX",
    "IsingXY",
    "IsingYY",
    "IsingZZ",
    "ISWAP",
    "MultiRZ",
    "PauliX",
    "PauliY",
    "PauliZ",
    "PhaseShift",
    "PSWAP",
    "QubitUnitary",
    "Rot",
    "RX",
    "RY",
    "RZ",
    "S",
    "SWAP",
    "T",
    "Toffoli",
    "GlobalPhase",
]

RUNTIME_OBSERVABLES = [
    "Identity",
    "PauliX",
    "PauliY",
    "PauliZ",
    "Hadamard",
    "Hermitian",
    "Hamiltonian",
    "LinearCombination",
    "Prod",
    "SProd",
    "Sum",
    "Tensor",
]

# The runtime interface does not care about specific gate properties, so set them all to True.
RUNTIME_OPERATIONS = {
    op: OperationProperties(invertible=True, controllable=True, differentiable=True)
    for op in RUNTIME_OPERATIONS
}

RUNTIME_OBSERVABLES = {
    obs: OperationProperties(invertible=True, controllable=True, differentiable=True)
    for obs in RUNTIME_OBSERVABLES
}

# TODO: This should be removed after implementing `get_c_interface`
# for the following backend devices:
SUPPORTED_RT_DEVICES = {
    "lightning.qubit": ("LightningSimulator", "librtd_lightning"),
    "lightning.kokkos": ("LightningKokkosSimulator", "librtd_lightning"),
    "braket.aws.qubit": ("OpenQasmDevice", "librtd_openqasm"),
    "braket.local.qubit": ("OpenQasmDevice", "librtd_openqasm"),
}


def get_device_shots(dev):
    """Helper function to get device shots."""
    return dev.shots if isinstance(dev, qml.devices.LegacyDevice) else dev.shots.total_shots


@dataclass
class BackendInfo:
    """Backend information"""

    device_name: str
    c_interface_name: str
    lpath: str
    kwargs: Dict[str, Any]


# pylint: disable=too-many-branches
@debug_logger
def extract_backend_info(device: qml.QubitDevice, capabilities: DeviceCapabilities) -> BackendInfo:
    """Extract the backend info from a quantum device. The device is expected to carry a reference
    to a valid TOML config file."""

    dname = device.name
    if isinstance(device, qml.devices.LegacyDevice):
        dname = device.short_name

    device_name = ""
    device_lpath = ""
    device_kwargs = {}

    if dname in SUPPORTED_RT_DEVICES:
        # Support backend devices without `get_c_interface`
        device_name = SUPPORTED_RT_DEVICES[dname][0]
        device_lpath = get_lib_path("runtime", "RUNTIME_LIB_DIR")
        sys_platform = platform.system()

        if sys_platform == "Linux":
            device_lpath = os.path.join(device_lpath, SUPPORTED_RT_DEVICES[dname][1] + ".so")
        elif sys_platform == "Darwin":  # pragma: no cover
            device_lpath = os.path.join(device_lpath, SUPPORTED_RT_DEVICES[dname][1] + ".dylib")
        else:  # pragma: no cover
            raise NotImplementedError(f"Platform not supported: {sys_platform}")
    elif hasattr(device, "get_c_interface"):
        # Support third party devices with `get_c_interface`
        device_name, device_lpath = device.get_c_interface()
    else:
        raise CompileError(f"The {dname} device does not provide C interface for compilation.")

    if not pathlib.Path(device_lpath).is_file():
        raise CompileError(f"Device at {device_lpath} cannot be found!")

    if hasattr(device, "shots"):
        shots = get_device_shots(device) or 0
        device_kwargs["shots"] = shots

    if dname == "braket.local.qubit":  # pragma: no cover
        device_kwargs["device_type"] = dname
        device_kwargs["backend"] = (
            # pylint: disable=protected-access
            device._device._delegate.DEVICE_ID
        )
    elif dname == "braket.aws.qubit":  # pragma: no cover
        device_kwargs["device_type"] = dname
        device_kwargs["device_arn"] = device._device._arn  # pylint: disable=protected-access
        if device._s3_folder:  # pylint: disable=protected-access
            device_kwargs["s3_destination_folder"] = str(
                device._s3_folder  # pylint: disable=protected-access
            )

    for k, v in capabilities.options.items():
        if hasattr(device, v):
            device_kwargs[k] = getattr(device, v)

    return BackendInfo(dname, device_name, device_lpath, device_kwargs)


@debug_logger
def get_qjit_device_capabilities(target_capabilities: DeviceCapabilities) -> Set[str]:
    """Calculate the set of supported quantum gates for the QJIT device from the gates
    allowed on the target quantum device."""
    # Supported gates of the target PennyLane's device
    qjit_capabilities = deepcopy(target_capabilities)

    # Gates and observables that Catalyst runtime supports
    qir_gates = RUNTIME_OPERATIONS
    qir_observables = RUNTIME_OBSERVABLES

    # Intersection of the above
    qjit_capabilities.native_ops = intersect_operations(target_capabilities.native_ops, qir_gates)
    qjit_capabilities.native_obs = intersect_operations(
        target_capabilities.native_obs, qir_observables
    )

    # Control-flow gates to be lowered down to the LLVM control-flow instructions
    qjit_capabilities.native_ops.update(
        {
            "Cond": OperationProperties(invertible=True, controllable=True, differentiable=True),
            "WhileLoop": OperationProperties(
                invertible=True, controllable=True, differentiable=True
            ),
            "ForLoop": OperationProperties(invertible=True, controllable=True, differentiable=True),
        }
    )

    # Optionally enable runtime-powered mid-circuit measurments
    if target_capabilities.mid_circuit_measurement_flag:  # pragma: no branch
        qjit_capabilities.native_ops.update(
            {
                "MidCircuitMeasure": OperationProperties(
                    invertible=False, controllable=False, differentiable=False
                )
            }
        )

    # Optionally enable runtime-powered quantum gate adjointing (inversions)
    if any(ng.invertible for ng in target_capabilities.native_ops.values()):
        qjit_capabilities.native_ops.update(
            {
                "HybridAdjoint": OperationProperties(
                    invertible=True, controllable=True, differentiable=True
                )
            }
        )

    # TODO: Optionally enable runtime-powered quantum gate controlling once they
    #       are supported natively in MLIR.
    # if any(ng.controllable for ng in target_capabilities.native_ops.values()):
    #     qjit_capabilities.native_ops.update(
    #         {
    #             "HybridCtrl": OperationProperties(
    #                 invertible=True, controllable=True, differentiable=True
    #             )
    #
    #     )

    return qjit_capabilities


class QJITDevice(qml.QubitDevice):
    """QJIT device.

    A device that interfaces the compilation pipeline of Pennylane programs.

    Args:
        wires (int): the number of wires to initialize the device with
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically
        backend_name (str): name of the device from the list of supported and compiled backend
            devices by the runtime
        backend_kwargs (Dict(str, AnyType)): An optional dictionary of the device specifications
    """

    name = "QJIT device"
    short_name = "qjit.device"
    pennylane_requires = "0.1.0"
    version = "0.0.1"
    author = ""

    @staticmethod
    def _get_operations_to_convert_to_matrix(_capabilities: DeviceCapabilities) -> Set[str]:
        # We currently override and only set a few gates to preserve existing behaviour.
        # We could choose to read from config and use the "matrix" gates.
        # However, that affects differentiability.
        # None of the "matrix" gates with more than 2 qubits parameters are differentiable.
        # TODO: https://github.com/PennyLaneAI/catalyst/issues/398
        return {"MultiControlledX", "BlockEncode"}

    @debug_logger_init
    def __init__(
        self,
        original_device,
        original_device_capabilities: DeviceCapabilities,
        backend: Optional[BackendInfo] = None,
    ):
        self.original_device = original_device
        super().__init__(wires=original_device.wires, shots=original_device.shots)

        check_device_wires(self.wires)

        self.backend_name = backend.c_interface_name if backend else "default"
        self.backend_lib = backend.lpath if backend else ""
        self.backend_kwargs = backend.kwargs if backend else {}

        self.qjit_capabilities = get_qjit_device_capabilities(original_device_capabilities)

    @property
    def operations(self) -> Set[str]:
        """Get the device operations using PennyLane's syntax"""
        return pennylane_operation_set(self.qjit_capabilities.native_ops)

    @property
    def observables(self) -> Set[str]:
        """Get the device observables"""
        return pennylane_operation_set(self.qjit_capabilities.native_obs)

    def apply(self, operations, **kwargs):
        """
        Raises: RuntimeError
        """
        raise RuntimeError("QJIT devices cannot apply operations.")  # pragma: no cover

    @debug_logger
    def default_expand_fn(self, circuit, max_expansion=10):
        """
        Most decomposition logic will be equivalent to PennyLane's decomposition.
        However, decomposition logic will differ in the following cases:

        1. All unsupported :class:`qml.Controlled <pennylane.ops.op_math.Controlled>` instances
            will decompose to :class:`qml.QubitUnitary <pennylane.QubitUnitary>` operations.
        2. The list of device-supported gates employed by Catalyst is currently different than
            that of the ``lightning.qubit`` device, as defined by the
            :class:`~.qjit_device.QJITDevice`.

        Args:
            circuit: circuit to expand
            max_expansion: the maximum number of expansion steps if no fixed-point is reached.
        """
        # Ensure catalyst.measure is used instead of qml.measure.
        if any(isinstance(op, MidMeasureMP) for op in circuit.operations):
            raise CompileError("Must use 'measure' from Catalyst instead of PennyLane.")

        decompose_to_qubit_unitary = QJITDevice._get_operations_to_convert_to_matrix(
            self.qjit_capabilities
        )

        def _decomp_to_unitary(self, *_args, **_kwargs):
            try:
                mat = self.matrix()
            except Exception as e:
                raise CompileError(
                    f"Operation {self} could not be decomposed, it might be unsupported."
                ) from e
            return [qml.QubitUnitary(mat, wires=self.wires)]

        # Fallback for controlled gates that won't decompose successfully.
        # Doing so before rather than after decomposition is generally a trade-off. For low
        # numbers of qubits, a unitary gate might be faster, while for large qubit numbers prior
        # decomposition is generally faster.
        # At the moment, bypassing decomposition for controlled gates will generally have a higher
        # success rate, as complex decomposition paths can fail to trace (c.f. PL #3521, #3522).
        overriden_methods = [  # pragma: no cover
            (qml.ops.Controlled, "has_decomposition", lambda self: True),
            (qml.ops.Controlled, "decomposition", _decomp_to_unitary),
        ]
        for gate in decompose_to_qubit_unitary:
            overriden_methods.append((getattr(qml, gate), "decomposition", _decomp_to_unitary))

        with Patcher(*overriden_methods):
            expanded_tape = super().default_expand_fn(circuit, max_expansion)

        self.check_validity(expanded_tape.operations, [])
        return expanded_tape


class QJITDeviceNewAPI(qml.devices.Device):
    """QJIT device for the new device API.
    A device that interfaces the compilation pipeline of Pennylane programs.
    Args:
        wires (Shots): the number of wires to initialize the device with
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified. Setting
            to ``None`` results in computing statistics like expectation values and
            variances analytically
        backend_name (str): name of the device from the list of supported and compiled backend
            devices by the runtime
        backend_kwargs (Dict(str, AnyType)): An optional dictionary of the device specifications
    """

    @debug_logger_init
    def __init__(
        self,
        original_device,
        original_device_capabilities: DeviceCapabilities,
        backend: Optional[BackendInfo] = None,
    ):
        self.original_device = original_device

        for key, value in original_device.__dict__.items():
            self.__setattr__(key, value)

        check_device_wires(original_device.wires)

        super().__init__(wires=original_device.wires, shots=original_device.shots)

        self.backend_name = backend.c_interface_name if backend else "default"
        self.backend_lib = backend.lpath if backend else ""
        self.backend_kwargs = backend.kwargs if backend else {}

        self.qjit_capabilities = get_qjit_device_capabilities(original_device_capabilities)

    @property
    def operations(self) -> Set[str]:
        """Get the device operations"""
        return pennylane_operation_set(self.qjit_capabilities.native_ops)

    @property
    def observables(self) -> Set[str]:
        """Get the device observables"""
        return pennylane_operation_set(self.qjit_capabilities.native_obs)

    @property
    def measurement_processes(self) -> Set[str]:
        """Get the device measurement processes"""
        return self.qjit_capabilities.measurement_processes

    @debug_logger
    def preprocess(
        self,
        ctx,
        execution_config: qml.devices.ExecutionConfig = qml.devices.DefaultExecutionConfig,
    ):
        """This function defines the device transform program to be applied and an updated device
        configuration. The transform program will be created and applied to the tape before
        compilation, in order to modify the operations and measurements to meet device
        specifications from the TOML file.

        The final transforms verify that the resulting tape is supported.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure
                describing parameters of the execution.

        Returns:
            TransformProgram: A transform program that when called returns QuantumTapes that can be
                compiled for the backend, and a postprocessing function to be called on the results
            ExecutionConfig: configuration with unset specifications filled in if relevant.

        This device supports operations and measurements based on the device ``capabilities``,
        which are created based on what is both compatible with Catalyst and what is supported the
        backend according to the backend TOML file).
        """

        _, config = self.original_device.preprocess(execution_config)
        program = TransformProgram()

        # measurement transforms (these may change operations on the tape to accommodate
        # measurement transformations, so must occur before decomposition of measurements)
        if self.qjit_capabilities.non_commuting_observables_flag is False:
            program.add_transform(split_non_commuting)
        if self.measurement_processes == {"Counts"}:
            program.add_transform(measurements_from_counts)

        # decomposition to supported ops/measurements
        ops_acceptance = partial(catalyst_acceptance, operations=self.operations)
        program.add_transform(
            catalyst_decompose,
            ctx=ctx,
            stopping_condition=ops_acceptance,
            capabilities=self.qjit_capabilities,
        )

        # Catalyst program verification and validation
        program.add_transform(
            verify_operations, grad_method=config.gradient_method, qjit_device=self
        )
        program.add_transform(
            validate_observables, self.qjit_capabilities, self.original_device.name
        )

        if config.gradient_method is not None:
            program.add_transform(verify_no_state_variance_returns)
        if config.gradient_method == "adjoint":
            program.add_transform(validate_observables_adjoint_diff, qjit_device=self)
        elif config.gradient_method == "parameter-shift":
            program.add_transform(validate_observables_parameter_shift)

        return program, config

    def execute(self, circuits, execution_config):
        """
        Raises: RuntimeError
        """
        raise RuntimeError("QJIT devices cannot execute tapes.")


# Alias for either an old-style or a new-style QJITDevice
AnyQJITDevice = Union[QJITDevice, QJITDeviceNewAPI]


def filter_out_modifiers(operations):
    """Remove Adjoint/Control from operations.

    Args:
        operations (Iterable[str]): list of operation names

    Returns:
        Set: filtered set of operation names
    """
    pattern = re.compile(r"^(C|Adjoint)\(.*\)$")

    def is_not_modifier(op):
        return not re.match(pattern, op)

    return set(filter(is_not_modifier, operations))


@debug_logger
def validate_device_capabilities(
    device: qml.QubitDevice, device_capabilities: DeviceCapabilities
) -> None:
    """Validate configuration document against the device attributes.
    Raise CompileError in case of mismatch:
    * If device is not qjit-compatible.
    * If configuration file does not exists.
    * If decomposable, matrix, and native gates have some overlap.
    * If decomposable, matrix, and native gates do not match gates in ``device.operations`` and
      ``device.observables``.

    Args:
        device (qml.devices.LegacyDevice): An instance of a quantum device.
        config (TOMLDocument): A TOML document representation.

    Raises: CompileError
    """

    if not device_capabilities.qjit_compatible_flag:
        raise CompileError(
            f"Attempting to compile program for incompatible device '{device.name}': "
            f"Config is not marked as qjit-compatible"
        )

    device_name = device.short_name if isinstance(device, qml.devices.LegacyDevice) else device.name

    native = set(device_capabilities.native_ops.keys())
    decomposable = set(device_capabilities.to_decomp_ops.keys())
    matrix = set(device_capabilities.to_matrix_ops.keys())

    overlap = (native & decomposable) | (native & matrix) | (decomposable & matrix)
    if overlap:
        raise CompileError(f"Device '{device_name}' has overlapping gates: {overlap}")

    if hasattr(device, "operations") and hasattr(device, "observables"):
        # For validation against PL device properties we only check base gates as Adjoint/Control
        # declarations can be very sporadic.
        device_gates = filter_out_modifiers(device.operations)
        spec_gates = native | decomposable | matrix
        if device_gates != spec_gates:
            raise CompileError(
                "Gates in qml.device.operations and specification file do not match for "
                f'"{device_name}".\n'
                f"Gates that present only in the device: {device_gates - spec_gates}\n"
                f"Gates that present only in spec: {spec_gates - device_gates}\n"
            )

        # For observables, we do not have `non-native` section in the config, so we check that
        # device data supercedes the specification.
        device_observables = set(device.observables)
        spec_observables = pennylane_operation_set(device_capabilities.native_obs)
        if (spec_observables - device_observables) != set():
            raise CompileError(
                "Observables in qml.device.observables and specification file do not match.\n"
                f"Observables that present only in spec: {spec_observables - device_observables}\n"
            )


def get_device_toml_config(device) -> TOMLDocument:
    """Get the contents of the device config file."""
    if hasattr(device, "config"):
        # The expected case: device specifies its own config.
        toml_file = device.config
    else:
        # TODO: Remove this section when `qml.Device`s are guaranteed to have their own config file
        # field.
        device_lpath = pathlib.Path(get_lib_path("runtime", "RUNTIME_LIB_DIR"))

        name = device.short_name if isinstance(device, qml.devices.LegacyDevice) else device.name
        # The toml files name convention we follow is to replace
        # the dots with underscores in the device short name.
        toml_file_name = name.replace(".", "_") + ".toml"
        # And they are currently saved in the following directory.
        toml_file = device_lpath.parent / "lib" / "backend" / toml_file_name

    try:
        config = read_toml_file(toml_file)
    except FileNotFoundError as e:
        raise CompileError(
            "Attempting to compile program for incompatible device: "
            f"Config file ({toml_file}) does not exist"
        ) from e

    return config


def get_device_capabilities(
    device, program_features: Optional[ProgramFeatures] = None
) -> DeviceCapabilities:
    """Get or load DeviceCapabilities structure from device"""
    if hasattr(device, "qjit_capabilities"):
        return device.qjit_capabilities
    else:
        program_features = (
            program_features if program_features else ProgramFeatures(device.shots is not None)
        )
        device_name = (
            device.short_name if isinstance(device, qml.devices.LegacyDevice) else device.name
        )
        device_config = get_device_toml_config(device)
        return load_device_capabilities(device_config, program_features, device_name)


def check_device_wires(wires):
    """Validate requirements Catalyst imposes on device wires."""
    if wires is None:
        raise AttributeError("Catalyst does not support device instances without set wires.")

    assert isinstance(wires, qml.wires.Wires)

    if not all(isinstance(wire, int) for wire in wires.labels):
        raise AttributeError("Catalyst requires continuous integer wire labels starting at 0.")

    if not wires.labels == tuple(range(len(wires))):
        raise AttributeError("Catalyst requires continuous integer wire labels starting at 0.")
