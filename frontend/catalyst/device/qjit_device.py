# Copyright 2024-2025 Xanadu Quantum Technologies Inc.

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
import textwrap
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import pennylane as qml
from jax.interpreters.partial_eval import DynamicJaxprTracer
from pennylane import CompilePipeline
from pennylane.devices.capabilities import (
    DeviceCapabilities,
    ExecutionCondition,
    OperatorProperties,
)
from pennylane.transforms import (
    diagonalize_measurements,
    split_non_commuting,
    split_to_single_terms,
)

from catalyst.device.decomposition import (
    catalyst_decompose,
    measurements_from_counts,
    measurements_from_samples,
)
from catalyst.device.verification import (
    validate_measurements,
    validate_observables_adjoint_diff,
    validate_observables_parameter_shift,
    verify_no_state_variance_returns,
    verify_operations,
)
from catalyst.logging import debug_logger, debug_logger_init
from catalyst.third_party.cuda import SoftwareQQPP
from catalyst.utils.exceptions import CompileError
from catalyst.utils.runtime_environment import get_lib_path

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
    "SingleExcitation",
    "DoubleExcitation",
    "ISWAP",
    "MultiRZ",
    "PauliX",
    "PauliY",
    "PauliZ",
    "PCPhase",
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
    "LinearCombination",
    "Prod",
    "SProd",
    "Sum",
]

RUNTIME_MPS = ["ExpectationMP", "SampleMP", "VarianceMP", "CountsMP", "StateMP", "ProbabilityMP"]

# The runtime interface does not care about specific gate properties, so set them all to True.
RUNTIME_OPERATIONS = {
    op: OperatorProperties(invertible=True, controllable=True, differentiable=True)
    for op in RUNTIME_OPERATIONS
}

RUNTIME_OBSERVABLES = {
    obs: OperatorProperties(invertible=True, controllable=True, differentiable=True)
    for obs in RUNTIME_OBSERVABLES
}

RUNTIME_MPS = {mp: [] for mp in RUNTIME_MPS}

# TODO: This should be removed after implementing `get_c_interface`
# for the following backend devices:
SUPPORTED_RT_DEVICES = {
    "null.qubit": ("NullQubit", "librtd_null_qubit"),
    "braket.aws.qubit": ("OpenQasmDevice", "librtd_openqasm"),
    "braket.local.qubit": ("OpenQasmDevice", "librtd_openqasm"),
}


@dataclass
class BackendInfo:
    """Backend information"""

    device_name: str
    c_interface_name: str
    lpath: str
    kwargs: Dict[str, Any]


# pylint: disable=too-many-branches
@debug_logger
def extract_backend_info(device: qml.devices.QubitDevice) -> BackendInfo:
    """Extract the backend info from a quantum device. The device is expected to carry a reference
    to a valid TOML config file."""

    dname = device.name
    if isinstance(device, qml.devices.LegacyDeviceFacade):
        dname = device.target_device.short_name  # pragma: no cover

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

    if dname == "braket.local.qubit":  # pragma: no cover
        device_kwargs["device_type"] = dname
        device_kwargs["backend"] = (
            # pylint: disable=protected-access
            device.target_device._device._delegate.DEVICE_ID
        )
    elif dname == "braket.aws.qubit":  # pragma: no cover
        device_kwargs["device_type"] = dname
        device_kwargs["device_arn"] = device._device._arn  # pylint: disable=protected-access
        if device.target_device._s3_folder:  # pylint: disable=protected-access
            device_kwargs["s3_destination_folder"] = str(
                device.target_device._s3_folder  # pylint: disable=protected-access
            )
    elif dname == "OQCDevice":
        device_kwargs["backend"] = device.backend

    for k, v in getattr(device, "device_kwargs", {}).items():
        if k not in device_kwargs:  # pragma: no branch
            device_kwargs[k] = v

    return BackendInfo(dname, device_name, device_lpath, device_kwargs)


def intersect_operations(
    a: Dict[str, OperatorProperties], b: Dict[str, OperatorProperties]
) -> Dict[str, OperatorProperties]:
    """Intersects two sets of operator properties"""
    return {k: a[k] & b[k] for k in (a.keys() & b.keys())}


def intersect_mps(a: dict[str, list], b: dict[str, list]) -> dict[str, list]:
    """Intersects two sets of measurement processes"""
    # In the dictionary, each measurement process is associated with a list of conditions.
    # Therefore, the intersection is really the union of constraints from both measurement
    # processes declarations, thus the | operator.
    return {k: list(set(a[k]) | set(b[k])) for k in (a.keys() & b.keys())}


@debug_logger
def get_qjit_device_capabilities(target_capabilities: DeviceCapabilities) -> DeviceCapabilities:
    """Calculate the set of supported quantum gates for the QJIT device from the gates
    allowed on the target quantum device."""

    # Supported gates of the target PennyLane's device
    qjit_capabilities = deepcopy(target_capabilities)

    # Intersection of gates and observables supported by the device and by Catalyst runtime.
    qjit_capabilities.operations = intersect_operations(
        target_capabilities.operations, RUNTIME_OPERATIONS
    )
    qjit_capabilities.observables = intersect_operations(
        target_capabilities.observables, RUNTIME_OBSERVABLES
    )
    qjit_capabilities.measurement_processes = intersect_mps(
        target_capabilities.measurement_processes, RUNTIME_MPS
    )

    # Enable dynamic qubit allocation with qml.allocate and qml.deallocate
    qjit_capabilities.operations.update(
        {
            "allocate": OperatorProperties(
                invertible=False, controllable=False, differentiable=False
            ),
            "deallocate": OperatorProperties(
                invertible=False, controllable=False, differentiable=False
            ),
        }
    )

    # Control-flow gates to be lowered down to the LLVM control-flow instructions
    qjit_capabilities.operations.update(
        {
            "Cond": OperatorProperties(invertible=True, controllable=True, differentiable=True),
            "WhileLoop": OperatorProperties(
                invertible=True, controllable=True, differentiable=True
            ),
            "ForLoop": OperatorProperties(invertible=True, controllable=True, differentiable=True),
            "Switch": OperatorProperties(invertible=True, controllable=True, differentiable=True),
        }
    )

    # Optionally enable runtime-powered mid-circuit measurements
    if target_capabilities.supported_mcm_methods:  # pragma: no branch
        qjit_capabilities.operations.update(
            {
                "MidCircuitMeasure": OperatorProperties(
                    invertible=False, controllable=False, differentiable=False
                )
            }
        )

    # Optionally enable runtime-powered adjoint of quantum gates (inversions)
    if any(ng.invertible for ng in target_capabilities.operations.values()):  # pragma: no branch
        qjit_capabilities.operations.update(
            {
                "HybridAdjoint": OperatorProperties(
                    invertible=True, controllable=True, differentiable=True
                )
            }
        )

    # Enable runtime-powered snapshot of quantum state at any particular instance
    qjit_capabilities.operations.update(
        {"Snapshot": OperatorProperties(invertible=False, controllable=False, differentiable=False)}
    )

    # TODO: Optionally enable runtime-powered quantum gate controlling once they
    #       are supported natively in MLIR.
    # if any(ng.controllable for ng in target_capabilities.operations.values()):
    #     qjit_capabilities.operations.update(
    #         {
    #             "HybridCtrl": OperatorProperties(
    #                 invertible=True, controllable=True, differentiable=True
    #             )
    #         }
    #     )

    return qjit_capabilities


class QJITDevice(qml.devices.Device):
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

    @staticmethod
    @debug_logger
    def extract_backend_info(device) -> BackendInfo:
        """Wrapper around extract_backend_info in the runtime module."""
        return extract_backend_info(device)

    @debug_logger_init
    def __init__(self, original_device):
        self.original_device = original_device

        for key, value in original_device.__dict__.items():
            self.__setattr__(key, value)

        check_device_wires(original_device.wires)

        super().__init__(wires=original_device.wires)

        # Capability loading
        # During initilization of QJITDevice, we just load the static toml device specs
        self.capabilities = get_qjit_device_capabilities(_load_device_capabilities(original_device))

        backend = QJITDevice.extract_backend_info(original_device)

        self.backend_name = backend.c_interface_name
        self.backend_lib = backend.lpath
        self.backend_kwargs = backend.kwargs

    @debug_logger
    def preprocess(
        self,
        ctx,
        execution_config: Optional[qml.devices.ExecutionConfig] = None,
        shots=None,
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
            CompilePipeline: A compile pipeline that when called returns QuantumTapes that can be
                compiled for the backend, and a postprocessing function to be called on the results
            ExecutionConfig: configuration with unset specifications filled in if relevant.

        This device supports operations and measurements based on the device ``capabilities``,
        which are created based on what is both compatible with Catalyst and what is supported the
        backend according to the backend TOML file).
        """

        if execution_config is None:
            execution_config = qml.devices.ExecutionConfig()
        _, config = self.original_device.preprocess(execution_config)

        pipeline = CompilePipeline()

        # During preprocessing, we now have info on whether the user is requesting execution
        # with shots.
        # Note that this new set of capabilities are only temporarily needed for the computation
        # of the preprocessing transform program.
        shots_not_provided = (shots is None) or (
            isinstance(shots, qml.measurements.shots.Shots) and shots.total_shots is None
        )
        if shots_not_provided and _requires_shots(self.capabilities):
            raise CompileError(
                textwrap.dedent(
                    f"""
                {self.original_device.name} does not support analytical simulation.
                Please supply the number of shots on the qnode.
                """
                )
            )
        capabilities = filter_device_capabilities_with_shots(
            capabilities=self.capabilities,
            shots_present=bool(shots),
            unitary_support=getattr(self.original_device, "_to_matrix_ops", None),
        )

        # measurement transforms may change operations on the tape to accommodate
        # measurement transformations, so must occur before decomposition
        measurement_transforms = self._measurement_transform_program(capabilities)
        config = replace(config, device_options=deepcopy(config.device_options))
        pipeline += measurement_transforms

        # decomposition to supported ops/measurements
        pipeline.add_transform(
            catalyst_decompose,
            ctx=ctx,
            capabilities=capabilities,
            grad_method=config.gradient_method,
        )

        # Catalyst program verification and validation
        pipeline.add_transform(
            verify_operations, grad_method=config.gradient_method, qjit_device=self
        )
        pipeline.add_transform(
            validate_measurements,
            capabilities,
            self.original_device.name,
            shots,
        )

        if config.gradient_method is not None:
            pipeline.add_transform(verify_no_state_variance_returns)
        if config.gradient_method == "adjoint":
            pipeline.add_transform(validate_observables_adjoint_diff, qjit_device=self)
        elif config.gradient_method == "parameter-shift":
            pipeline.add_transform(validate_observables_parameter_shift)

        return pipeline, config

    def _measurement_transform_program(self, capabilities):
        measurement_pipeline = CompilePipeline()
        if isinstance(self.original_device, SoftwareQQPP):
            return measurement_pipeline

        supports_sum_observables = "Sum" in capabilities.observables
        if capabilities.non_commuting_observables is False:
            measurement_pipeline.add_transform(split_non_commuting)
        elif not supports_sum_observables:
            measurement_pipeline.add_transform(split_to_single_terms)

        # if no observables are supported, we apply a transform to convert *everything* to the
        # readout basis, using either sample or counts based on device specification
        if not capabilities.observables:
            if not split_non_commuting in measurement_pipeline:
                # this *should* be redundant, a TOML that doesn't have observables should have
                # a False non_commuting_observables flag, but we aren't enforcing that
                measurement_pipeline.add_transform(split_non_commuting)
            if "SampleMP" in capabilities.measurement_processes:
                measurement_pipeline.add_transform(measurements_from_samples, self.wires)
            elif "CountsMP" in capabilities.measurement_processes:
                measurement_pipeline.add_transform(measurements_from_counts, self.wires)
            else:
                raise RuntimeError("The device does not support observables or sample/counts")

        elif not capabilities.measurement_processes.keys() - {"CountsMP", "SampleMP"}:
            # ToDo: this branch should become unnecessary when selective conversion of
            # unsupported MPs is finished, see ToDo below
            if not split_non_commuting in measurement_pipeline:  # pragma: no branch
                measurement_pipeline.add_transform(split_non_commuting)
            mp_transform = (
                measurements_from_samples
                if "SampleMP" in capabilities.measurement_processes
                else measurements_from_counts
            )
            measurement_pipeline.add_transform(mp_transform, self.wires)

        # if only some observables are supported, we try to diagonalize those that aren't
        elif not {"PauliX", "PauliY", "PauliZ", "Hadamard"}.issubset(capabilities.observables):
            if not split_non_commuting in measurement_pipeline:
                # the device might support non commuting measurements but not all the
                # Pauli + Hadamard observables, so here it is needed
                measurement_pipeline.add_transform(split_non_commuting)
            _obs_dict = {
                "PauliX": qml.X,
                "PauliY": qml.Y,
                "PauliZ": qml.Z,
                "Hadamard": qml.Hadamard,
            }
            # checking which base observables are unsupported and need to be diagonalized
            supported_observables = {"PauliX", "PauliY", "PauliZ", "Hadamard"}.intersection(
                capabilities.observables
            )
            supported_observables = [_obs_dict[obs] for obs in supported_observables]

            measurement_pipeline.add_transform(
                diagonalize_measurements, supported_base_obs=supported_observables
            )

        # ToDo: if some measurement types are unsupported, convert the unsupported MPs to
        # samples or counts (without diagonalizing or modifying observables). See ToDo above.

        return measurement_pipeline

    def execute(self, circuits, execution_config):
        """
        Raises: RuntimeError
        """
        raise RuntimeError("QJIT devices cannot execute tapes.")


# pragam: no cover
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


def _load_device_capabilities(device) -> DeviceCapabilities:
    """Get the contents of the device config toml file."""

    # TODO: This code exists purely for testing. Find another way to customize device Find a
    #       better way for a device to customize its capabilities as seen by Catalyst.
    if hasattr(device, "qjit_capabilities"):
        return device.qjit_capabilities

    if getattr(device, "config_filepath") is not None:
        toml_file = device.config_filepath

    else:
        # TODO: Remove this section when devices are guaranteed to have their own config file
        device_lpath = pathlib.Path(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
        name = device.short_name if isinstance(device, qml.devices.LegacyDevice) else device.name
        # The toml files name convention we follow is to replace
        # the dots with underscores in the device short name.
        toml_file_name = name.replace(".", "_") + ".toml"
        # And they are currently saved in the following directory.
        toml_file = device_lpath.parent / "lib" / "backend" / toml_file_name

    try:
        capabilities = DeviceCapabilities.from_toml_file(toml_file, "qjit")

    except FileNotFoundError as e:
        raise CompileError(
            "Attempting to compile program for incompatible device: "
            f"Config file ({toml_file}) does not exist"
        ) from e

    return capabilities


def filter_device_capabilities_with_shots(
    capabilities, shots_present, unitary_support=None
) -> DeviceCapabilities:
    """
    Process the device capabilities depending on whether shots are present in the user program,
    and whether device supports QubitUnitary ops.
    """

    device_capabilities = capabilities.filter(finite_shots=shots_present)

    # TODO: This is a temporary measure to ensure consistency of behaviour. Remove this
    #       when customizable multi-pathway decomposition is implemented. (Epic 74474)
    if unitary_support is not None:
        _to_matrix_ops = unitary_support
        setattr(device_capabilities, "to_matrix_ops", _to_matrix_ops)
        if _to_matrix_ops and not device_capabilities.supports_operation("QubitUnitary"):
            raise CompileError("The device that specifies to_matrix_ops must support QubitUnitary.")

    return device_capabilities


def get_device_capabilities(device, shots=False) -> DeviceCapabilities:
    """
    Get the capabilities from the device.

    TODO: this function is not actually used in the codebase, but is just used by the
    tests that want custom device capabilities.

    These tests piggy-back off the lightning device (which has "full capabilities") by
    calling this get_device_capabilities() on lightning, and manually delete some capabilities.

    We leave this function in for now, just for the tests.
    However, these tests should construct their capabilities properly, instead of piggy-back off
    lightning.
    """

    assert not isinstance(device, QJITDevice)

    return filter_device_capabilities_with_shots(
        _load_device_capabilities(device), bool(shots), getattr(device, "_to_matrix_ops", None)
    )


def is_dynamic_wires(wires: qml.wires.Wires):
    """
    Checks if a pennylane Wires object corresponds to a concrete number
    of wires or a dynamic number of wires.

    If the number of wires is static, the Wires object contains a list of wire labels,
    one label for each wires.
    If the number of wires is dynamic, the Wires object contains a single tracer that
    represents the number of wires.
    """
    # Automatic qubit management mode should not encounter this query
    assert wires is not None
    return (len(wires) == 1) and (isinstance(wires[0], DynamicJaxprTracer))


def check_device_wires(wires):
    """Validate requirements Catalyst imposes on device wires."""

    if wires is None:
        # Automatic qubit management mode, nothing to check
        return

    if len(wires) >= 2 or (not is_dynamic_wires(wires)):
        # A dynamic number of wires correspond to a single tracer for the number
        # Thus if more than one entry, must be static wires
        assert isinstance(wires, qml.wires.Wires)

        if not all(isinstance(wire, int) for wire in wires.labels):
            raise AttributeError("Catalyst requires continuous integer wire labels starting at 0.")

        if not wires.labels == tuple(range(len(wires))):
            raise AttributeError("Catalyst requires continuous integer wire labels starting at 0.")
    else:
        assert len(wires) == 1
        assert wires[0].shape in ((), (1,))
        if not wires[0].dtype == "int64":
            raise AttributeError("Number of wires on the device should be a scalar integer.")


def _requires_shots(capabilities):
    """
    Checks if a device capabilities requires shots.

    A device requires shots if all of its MPs are finite shots only.
    """
    return all(
        ExecutionCondition.FINITE_SHOTS_ONLY in MP_conditions
        for _, MP_conditions in capabilities.measurement_processes.items()
    )
