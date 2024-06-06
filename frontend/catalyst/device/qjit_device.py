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
from typing import Any, Dict, Optional, Set

import pennylane as qml
from pennylane.measurements import MidMeasureMP
from pennylane.transforms.core import TransformProgram

from catalyst.device.decomposition import (
    catalyst_acceptance,
    catalyst_decompose,
    measurements_from_counts,
)
from catalyst.logging import debug_logger, debug_logger_init
from catalyst.utils.exceptions import CompileError
from catalyst.utils.patching import Patcher
from catalyst.utils.runtime_environment import get_lib_path
from catalyst.utils.toml import (
    DeviceCapabilities,
    OperationProperties,
    intersect_operations,
    pennylane_operation_set,
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
    "ControlledQubitUnitary",
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
# The runtime interface does not care about specific gate properties, so set them all to True.
RUNTIME_OPERATIONS = {
    op: OperationProperties(invertible=True, controllable=True, differentiable=True)
    for op in RUNTIME_OPERATIONS
}

# TODO: This should be removed after implementing `get_c_interface`
# for the following backend devices:
SUPPORTED_RT_DEVICES = {
    "lightning.qubit": ("LightningSimulator", "librtd_lightning"),
    "lightning.kokkos": ("LightningKokkosSimulator", "librtd_lightning"),
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
def extract_backend_info(device: qml.QubitDevice, capabilities: DeviceCapabilities) -> BackendInfo:
    """Extract the backend info from a quantum device. The device is expected to carry a reference
    to a valid TOML config file."""

    dname = device.name
    if isinstance(device, qml.Device):
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
        if isinstance(device, qml.Device):
            device_kwargs["shots"] = device.shots if device.shots else 0
        else:
            # TODO: support shot vectors
            device_kwargs["shots"] = device.shots.total_shots if device.shots else 0

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
    qjit_config = deepcopy(target_capabilities)

    # Gates that Catalyst runtime supports
    qir_gates = RUNTIME_OPERATIONS

    # Intersection of the above
    qjit_config.native_ops = intersect_operations(target_capabilities.native_ops, qir_gates)

    # Control-flow gates to be lowered down to the LLVM control-flow instructions,
    # all of which can be inverted, controlled, and differentiated.
    qjit_config.native_ops.update(
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
        qjit_config.native_ops.update(
            {
                "MidCircuitMeasure": OperationProperties(
                    invertible=True, controllable=True, differentiable=False
                )
            }
        )

    # Optionally enable runtime-powered quantum gate adjointing (inversions)
    if all(ng.invertible for ng in target_capabilities.native_ops.values()):
        qjit_config.native_ops.update(
            {
                "HybridAdjoint": OperationProperties(
                    invertible=True, controllable=True, differentiable=True
                )
            }
        )

    return qjit_config


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
        original_device_capabilities: DeviceCapabilities,
        shots=None,
        wires=None,
        backend: Optional[BackendInfo] = None,
    ):
        super().__init__(wires=wires, shots=shots)

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

        1. All :class:`qml.QubitUnitary <pennylane.ops.op_math.Controlled>` operations
            will decompose to :class:`qml.QubitUnitary <pennylane.QubitUnitary>` operations.
        2. :class:`qml.ControlledQubitUnitary <pennylane.ControlledQubitUnitary>` operations
            will decompose to :class:`qml.QubitUnitary <pennylane.QubitUnitary>` operations.
        3. The list of device-supported gates employed by Catalyst is currently different than
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
            self.capabilities
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

        if original_device.wires is None:
            raise AttributeError("Catalyst does not support devices without set wires.")

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
        """Device preprocessing function."""
        # TODO: readd the device preprocessing program once transforms are compatible with
        # TOML files
        _, config = self.original_device.preprocess(execution_config)
        program = TransformProgram()

        ops_acceptance = partial(catalyst_acceptance, operations=self.operations)
        program.add_transform(
            catalyst_decompose,
            ctx=ctx,
            stopping_condition=ops_acceptance,
            capabilities=self.qjit_capabilities,
        )

        if self.measurement_processes == {"Counts"}:
            program.add_transform(measurements_from_counts)

        # TODO: Add Catalyst program verification and validation
        return program, config

    def execute(self, circuits, execution_config):
        """
        Raises: RuntimeError
        """
        raise RuntimeError("QJIT devices cannot execute tapes.")


def filter_out_adjoint(operations):
    """Remove Adjoint from operations.

    Args:
        operations (List[Str]): List of strings with names of supported operations

    Returns:
        List: A list of strings with names of supported operations with Adjoint and C gates
        removed.
    """
    adjoint = re.compile(r"^Adjoint\(.*\)$")

    def is_not_adj(op):
        return not re.match(adjoint, op)

    operations_no_adj = filter(is_not_adj, operations)
    return set(operations_no_adj)


def check_no_overlap(*args, device_name):
    """Check items in *args are mutually exclusive.

    Args:
        *args (List[Str]): List of strings.
        device_name (str): Device name for error reporting.

    Raises:
        CompileError
    """
    set_of_sets = [set(arg) for arg in args]
    union = set.union(*set_of_sets)
    len_of_sets = [len(arg) for arg in args]
    if sum(len_of_sets) == len(union):
        return

    overlaps = set()
    for s in set_of_sets:
        overlaps.update(s - union)
        union = union - s

    msg = f"Device '{device_name}' has overlapping gates: {overlaps}"
    raise CompileError(msg)


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
        device (qml.Device): An instance of a quantum device.
        config (TOMLDocument): A TOML document representation.

    Raises: CompileError
    """

    if not device_capabilities.qjit_compatible_flag:
        raise CompileError(
            f"Attempting to compile program for incompatible device '{device.name}': "
            f"Config is not marked as qjit-compatible"
        )

    device_name = device.short_name if isinstance(device, qml.Device) else device.name

    native = pennylane_operation_set(device_capabilities.native_ops)
    decomposable = pennylane_operation_set(device_capabilities.to_decomp_ops)
    matrix = pennylane_operation_set(device_capabilities.to_matrix_ops)

    check_no_overlap(native, decomposable, matrix, device_name=device_name)

    if hasattr(device, "operations") and hasattr(device, "observables"):
        # For gates, we require strict match
        device_gates = filter_out_adjoint(set(device.operations))
        # Lightning-kokkis might support C(GlobalPhase) in Python, but not in C++. We remove this
        # gate before calling the validation.
        # See https://github.com/PennyLaneAI/pennylane-lightning/pull/642#discussion_r1535478642
        if device_name == "lightning.kokkos":
            device_gates = device_gates - {"C(GlobalPhase)"}
        spec_gates = filter_out_adjoint(set.union(native, matrix, decomposable))
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
