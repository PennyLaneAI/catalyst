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

from copy import deepcopy
from functools import partial
from typing import Optional, Set, Union

import pennylane as qml
from pennylane.measurements import MidMeasureMP
from pennylane.transforms.core import TransformProgram

from catalyst.device.decomposition import (
    catalyst_acceptance,
    decompose,
    measurements_from_counts,
)
from catalyst.utils.exceptions import CompileError
from catalyst.utils.patching import Patcher
from catalyst.utils.runtime import BackendInfo
from catalyst.utils.toml import (
    DeviceCapabilities,
    OperationProperties,
    intersect_operations,
    pennylane_operation_set,
)

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


def get_qjit_device_capabilities(target_capabilities: DeviceCapabilities) -> Set[str]:
    """Calculate the set of supported quantum gates for the QJIT device from the gates
    allowed on the target quantum device."""
    # Supported gates of the target PennyLane's device
    qjit_capabilities = deepcopy(target_capabilities)

    # Gates that Catalyst runtime supports
    qir_gates = RUNTIME_OPERATIONS

    # Intersection of the above
    qjit_capabilities.native_ops = intersect_operations(target_capabilities.native_ops, qir_gates)

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
                    invertible=True, controllable=True, differentiable=False
                )
            }
        )

    # Optionally enable runtime-powered quantum gate adjointing (inversions)
    if all(ng.invertible for ng in target_capabilities.native_ops.values()):
        qjit_capabilities.native_ops.update(
            {
                "Adjoint": OperationProperties(
                    invertible=True, controllable=True, differentiable=True
                )
            }
        )

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
        program.add_transform(decompose, ctx=ctx, stopping_condition=ops_acceptance)

        if self.measurement_processes == {"Counts"}:
            program.add_transform(measurements_from_counts)

        # TODO: Add Catalyst program verification and validation
        return program, config

    def execute(self, circuits, execution_config):
        """
        Raises: RuntimeError
        """
        raise RuntimeError("QJIT devices cannot execute tapes.")


AnyQJITDevice = Union[QJITDevice, QJITDeviceNewAPI]
