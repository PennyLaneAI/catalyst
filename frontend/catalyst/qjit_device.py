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
"""This module contains the qjit device classes.
"""
from functools import partial
from typing import Optional, Set

import pennylane as qml
from pennylane.devices.preprocess import decompose
from pennylane.measurements import MidMeasureMP
from pennylane.transforms.core import TransformProgram

from catalyst.preprocess import catalyst_acceptance, decompose_ops_to_unitary
from catalyst.utils.exceptions import CompileError
from catalyst.utils.patching import Patcher
from catalyst.utils.runtime import (
    BackendInfo,
    get_pennylane_observables,
    get_pennylane_operations,
)
from catalyst.utils.toml import (
    TOMLDocument,
    check_adjoint_flag,
    check_mid_circuit_measurement_flag,
)

RUNTIME_OPERATIONS = {
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
    "Rot",
    "RX",
    "RY",
    "RZ",
    "S",
    "SWAP",
    "T",
    "Toffoli",
    "GlobalPhase",
    "C(GlobalPhase)",
    "C(Hadamard)",
    "C(IsingXX)",
    "C(IsingXY)",
    "C(IsingYY)",
    "C(ISWAP)",
    "C(MultiRZ)",
    "ControlledQubitUnitary",
    "C(PauliX)",
    "C(PauliY)",
    "C(PauliZ)",
    "C(PhaseShift)",
    "C(PSWAP)",
    "C(Rot)",
    "C(RX)",
    "C(RY)",
    "C(RZ)",
    "C(S)",
    "C(SWAP)",
    "C(T)",
}


def get_qjit_pennylane_operations(
    config: TOMLDocument, shots_present: bool, device_name: str
) -> Set[str]:
    """Calculate the set of supported quantum gates for the QJIT device from the gates
    allowed on the target quantum device."""
    # Supported gates of the target PennyLane's device
    native_gates = get_pennylane_operations(config, shots_present, device_name)
    # Gates that Catalyst runtime supports
    qir_gates = RUNTIME_OPERATIONS
    supported_gates = set.intersection(native_gates, qir_gates)

    # Control-flow gates to be lowered down to the LLVM control-flow instructions
    supported_gates.update({"Cond", "WhileLoop", "ForLoop"})

    # Optionally enable runtime-powered mid-circuit measurments
    if check_mid_circuit_measurement_flag(config):  # pragma: no branch
        supported_gates.update({"MidCircuitMeasure"})

    # Optionally enable runtime-powered quantum gate adjointing (inversions)
    if check_adjoint_flag(config, shots_present):
        supported_gates.update({"Adjoint"})

    return supported_gates


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
    def _get_operations_to_convert_to_matrix(_config: TOMLDocument) -> Set[str]:
        # We currently override and only set a few gates to preserve existing behaviour.
        # We could choose to read from config and use the "matrix" gates.
        # However, that affects differentiability.
        # None of the "matrix" gates with more than 2 qubits parameters are differentiable.
        # TODO: https://github.com/PennyLaneAI/catalyst/issues/398
        return {"MultiControlledX", "BlockEncode"}

    def __init__(
        self,
        target_config: TOMLDocument,
        shots=None,
        wires=None,
        backend: Optional[BackendInfo] = None,
    ):
        super().__init__(wires=wires, shots=shots)

        self.target_config = target_config
        self.backend_name = backend.c_interface_name if backend else "default"
        self.backend_lib = backend.lpath if backend else ""
        self.backend_kwargs = backend.kwargs if backend else {}
        device_name = backend.device_name if backend else "default"

        shots_present = shots is not None
        self._operations = get_qjit_pennylane_operations(target_config, shots_present, device_name)
        self._observables = get_pennylane_observables(target_config, shots_present, device_name)

    @property
    def operations(self) -> Set[str]:
        """Get the device operations"""
        return self._operations

    @property
    def observables(self) -> Set[str]:
        """Get the device observables"""
        return self._observables

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
            :class:`~.pennylane_extensions.QJITDevice`.

        Args:
            circuit: circuit to expand
            max_expansion: the maximum number of expansion steps if no fixed-point is reached.
        """
        # Ensure catalyst.measure is used instead of qml.measure.
        if any(isinstance(op, MidMeasureMP) for op in circuit.operations):
            raise CompileError("Must use 'measure' from Catalyst instead of PennyLane.")

        decompose_to_qubit_unitary = QJITDevice._get_operations_to_convert_to_matrix(
            self.target_config
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

    @staticmethod
    def _get_operations_to_convert_to_matrix(_config: TOMLDocument) -> Set[str]:  # pragma: no cover
        # We currently override and only set a few gates to preserve existing behaviour.
        # We could choose to read from config and use the "matrix" gates.
        # However, that affects differentiability.
        # None of the "matrix" gates with more than 2 qubits parameters are differentiable.
        # TODO: https://github.com/PennyLaneAI/catalyst/issues/398
        return {"MultiControlledX", "BlockEncode"}

    def __init__(
        self,
        original_device,
        target_config: TOMLDocument,
        backend: Optional[BackendInfo] = None,
    ):
        self.original_device = original_device

        for key, value in original_device.__dict__.items():
            self.__setattr__(key, value)
        super().__init__(wires=original_device.wires, shots=original_device.shots)

        self.target_config = target_config
        self.backend_name = backend.c_interface_name if backend else "default"
        self.backend_lib = backend.lpath if backend else ""
        self.backend_kwargs = backend.kwargs if backend else {}
        device_name = backend.device_name if backend else "default"

        shots_present = original_device.shots is not None
        self._operations = get_qjit_pennylane_operations(target_config, shots_present, device_name)
        self._observables = get_pennylane_observables(target_config, shots_present, device_name)

    @property
    def operations(self) -> Set[str]:
        """Get the device operations"""
        return self._operations

    @property
    def observables(self) -> Set[str]:
        """Get the device observables"""
        return self._observables

    def preprocess(
        self,
        execution_config: qml.devices.ExecutionConfig = qml.devices.DefaultExecutionConfig,
    ):
        """Device preprocessing function."""
        # TODO: readd the user preprocessing program once transforms are compatible with TOML files
        _, config = self.original_device.preprocess(execution_config)
        program = TransformProgram()

        convert_to_matrix_ops = {"MultiControlledX", "BlockEncode"}
        program.add_transform(decompose_ops_to_unitary, convert_to_matrix_ops)

        ops_acceptance = partial(catalyst_acceptance, operations=self.operations)
        program.add_transform(
            decompose, stopping_condition=ops_acceptance, name=self.original_device.name
        )

        # TODO: Add Catalyst program verification and validation
        return program, config

    def execute(self, circuits, execution_config):
        """
        Raises: RuntimeError
        """
        raise RuntimeError("QJIT devices cannot execute tapes.")
