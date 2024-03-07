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
from typing import Optional, Set

import pennylane as qml
from pennylane.measurements import MidMeasureMP

from catalyst.utils.exceptions import CompileError
from catalyst.utils.patching import Patcher
from catalyst.utils.runtime import (
    BackendInfo,
    deduce_native_controlled_gates,
    get_pennylane_operations,
)
from catalyst.utils.toml import (
    TOMLDocument,
    check_adjoint_flag,
    check_mid_circuit_measurement_flag,
    get_observables,
)


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

    operations_supported_by_QIR_runtime = {
        "Identity",
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "S",
        "T",
        "PhaseShift",
        "RX",
        "RY",
        "RZ",
        "Rot",
        "CNOT",
        "CY",
        "CZ",
        "SWAP",
        "IsingXX",
        "IsingYY",
        "IsingXY",
        "ControlledPhaseShift",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
        "CSWAP",
        "Toffoli",
        "MultiRZ",
        "QubitUnitary",
        "ISWAP",
        "PSWAP",
        "GlobalPhase",
    }

    @staticmethod
    def _get_operations_to_convert_to_matrix(_config: TOMLDocument) -> Set[str]:
        # We currently override and only set a few gates to preserve existing behaviour.
        # We could choose to read from config and use the "matrix" gates.
        # However, that affects differentiability.
        # None of the "matrix" gates with more than 2 qubits parameters are differentiable.
        # TODO: https://github.com/PennyLaneAI/catalyst/issues/398
        return {"MultiControlledX", "BlockEncode"}

    @staticmethod
    def _get_supported_operations(config: TOMLDocument, shots_present) -> Set[str]:
        """Override the set of supported operations."""
        # Supported gates of the target PennyLane's device
        native_gates = get_pennylane_operations(config, shots_present)
        qir_gates = set.union(
            QJITDevice.operations_supported_by_QIR_runtime,
            deduce_native_controlled_gates(QJITDevice.operations_supported_by_QIR_runtime),
        )
        supported_gates = list(set.intersection(native_gates, qir_gates))

        # These are added unconditionally.
        supported_gates += ["Cond", "WhileLoop", "ForLoop"]

        if check_mid_circuit_measurement_flag(config):  # pragma: no branch
            supported_gates += ["MidCircuitMeasure"]

        if check_adjoint_flag(config):
            supported_gates += ["Adjoint"]

        supported_gates += ["ControlledQubitUnitary"]
        return set(supported_gates)

    def __init__(
        self,
        target_config: TOMLDocument,
        shots=None,
        wires=None,
        backend: Optional[BackendInfo] = None,
    ):
        super().__init__(wires=wires, shots=shots)

        self.target_config = target_config
        self.backend_name = backend.name if backend else "default"
        self.backend_lib = backend.lpath if backend else ""
        self.backend_kwargs = backend.kwargs if backend else {}

        shots_present = shots is not None
        self._operations = set(QJITDevice._get_supported_operations(target_config, shots_present))
        self._observables = set(get_observables(target_config, shots_present))

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
