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
import pennylane as qml
from pennylane.measurements import MidMeasureMP

from catalyst.utils.exceptions import CompileError
from catalyst.utils.patching import Patcher


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

    # These must be present even if empty.
    operations = []
    observables = []

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
    def _get_operations_to_convert_to_matrix(_config):
        # We currently override and only set a few gates to preserve existing behaviour.
        # We could choose to read from config and use the "matrix" gates.
        # However, that affects differentiability.
        # None of the "matrix" gates with more than 2 qubits parameters are differentiable.
        # TODO: https://github.com/PennyLaneAI/catalyst/issues/398
        return {"MultiControlledX", "BlockEncode"}

    @staticmethod
    def _check_mid_circuit_measurement(config):
        return config["compilation"]["mid_circuit_measurement"]

    @staticmethod
    def _check_adjoint(config):
        return config["compilation"]["quantum_adjoint"]

    @staticmethod
    def _check_quantum_control(config):
        return config["compilation"]["quantum_control"]

    @staticmethod
    def _set_supported_operations(config):
        """Override the set of supported operations."""
        native_gates = set(config["operators"]["gates"][0]["native"])
        qir_gates = QJITDevice.operations_supported_by_QIR_runtime
        supported_native_gates = list(set.intersection(native_gates, qir_gates))
        QJITDevice.operations = supported_native_gates

        # These are added unconditionally.
        QJITDevice.operations += ["Cond", "WhileLoop", "ForLoop"]

        if QJITDevice._check_mid_circuit_measurement(config):  # pragma: no branch
            QJITDevice.operations += ["MidCircuitMeasure"]

        if QJITDevice._check_adjoint(config):
            QJITDevice.operations += ["Adjoint"]

        if QJITDevice._check_quantum_control(config):  # pragma: nocover
            # TODO: Once control is added on the frontend.
            gates_to_be_decomposed_if_controlled = [
                "Identity",
                "CNOT",
                "CY",
                "CZ",
                "CSWAP",
                "CRX",
                "CRY",
                "CRZ",
                "CRot",
            ]
            native_controlled_gates = ["ControlledQubitUnitary"] + [
                f"C({gate})"
                for gate in native_gates
                if gate not in gates_to_be_decomposed_if_controlled
            ]
            QJITDevice.operations += native_controlled_gates

    @staticmethod
    def _set_supported_observables(config):
        """Override the set of supported observables."""
        QJITDevice.observables = config["operators"]["observables"]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        config,
        shots=None,
        wires=None,
        backend_name=None,
        backend_lib=None,
        backend_kwargs=None,
    ):
        QJITDevice._set_supported_operations(config)
        QJITDevice._set_supported_observables(config)

        self.config = config
        self.backend_name = backend_name if backend_name else "default"
        self.backend_lib = backend_lib if backend_lib else ""
        self.backend_kwargs = backend_kwargs if backend_kwargs else {}
        super().__init__(wires=wires, shots=shots)

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

        decompose_to_qubit_unitary = QJITDevice._get_operations_to_convert_to_matrix(self.config)

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
