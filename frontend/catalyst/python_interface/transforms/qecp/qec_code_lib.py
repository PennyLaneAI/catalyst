# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains a library of QEC codes."""

from dataclasses import dataclass, fields
from enum import StrEnum
from functools import partial
from typing import Any, Callable, Self

import numpy as np
from xdsl.ir import Operation

from catalyst.python_interface.dialects import qecp
from catalyst.python_interface.transforms.qecp._code_registry import _CODE_REGISTRY


class SupportedGates(StrEnum):
    """Enum of gate string identifiers that are supported for QEC code definition."""

    I = "I"  # Identity  # noqa: E741
    X = "X"  # Pauli X
    Y = "Y"  # Pauli Y
    Z = "Z"  # Pauli Z
    H = "H"  # Hadamard
    S = "S"  # S phase
    Sa = "Sa"  # Adjoint of S phase
    CNOT = "CNOT"  # CNOT


def qecp_gate_op_from_string(gate_str: str) -> Callable[..., Operation]:
    """Parse a gate string identifier from a QEC code definition and return the corresponding
    constructible qecp operation type. In cases where the gate string identifier specifies the
    adjoint of a gate, a `functools.partial` wrapper object is returned with the `adjoint=True`
    parameter set.

    Raises a ValueError for invalid gate string identifiers.
    """
    op_type: Callable[..., Operation]

    match gate_str:
        case SupportedGates.I:
            op_type = qecp.IdentityOp
        case SupportedGates.X:
            op_type = qecp.PauliXOp
        case SupportedGates.Y:
            op_type = qecp.PauliYOp
        case SupportedGates.Z:
            op_type = qecp.PauliZOp
        case SupportedGates.H:
            op_type = qecp.HadamardOp
        case SupportedGates.S:
            op_type = qecp.SOp
        case SupportedGates.Sa:
            op_type = partial(qecp.SOp, adjoint=True)
        case SupportedGates.CNOT:
            op_type = qecp.CnotOp
        case _:
            supported_gates_str = ", ".join(gate for gate in SupportedGates)
            raise ValueError(
                f"Invalid gate in QEC code definition: '{gate_str}'. Supported gates are: "
                f"{supported_gates_str}"
            )

    return op_type


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class QecCode:
    """A class to store all relevant information for any [[n, k, d]] stabilizer QEC code.

    Args:
        name (str): A unique identifier of the QEC code.
        n (int): The code's number of QEC physical qubits.
        k (int): The code's number of QEC logical qubits.
        d (int): The code's distance.
        x_tanner (np.ndarray): The code's X Tanner graph
        z_tanner (np.ndarray): The code's Z Tanner graph
        transversal_1q_gates (dict): A dictionary of single-qubit transversal gates. The
            key should match the gate name in the qecl dialect, and the value is a tuple of gate
            string identifiers specifying the qecp ops to be applied. Assumes k=1.
        transversal_2q_gates (dict): A dictionary of two-qubit transversal gates. The
            key should match the gate name in the qecl dialect, and the value is string identifier
            of the qecp op to be applied. Assumes k=1 and that two-qubit gates are applied between
            two codeblocks, where the gate is applied between all pairs of corresponding qubits.
        unitary_encoding (dict): A dictionary defining the unitary encoding for the code words.
            It includes 'ops' (a list of tuples that each indicate a qecp gate and the codeblock
            indices it should be applied to), and a state-prep index. The state-prep index is the
            index to apply physical gates to before encoding, in order to encode a non-zero state -
            for example, applying H-T at this index before unitary encoding generates a magic state
            (not fault-tolerantly). For this to work, the chosen encoder should be an isometric
            encoder, i.e. it should map the input on one of the wires to the codespace, rather than
            just encoding zero.
    """

    name: str
    n: int
    k: int
    d: int
    x_tanner: np.ndarray
    z_tanner: np.ndarray
    transversal_1q_gates: dict[str, tuple[str, ...]]
    transversal_2q_gates: dict[str, str]
    unitary_encoding: dict[str, Any]

    def __str__(self):
        if self.name == "" or str.isspace(self.name):
            name = "<unknown>"
        else:
            name = self.name

        return f"[[{self.n}, {self.k}, {self.d}]] {name}"

    def __repr__(self):
        if self.name == "" or str.isspace(self.name):
            name = "<unknown>"
        else:
            name = self.name

        return f"QecCode(name='{name}', n={self.n}, k={self.k}, d={self.d})"

    def __post_init__(self):
        invalid_transversal_gates: list[str] = []

        for gate_name, gate_ops in self.transversal_1q_gates.items():
            if len(gate_ops) != self.n:
                invalid_transversal_gates.append(gate_name)

        if invalid_transversal_gates:
            err_msg = (
                f"Invalid single-qubit transversal gate definition(s): attempting to instantiate a "
                f"QEC code '{self.name}' with physical codeblock size n = {self.n}, but with "
                f"transversal "
            )

            err_msg += ", ".join(
                [
                    f"gate '{gate_name}' of length {len(self.transversal_1q_gates[gate_name])}"
                    for gate_name in invalid_transversal_gates
                ]
            )

            raise ValueError(err_msg)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """A builder function that returns a `QecCode` instance from a dictionary.

        Keys in the dictionary that do not have a corresponding field in `QecCode` are dropped.

        Example
        -------

        >>> QecCode.from_dict({
        ...     'name': "Steane",
        ...     'n': 7,
        ...     'k': 1,
        ...     'd': 3,
        ...     "x_tanner": np.eye(7),
        ...     "z_tanner": np.eye(7),
        ...     "transversal_1q_gates": {},
        ...     "transversal_2q_gates": {},
        ...     "unitary_encoding": {}
        ... })
        QecCode(name='Steane', n=7, k=1, d=3)
        """
        # Filter dictionary to keep only keys that are fields of this dataclass
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)

    @classmethod
    def get(cls, name: str) -> Self:
        """A builder function that returns a `QecCode` instance for a supported QEC code.

        Example
        -------

        >>> QecCode.get("Steane")
        QecCode(name='Steane', n=7, k=1, d=3)
        """
        qec_code_params = _CODE_REGISTRY.get(name)
        if qec_code_params is None:
            raise KeyError(f"QEC code {name} not found")

        return cls(name, *qec_code_params)

    @property
    def correctable_errors(self) -> int:
        """Return the number of correctable errors of the QEC code.

        For a code with distance :math:`d`, the number of correctable errors :math:`t` is given by

        .. math::

            t = \\lfloor (d - 1) / 2 \\rfloor

        Example
        -------

        >>> code = QecCode.get("Steane")
        >>> code.correctable_errors
        """
        return (self.d - 1) // 2
