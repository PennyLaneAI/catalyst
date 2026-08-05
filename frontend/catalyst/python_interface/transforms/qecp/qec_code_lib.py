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


_CODE_REGISTRY: dict[str, tuple[Any, ...]] = {
    # the indices/ordering for the operators and encodings in the Steane code are those used
    # in https://arxiv.org/pdf/2107.07505
    "Steane": (
        7,
        1,
        3,
        #### Stabilizers ####
        np.array([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]]),
        np.array([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]]),
        #### Transversal gates ####
        {
            # Keys need to match the names of the corresponding qecl.gate gates; if any adjoint
            # gates are supported, they should be included as a separate entry with key
            # "gatename_adj". Values are a tuple of gate string identifiers specifying the qecp ops
            # the codeblock will need to be refactored for k>1
            "x": ("I", "I", "I", "I", "X", "X", "X"),
            "y": ("I", "I", "I", "I", "Y", "Y", "Y"),
            "z": ("I", "I", "I", "I", "Z", "Z", "Z"),
            "hadamard": ("H", "H", "H", "H", "H", "H", "H"),
            "s": ("Sa", "Sa", "Sa", "Sa", "Sa", "Sa", "Sa"),
            "s_adj": ("S", "S", "S", "S", "S", "S", "S"),
        },
        {
            "cnot": "CNOT",
        },
        #### Unitary encoding circuit ####
        {
            # ops (in the form of a gate string identifier and the indices of the codeblock
            # it should be applied on) defining a transporter encoding circuit, i.e.
            # one that maps an input to the logical version of that input, rather
            # than just encoding logical 0
            "ops": [
                ("H", [1]),
                ("H", [2]),
                ("H", [3]),
                ("CNOT", [1, 0]),
                ("CNOT", [2, 4]),
                ("CNOT", [6, 5]),
                ("CNOT", [2, 0]),
                ("CNOT", [3, 5]),
                ("CNOT", [6, 4]),
                ("CNOT", [2, 6]),
                ("CNOT", [3, 4]),
                ("CNOT", [1, 5]),
                ("CNOT", [1, 6]),
                ("CNOT", [3, 0]),
            ],
            # The state_prep_index is the index of the physical qubit that the state is
            # injected on (i.e. for a magic state, -H-T is applied here pre-encoding).
            # Must be consistent with the qubit treated as the encoding "input" by the
            # cnot_indices ordering above. See https://arxiv.org/pdf/2107.07505 (Fig 10)
            "state_prep_index": 6,
        },
    ),
    "Shor913": (
        # see Steane code for general comments on the inputs to define the code
        9,
        1,
        3,
        #### Stabilizers ####
        # from error correction zoo, https://errorcorrectionzoo.org/c/shor_nine
        np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 1]]),
        np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
            ]
        ),
        #### Transversal gates ####
        # X: physical Z on a single qubit from each set of 3, make +|111> into -|111> and vice-versa
        # Z: X-flip on the all the bits on one set of 3: doesn't modify |0>, generates overall -1
        #    sign for |1>
        # Y: Y = iXZ (we ignore global phase so can use Y ~ XZ)
        # CNOT is transversal for all CSS codes
        # There are no transversal Hadamard or S gates for this code
        {
            "x": ("Z", "I", "I", "Z", "I", "I", "Z", "I", "I"),
            "y": ("Y", "X", "X", "Z", "I", "I", "Z", "I", "I"),
            "z": ("X", "X", "X", "I", "I", "I", "I", "I", "I"),
        },
        {"cnot": "CNOT"},
        #### Unitary encoding circuit ####
        # References:
        #   [1] P. Shor (1995), Scheme for reducing decoherence in quantum computer memory,
        #         Phys. Rev. A 52, R2493. https://doi.org/10.1103/PhysRevA.52.R2493.
        #   [2] O. Khalifa, et al. (2021), Digital System Design for Quantum Error Correction Codes,
        #         Contrast Media & Molecular Imaging, 1101911. https://doi.org/10.1155/2021/1101911.
        #         (Open Access)
        {
            "ops": [
                ("CNOT", [0, 3]),
                ("CNOT", [0, 6]),
                ("H", [0]),
                ("H", [3]),
                ("H", [6]),
                ("CNOT", [0, 1]),
                ("CNOT", [0, 2]),
                ("CNOT", [3, 4]),
                ("CNOT", [3, 5]),
                ("CNOT", [6, 7]),
                ("CNOT", [6, 8]),
            ],
            "state_prep_index": 0,
        },
    ),
}


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
