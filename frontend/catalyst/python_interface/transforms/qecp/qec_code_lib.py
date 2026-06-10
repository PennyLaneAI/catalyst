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
from functools import partial
from typing import Any, Self

import numpy as np

from catalyst.python_interface.dialects import qecp

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
            # keys need to match the names of the corresponding qecl.gate gates; if any adjoint
            # gates are supported, they should be included as a separate entry with key "gatename_adj"
            # values are a tuple of the qecp gate, and the indices its applied at in the codeblock
            # will need to be refactored for k>1
            "x": (qecp.PauliXOp, [4, 5, 6]),
            "y": (qecp.PauliYOp, [4, 5, 6]),
            "z": (qecp.PauliZOp, [4, 5, 6]),
            "hadamard": (qecp.HadamardOp, [0, 1, 2, 3, 4, 5, 6]),
            "s": (partial(qecp.SOp, adjoint=True), [0, 1, 2, 3, 4, 5, 6]),
            "s_adj": (qecp.SOp, [0, 1, 2, 3, 4, 5, 6]),
        },
        {
            "cnot": qecp.CnotOp,
        },
        #### Unitary encoding circuit ####
        {
            # ops (in the form of a qecp operator and the indices of the codeblock
            # it should be applied on) defining a transporter encoding circuit, i.e.
            # one that maps an input to the logical version of that input, rather
            # than just encoding logical 0
            "ops": [
                (qecp.HadamardOp, [1]),
                (qecp.HadamardOp, [2]),
                (qecp.HadamardOp, [3]),
                (qecp.CnotOp, [1, 0]),
                (qecp.CnotOp, [2, 4]),
                (qecp.CnotOp, [6, 5]),
                (qecp.CnotOp, [2, 0]),
                (qecp.CnotOp, [3, 5]),
                (qecp.CnotOp, [6, 4]),
                (qecp.CnotOp, [2, 6]),
                (qecp.CnotOp, [3, 4]),
                (qecp.CnotOp, [1, 5]),
                (qecp.CnotOp, [1, 6]),
                (qecp.CnotOp, [3, 0]),
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
        # Z: X-flip on the all the bits on one set of 3: doesn't modify |0>, generates overall -1 sign for |1>
        # CNOT is transversal for all CSS codes
        # There are no Hadamard or S gates for this code #ToDo: reference
        {"x": (qecp.PauliZOp, [0, 3, 6]), "z": (qecp.PauliXOp, [0, 1, 2])},
        {"cnot": qecp.CnotOp},
        #### Unitary encoding circuit ####
        # jukebox generated this and verified it in simulation. # ToDo: provide validation info on notion
        {
            "ops": [
                (qecp.CnotOp, [0, 3]),
                (qecp.CnotOp, [0, 6]),
                (qecp.HadamardOp, [0]),
                (qecp.HadamardOp, [3]),
                (qecp.HadamardOp, [6]),
                (qecp.CnotOp, [0, 1]),
                (qecp.CnotOp, [0, 2]),
                (qecp.CnotOp, [3, 4]),
                (qecp.CnotOp, [3, 5]),
                (qecp.CnotOp, [6, 7]),
                (qecp.CnotOp, [6, 8]),
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
            key should match the gate name in the qecl dialect, and the value is a tuple
            containing the qecp op to be applied, and the indices. Assumes k=1.
        transversal_2q_gates (dict): A dictionary of two-qubit transversal gates. The
            key should match the gate name in the qecl dialect, and the value is the qecp
            op to be applied, and the indices. Assumes k=1. Does not specify indices - for
            now, we assume 2-qubit gates between two codeblocks, where the gate is applied
            between all pairs of corresponding qubits.
        unitary_encoding (dict): A dictionary defining the unitary encoding for the code words.
            It includes 'ops' (a list of tuples that each indicate a qecp gate and the codeblock
            indices it should be applied to), and a state-prep index. The state-prep index is the
            index to apply physical gates to before encoding, in order to encode a non-zero state
            - for example, applying H-T at this index before unitary encoding generates a magic
            state (not fault-tolerantly). For this to work, the chosen encoder should be an isometric
            encoder, i.e. it should map the input on one of the wires to the codespace, rather than
            just encoding zero.
    """

    name: str
    n: int
    k: int
    d: int
    x_tanner: np.ndarray
    z_tanner: np.ndarray
    transversal_1q_gates: dict
    transversal_2q_gates: dict
    unitary_encoding: dict

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

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """A builder function that returns a `QecCode` instance from a dictionary.

        Keys in the dictionary that do not have a corresponding field in `QecCode` are dropped.

        Example
        -------

        >>> QecCode.from_dict({
        ...    'name': "Steane",
        ...    'n': 7,
        ...    'k': 1,
        ...    'd': 3,
        ...    "x_tanner": np.eye(7),
        ...    "z_tanner": np.eye(7),
        ...    "transversal_1q_gates": {},
        ...    "transversal_2q_gates": {},
        ...    "unitary_encoding": {}
        ...    })
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
