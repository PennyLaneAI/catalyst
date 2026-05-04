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

import math
from dataclasses import dataclass, fields
from typing import Any, Self

import numpy as np

_CODE_REGISTRY: dict[str, tuple[Any, ...]] = {
    "Steane": (
        7,
        1,
        3,
        np.array([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]]),
        np.array([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]]),
    ),
}


@dataclass(frozen=True)
class QecCode:
    """A class to store all relevant information for any [[n, k, d]] stabilizer QEC code.

    Args:
        name (str): A unique identifier of the QEC code.
        n (int): The code's number of QEC physical qubits.
        k (int): The code's number of QEC logical qubits.
        d (int): The code's distance.
    """

    name: str
    n: int
    k: int
    d: int
    x_tanner: np.ndarray
    z_tanner: np.ndarray

    def __str__(self):
        if self.name == "" or str.isspace(self.name):
            name = "<unknown>"
        else:
            name = self.name

        return f"[[{self.n}, {self.k}, {self.d}]] {name}"

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """A builder function that returns a `QecCode` instance from a dictionary.

        Keys in the dictionary that do not have a corresponding field in `QecCode` are dropped.

        Example
        -------

        >>> QecCode.from_dict({'name': "Steane", 'n': 7, 'k': 1, 'd': 3})
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
