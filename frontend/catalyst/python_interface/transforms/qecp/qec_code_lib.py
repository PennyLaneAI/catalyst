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

import numpy as np

from dataclasses import dataclass

_CODE_REGISTRY: dict[str, tuple[int, int, int]] = {
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

    def __repr__(self):
        return f"QecCode(name='{self.name}', n={self.n}, k={self.k}, d={self.d})"

    @staticmethod
    def get(name: str) -> "QecCode":
        """A builder function that returns a `QecCode` instance for a supported QEC code.

        Example
        -------

        >>> QecCode.get("Steane")
        QecCode(name='Steane', n=7, k=1, d=3)
        """
        qec_code = QecCode(name, *_CODE_REGISTRY[name])
        if qec_code is None:
            raise KeyError(f"QEC code {name} not found")

        return qec_code
