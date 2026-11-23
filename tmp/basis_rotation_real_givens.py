# Copyright 2026 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Draw a three-wire real BasisRotation with determinant -1."""

import matplotlib.pyplot as plt
import pennylane as qp
from pennylane.templates.subroutines.qchem.basis_rotation import _real_basis_rotation_decomp

matrix = qp.math.array([[1, 2, 3], [4, 2, 1], [2, 1, 4]], dtype=float)
unitary, _ = qp.math.linalg.qr(matrix)
unitary = unitary @ qp.math.diag(qp.math.array([-qp.math.linalg.det(unitary), 1, 1]))


def circuit():
    _real_basis_rotation_decomp(unitary, wires=qp.wires.Wires(range(3)))


if __name__ == "__main__":
    qp.draw_mpl(circuit, decimals=2)()
    plt.show()
