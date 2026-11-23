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
"""Draw one second-order TrotterCGF step with two modes and three modals."""

import matplotlib.pyplot as plt
import pennylane as qp
from pennylane.templates.subroutines.time_evolution.trotter_cgf import _trotter_cgf_decomposition

matrix = qp.math.array([[1, 2, 3], [4, 2, 1], [2, 1, 4]], dtype=float)
leaf, _ = qp.math.linalg.qr(matrix)
leaves = qp.math.stack([qp.math.stack([leaf, leaf]), qp.math.stack([leaf.T, leaf.T])])
hamiltonian = qp.CGFHamiltonian(
    core_tensors=qp.math.reshape(qp.math.arange(72, dtype=float) / 100 + 0.1, (2, 2, 2, 3, 3)),
    leaf_tensors=leaves, nuc_constant=0.5,
)


def circuit():
    _trotter_cgf_decomposition(
        evolution_time=1.0, num_trotter_steps=1, hamiltonian=hamiltonian,
        wires=qp.wires.Wires(range(6)), double_phase=False,
    )


if __name__ == "__main__":
    qp.draw_mpl(circuit, max_length=30)()
    plt.show()
