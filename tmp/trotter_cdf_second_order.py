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
"""Draw one second-order TrotterCDF step with two orbitals and one fragment."""

import matplotlib.pyplot as plt
import pennylane as qp
from pennylane.templates.subroutines.time_evolution.trotter_cdf import _trotter_cdf_decomposition

hamiltonian = qp.CDFHamiltonian(
    core_tensors=qp.math.array([[[0.7, 0.2], [0.2, -0.4]], [[0.3, 0.1], [0.1, 0.6]]]),
    leaf_tensors=qp.math.array([[[0.8, -0.6], [0.6, 0.8]], [[0.6, -0.8], [0.8, 0.6]]]),
    nuc_constant=0.5,
)


def circuit():
    _trotter_cdf_decomposition(
        evolution_time=1.0,
        num_trotter_steps=1,
        hamiltonian=hamiltonian,
        wires=qp.wires.Wires(range(4)),
        double_phase=False,
        work_wires=(),
    )


if __name__ == "__main__":
    qp.draw_mpl(circuit)()
    plt.show()
