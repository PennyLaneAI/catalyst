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
"""Draw one TrotterVibronic step with one mode and one bit per register."""

import matplotlib.pyplot as plt
import pennylane as qp
from pennylane.templates.subroutines.time_evolution.trotter_vibronic import (
    _trotter_vibronic_decomposition,
)

constant = qp.math.array([[[0.4, 0.0], [0.0, -0.4]]])
linear = qp.math.reshape(qp.math.array([[[0.2, 0.0], [0.0, -0.2]]]), (1, 2, 2, 1))
hamiltonian = qp.VibronicHamiltonian(
    constant=constant,
    linear=linear,
    quadratic=qp.math.zeros((1, 2, 2, 1, 1)),
    kinetic=qp.math.reshape(0.3 * qp.math.eye(2), (2, 2, 1, 1)),
)
registers = qp.registers(
    {
        "electronic_wires": 1,
        "vib_wires": 1,
        "cache_wires": 2,
        "coefficient_wires": 1,
        "phase_gradient_wires": 1,
        "work_wires": 4,
    }
)


def circuit():
    _trotter_vibronic_decomposition(
        evolution_time=1.0,
        num_trotter_steps=1,
        hamiltonian=hamiltonian,
        aqft_order=0,
        **registers,
    )


if __name__ == "__main__":
    qp.draw_mpl(circuit, max_length=30)()
    plt.show()
