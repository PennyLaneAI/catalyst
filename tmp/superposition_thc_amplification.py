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
"""Draw SuperpositionTHC for M=1, N=2."""

import matplotlib.pyplot as plt
import pennylane as qp
from pennylane.templates.subroutines.superposition_thc import _superposition_thc

registers = qp.registers({"mu_wires": 1, "nu_wires": 1, "work_wires": 8})


def circuit():
    _superposition_thc(M=1, N=2, **registers)


if __name__ == "__main__":
    qp.draw_mpl(circuit)()
    plt.show()
