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
"""Draw AliasSamplingTHC for M=1, N=2, aleph=1."""

import matplotlib.pyplot as plt
import pennylane as qp
from pennylane.templates.subroutines.alias_sampling_thc import _alias_sampling_thc_decomp

sizes = qp.alias_sampling_thc_wires(M=1, N=2, aleph=1)
registers = qp.registers({
    "mu_wires": sizes["mu_wires"], "nu_wires": sizes["nu_wires"],
    "edge_flag": 1, "work_wires": sizes["work_wires"],
})


def circuit():
    _alias_sampling_thc_decomp(
        M=1, N=2, zeta=((1.0,),), t_ell=(0.5,), aleph=1, **registers,
    )


if __name__ == "__main__":
    qp.draw_mpl(circuit)()
    plt.show()
