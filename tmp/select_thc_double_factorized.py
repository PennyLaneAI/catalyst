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
"""Draw SelectTHC for M=1, N=4, beth=1, one batch."""

import matplotlib.pyplot as plt
import pennylane as qp
from pennylane.templates.subroutines.select_thc import _select_thc_decomp

registers = qp.registers(qp.select_thc_wires(M=1, N=4, beth=1))
registers = {
    name: qp.wires.Wires(
        [r"$\mu$", r"$\nu$"]
        if name == "index_wires"
        else [rf"$\mathrm{{{name.removesuffix('_wires')}}}_{{{i}}}$" for i in range(len(wires))]
    )
    for name, wires in registers.items()
}
wire_order = qp.wires.Wires.all_wires(
    [
        registers[name]
        for name in (
            "index_wires",
            "flag_wires",
            "gradient_wires",
            "work_wires",
            "system_wires",
        )
    ]
)
gate_set = {
    qp.QROM,
    "Adjoint(QROM)",
    qp.CSWAP,
    "C(SWAP)",
    qp.Hadamard,
    qp.CNOT,
    qp.S,
    "Adjoint(S)",
    qp.Z,
    "C(PauliZ)",
    "C(MultiX)",
    qp.SemiAdder,
    "Adjoint(SemiAdder)",
    qp.X,
    "Adjoint(PauliX)",
}

qp.decomposition.enable_graph()


@qp.transforms._tape_decompose(
    gate_set=gate_set,
    fixed_decomps={qp.SelectTHC: _select_thc_decomp},
)
@qp.qnode(qp.device("null.qubit"))
def circuit():
    qp.SelectTHC(
        chi=((0.6, 0.8),),
        t_eigenvectors=((0.8, -0.6), (0.6, 0.8)),
        beth=1,
        num_batches=1,
        **registers,
    )
    return qp.state()


if __name__ == "__main__":
    qp.draw_mpl(circuit, level="user", wire_order=wire_order, style="pennylane")()
    plt.show()
