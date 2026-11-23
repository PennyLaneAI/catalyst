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
"""Draw QubitizationTHC for M=1, N=2, aleph=beth=1."""

import matplotlib.pyplot as plt
import pennylane as qp
from pennylane.templates.subroutines.qubitization_thc import _qubitization_thc_decomp

registers = qp.registers(qp.qubitization_thc_wires(M=1, N=2, aleph=1, beth=1))
for name, wires in registers.items():
    if name == "prep_garbage_wires":
        labels = [rf"$\mathrm{{garbage}}_{{{i}}}$" for i in range(len(wires) - 2)] + [
            rf"$\mathrm{{spin}}_{{{i}}}$" for i in range(2)
        ]
    else:
        prefix = name.removesuffix("_wires")
        labels = [rf"$\mathrm{{{prefix}}}_{{{i}}}$" for i in range(len(wires))]
    registers[name] = qp.wires.Wires(labels)
wire_order = qp.wires.Wires.all_wires(list(registers.values()))

gate_set = {
    qp.SuperpositionTHC,
    qp.AliasSamplingTHC,
    qp.SelectTHC,
    "Adjoint(SuperpositionTHC)",
    "Adjoint(AliasSamplingTHC)",
    qp.Hadamard,
    qp.Z,
    qp.FlipSign,
    qp.GlobalPhase,
}

qp.decomposition.enable_graph()


@qp.transforms._tape_decompose(
    gate_set=gate_set,
    fixed_decomps={qp.QubitizationTHC: _qubitization_thc_decomp},
)
@qp.qnode(qp.device("null.qubit"))
def circuit():
    qp.QubitizationTHC(
        zeta=((1.0,),),
        t_ell=(0.5,),
        chi=((1.0,),),
        t_eigenvectors=((1.0,),),
        aleph=1,
        beth=1,
        **registers,
    )
    return qp.state()


if __name__ == "__main__":
    tape = qp.workflow.construct_tape(circuit, level="user")()
    qp.drawer.tape_mpl(
        tape.copy(measurements=[]),
        wire_order=wire_order,
        show_all_wires=True,
        style="pennylane",
    )
    plt.show()
