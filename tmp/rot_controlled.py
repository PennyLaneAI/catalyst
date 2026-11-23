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
"""Draw the Rot decomposition with two positive controls."""

import matplotlib.pyplot as plt
import pennylane as qp
from pennylane.ops.qubit.parametric_ops_single_qubit import _controlled_rot_decomp


def circuit():
    with qp.QueuingManager.stop_recording():
        base = qp.Rot(0.2, 0.4, 0.6, wires=2)
    _controlled_rot_decomp(
        base,
        control_wires=qp.wires.Wires([0, 1]),
        control_values=[True, True],
        work_wires=qp.wires.Wires([]),
        work_wire_type="zeroed",
    )


if __name__ == "__main__":
    qp.draw_mpl(circuit, wire_order=[0, 1, 2], decimals=2)()
    plt.show()
