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
"""Draw the two-orbital OneBodyBlockEncoding with one precision bit."""

import matplotlib.pyplot as plt
import pennylane as qp
from pennylane.templates.subroutines.one_body_block_encoding import _one_body_block_encoding_decomp

registers = qp.registers(qp.one_body_block_encoding_wires(2, alias_sampling_nbits=1))


def circuit():
    _one_body_block_encoding_decomp(
        op_matrix=((1.0, 2.0), (2.0, 1.0)),
        alias_sampling_nbits=1,
        **registers,
    )


if __name__ == "__main__":
    qp.draw_mpl(circuit)()
    plt.show()
