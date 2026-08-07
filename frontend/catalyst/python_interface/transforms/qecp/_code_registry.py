# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Raw numpy definitions of the QEC codes exposed by ``qec_code_lib``.

This is a leaf module: it imports only ``numpy`` and stdlib, deliberately
avoiding ``xdsl`` / ``catalyst.python_interface.dialects`` so external tooling
(e.g. the runtime C++ test at
``runtime/tests/Test_transport_steane_LUT_decoder.cpp``) can load it directly
via ``importlib.util.spec_from_file_location`` without triggering the wider
``catalyst`` package import chain.
"""

from typing import Any

import numpy as np

_CODE_REGISTRY: dict[str, tuple[Any, ...]] = {
    # the indices/ordering for the operators and encodings in the Steane code are those used
    # in https://arxiv.org/pdf/2107.07505
    "Steane": (
        7,
        1,
        3,
        #### Stabilizers ####
        np.array([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]]),
        np.array([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1]]),
        #### Transversal gates ####
        {
            # Keys need to match the names of the corresponding qecl.gate gates; if any adjoint
            # gates are supported, they should be included as a separate entry with key
            # "gatename_adj". Values are a tuple of gate string identifiers specifying the qecp ops
            # the codeblock will need to be refactored for k>1
            "x": ("I", "I", "I", "I", "X", "X", "X"),
            "y": ("I", "I", "I", "I", "Y", "Y", "Y"),
            "z": ("I", "I", "I", "I", "Z", "Z", "Z"),
            "hadamard": ("H", "H", "H", "H", "H", "H", "H"),
            "s": ("Sa", "Sa", "Sa", "Sa", "Sa", "Sa", "Sa"),
            "s_adj": ("S", "S", "S", "S", "S", "S", "S"),
        },
        {
            "cnot": "CNOT",
        },
        #### Unitary encoding circuit ####
        {
            # ops (in the form of a gate string identifier and the indices of the codeblock
            # it should be applied on) defining a transporter encoding circuit, i.e.
            # one that maps an input to the logical version of that input, rather
            # than just encoding logical 0
            "ops": [
                ("H", [1]),
                ("H", [2]),
                ("H", [3]),
                ("CNOT", [1, 0]),
                ("CNOT", [2, 4]),
                ("CNOT", [6, 5]),
                ("CNOT", [2, 0]),
                ("CNOT", [3, 5]),
                ("CNOT", [6, 4]),
                ("CNOT", [2, 6]),
                ("CNOT", [3, 4]),
                ("CNOT", [1, 5]),
                ("CNOT", [1, 6]),
                ("CNOT", [3, 0]),
            ],
            # The state_prep_index is the index of the physical qubit that the state is
            # injected on (i.e. for a magic state, -H-T is applied here pre-encoding).
            # Must be consistent with the qubit treated as the encoding "input" by the
            # cnot_indices ordering above. See https://arxiv.org/pdf/2107.07505 (Fig 10)
            "state_prep_index": 6,
        },
    ),
    "Shor913": (
        # see Steane code for general comments on the inputs to define the code
        9,
        1,
        3,
        #### Stabilizers ####
        # from error correction zoo, https://errorcorrectionzoo.org/c/shor_nine
        np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 1]]),
        np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1],
            ]
        ),
        #### Transversal gates ####
        # X: physical Z on a single qubit from each set of 3, make +|111> into -|111> and vice-versa
        # Z: X-flip on the all the bits on one set of 3: doesn't modify |0>, generates overall -1
        #    sign for |1>
        # Y: Y = iXZ (we ignore global phase so can use Y ~ XZ)
        # CNOT is transversal for all CSS codes
        # There are no transversal Hadamard or S gates for this code
        {
            "x": ("Z", "I", "I", "Z", "I", "I", "Z", "I", "I"),
            "y": ("Y", "X", "X", "Z", "I", "I", "Z", "I", "I"),
            "z": ("X", "X", "X", "I", "I", "I", "I", "I", "I"),
        },
        {"cnot": "CNOT"},
        #### Unitary encoding circuit ####
        # References:
        #   [1] P. Shor (1995), Scheme for reducing decoherence in quantum computer memory,
        #         Phys. Rev. A 52, R2493. https://doi.org/10.1103/PhysRevA.52.R2493.
        #   [2] O. Khalifa, et al. (2021), Digital System Design for Quantum Error Correction Codes,
        #         Contrast Media & Molecular Imaging, 1101911. https://doi.org/10.1155/2021/1101911.
        #         (Open Access)
        {
            "ops": [
                ("CNOT", [0, 3]),
                ("CNOT", [0, 6]),
                ("H", [0]),
                ("H", [3]),
                ("H", [6]),
                ("CNOT", [0, 1]),
                ("CNOT", [0, 2]),
                ("CNOT", [3, 4]),
                ("CNOT", [3, 5]),
                ("CNOT", [6, 7]),
                ("CNOT", [6, 8]),
            ],
            "state_prep_index": 0,
        },
    ),
}
