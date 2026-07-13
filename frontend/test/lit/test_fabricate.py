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

# RUN: %PYTHON %s | FileCheck %s

"""Test fabricate lowering to pbc.fabricate."""

import pennylane as qp
from pennylane.transforms import to_ppr

from catalyst import qjit


def test_fabricate_lowering():
    """Test that qp.fabricate is lowered to pbc.fabricate."""
    qp.capture.enable()
    dev = qp.device("null.qubit", wires=2)

    pipeline = [
        (
            "pipe",
            [
                "canonicalize",
                "verify-no-quantum-use-after-free",
                "convert-to-value-semantics",
                "canonicalize",
            ],
        )
    ]

    @qjit(pipelines=pipeline, target="mlir")
    @qp.qnode(device=dev)
    def circuit():
        magic = qp.fabricate("magic")
        qp.pauli_measure("ZZ", wires=[0, magic])
        qp.deallocate(magic)
        return qp.expval(qp.Z(0))

    # CHECK: pbc.fabricate magic
    # CHECK: pbc.ppm ["Z", "Z"]
    print(circuit.mlir_opt)
    qp.capture.disable()


test_fabricate_lowering()
