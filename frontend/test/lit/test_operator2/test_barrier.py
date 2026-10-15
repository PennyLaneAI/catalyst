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

"""
Test that qp.Barrier lowers to qref.barrier under capture.

# RUN: %PYTHON %s | FileCheck %s
"""

# pylint: disable = missing-function-docstring

import pennylane as qp


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=2))
def circuit():
    # CHECK-LABEL: func.func public @circuit
    qp.Hadamard(0)
    # CHECK: qref.barrier {{%.+}}, {{%.+}} : !qref.bit, !qref.bit
    qp.Barrier(wires=[0, 1])
    qp.CNOT(wires=[0, 1])
    return qp.state()


print(circuit.mlir)
