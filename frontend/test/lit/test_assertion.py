# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the catalyst assert feature."""

# RUN: %PYTHON %s | FileCheck %s

import pennylane as qml

from catalyst import debug_assert, measure, qjit


@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit(x: float):
    """Test a simple assert example."""

    qml.RX(x, wires=0)
    # CHECK: tensor.extract {{%.+}}[] : tensor<i1>
    # CHECK: "catalyst.assert"({{%.+}}) <{error = "x less than 5.0"}> : (i1) -> ()
    debug_assert(x > 5.0, "x less than 5.0")
    m = measure(wires=0)
    return m


print(circuit.mlir)
