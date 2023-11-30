# Copyright 2023 Xanadu Quantum Technologies Inc.

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
"""
Test that mcmc, num_burnin, and kernel_name are set in MLIR
"""

import pennylane as qml

from catalyst import qjit


@qjit
@qml.qnode(
    qml.device(
        "lightning.qubit",
        wires=1,
        mcmc=True,
        num_burnin=200,
        kernel_name="NonZeroRandom",
        shots=1000,
    )
)
def circuit():
    return qml.state()


# CHECK: 'mcmc': True, 'num_burnin': 200, 'kernel_name': 'NonZeroRandom'
print(circuit.mlir)
