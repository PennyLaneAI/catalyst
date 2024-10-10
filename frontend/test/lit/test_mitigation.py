# Copyright 2024 Xanadu Quantum Technologies Inc.

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

import pennylane as qml

from catalyst import mitigate_with_zne, qjit


@qjit(target="mlir")
def mcm_method_with_zne():
    """Test that the dynamic_one_shot works with ZNE."""
    dev = qml.device("lightning.qubit", wires=1, shots=5)

    def circuit():
        return qml.expval(qml.PauliZ(0))

    s = [1, 3]
    g = qml.QNode(circuit, dev, mcm_method="one-shot")
    return mitigate_with_zne(g, scale_factors=s)()


# CHECK: func.func public @jit_mcm_method_with_zne() -> tensor<f64>
# CHECK: mitigation.zne @one_shot_wrapper(%c_0) folding( global) numFolds(%6 : tensor<2xi64>) : (tensor<5xi1>) -> tensor<2xf64>

# CHECK: func.func private @one_shot_wrapper(%arg0: tensor<5xi1>) -> tensor<f64>
# CHECK: call @circuit() : () -> tensor<f64>
# CHECK: scf.for
# CHECK: call @circuit() : () -> tensor<f64>
print(mcm_method_with_zne.mlir)
