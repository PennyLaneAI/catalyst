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

# RUN: %PYTHON %s | FileCheck %s

import pennylane as qml

from catalyst import measure, qjit


@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit(x: float):
    qml.RX(x, wires=0)
    # CHECK: {{%.+}}, {{%.+}} = quantum.measure {{%.+}}
    m = measure(wires=0)
    return m


print(circuit.mlir)


@qjit(static_argnums=0)
def test_one_shot_with_static_argnums(N):
    """
    Test static argnums is passed correctly to the one shot qnodes.
    """
    # CHECK: func.func public @jit_test_one_shot_with_static_argnums() -> tensor<1024xf64>
    # CHECK: {{%.+}} = call @one_shot_wrapper() : () -> tensor<1024xf64>

    # CHECK: func.func private @one_shot_wrapper() -> tensor<1024xf64>
    # CHECK-DAG: [[one:%.+]] = arith.constant 1 : index
    # CHECK-DAG: [[eleven:%.+]] = arith.constant 11 : index
    # CHECK-DAG: [[zero:%.+]] = arith.constant 0 : index
    # CHECK: scf.for %arg0 = [[zero]] to [[eleven]] step [[one]]
    # CHECK: {{%.+}} = catalyst.launch_kernel @module_circ::@circ() : () -> tensor<1024xf64>

    # CHECK: func.func public @circ() -> tensor<1024xf64>
    # CHECK: [[one:%.+]] = arith.constant 1 : i64
    # CHECK: quantum.device shots([[one]])
    # CHECK: quantum.alloc( 10)

    dev = qml.device("lightning.qubit", wires=N)

    @qml.set_shots(N + 1)
    @qml.qnode(dev, mcm_method="one-shot")
    def circ():
        return qml.probs()

    return circ()


test_one_shot_with_static_argnums(10)
print(test_one_shot_with_static_argnums.mlir)
