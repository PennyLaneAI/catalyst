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
from catalyst.passes import merge_rotations


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
    # CHECK-DAG: [[ten:%.+]] = arith.constant 10 : index
    # CHECK-DAG: [[zero:%.+]] = arith.constant 0 : index
    # CHECK: scf.for %arg0 = [[zero]] to [[ten]] step [[one]]
    # CHECK: {{%.+}} = catalyst.launch_kernel @module_circ::@circ() : () -> tensor<1024xf64>

    # CHECK: func.func public @circ() -> tensor<1024xf64>
    # CHECK: [[one:%.+]] = arith.constant 1 : i64
    # CHECK: quantum.device shots([[one]])
    # CHECK: quantum.alloc( 10)

    dev = qml.device("lightning.qubit", wires=N)

    @qml.set_shots(N)
    @qml.qnode(dev, mcm_method="one-shot")
    def circ():
        return qml.probs()

    return circ()


test_one_shot_with_static_argnums(10)
print(test_one_shot_with_static_argnums.mlir)


@qjit(target="mlir")
def test_one_shot_with_passes():
    """
    Test pass pipeline is passed correctly to the one shot qnodes.
    """
    # CHECK: func.func public @jit_test_one_shot_with_passes() -> tensor<10x1xi64>
    # CHECK: {{%.+}} = call @one_shot_wrapper() : () -> tensor<10x1xi64>

    # CHECK: func.func private @one_shot_wrapper() -> tensor<10x1xi64>
    # CHECK: scf.for
    # CHECK: {{%.+}} = catalyst.launch_kernel @module_circ::@circ() : () -> tensor<1x1xi64>

    # CHECK: module @module_circ
    # CHECK: transform.named_sequence @__transform_main
    # CHECK: transform.apply_registered_pass "merge-rotations"
    # CHECK: func.func public @circ() -> tensor<1x1xi64>

    dev = qml.device("lightning.qubit", wires=1)

    @merge_rotations
    @qml.set_shots(10)
    @qml.qnode(dev, mcm_method="one-shot")
    def circ():
        return qml.sample()

    return circ()


print(test_one_shot_with_passes.mlir)


@qjit(capture=True, target="mlir")
def test_mcm_obs():
    """
    Test generation of mcm observale operation.
    """

    # CHECK:  [[m0:%.+]], {{%.+}} = quantum.measure
    # CHECK:  [[m1:%.+]], {{%.+}} = quantum.measure
    # CHECK:  [[expvalObs:%.+]] = quantum.mcmobs [[m0]] : !quantum.obs
    # CHECK:  [[expval:%.+]] = quantum.expval [[expvalObs]] : f64
    # CHECK:  [[sampleObs:%.+]] = quantum.mcmobs [[m0]], [[m1]] : !quantum.obs
    # CHECK:  [[sample:%.+]] = quantum.sample [[sampleObs]] : tensor<1000x2xf64>
    # CHECK:  [[probsObs:%.+]] = quantum.mcmobs [[m0]], [[m1]] : !quantum.obs
    # CHECK:  [[probs:%.+]] = quantum.probs [[probsObs]] : tensor<4xf64>
    # CHECK:  [[varObs:%.+]] = quantum.mcmobs [[m0]] : !quantum.obs
    # CHECK:  [[var:%.+]] = quantum.var [[varObs]] : f64
    # CHECK:  [[countsObs:%.+]] = quantum.mcmobs [[m0]], [[m1]] : !quantum.obs
    # CHECK:  [[counts:%.+]] = quantum.counts [[countsObs]] : tensor<4xf64>, tensor<4xi64>

    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(dev, shots=1000)
    def circ():
        m0 = qml.measure(0)
        m1 = qml.measure(1)
        return (
            qml.expval(m0),
            qml.sample([m0, m1]),
            qml.probs(op=[m0, m1]),
            qml.var(m0),
            qml.counts([m0, m1]),
        )

    return circ()


print(test_mcm_obs.mlir)
