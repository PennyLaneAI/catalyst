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

import pennylane as qp

from catalyst import measure, qjit
from catalyst.passes import merge_rotations

# pylint: disable=line-too-long


@qjit(target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=1))
def circuit(x: float):
    qp.RX(x, wires=0)
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

    dev = qp.device("lightning.qubit", wires=N)

    @qp.set_shots(N)
    @qp.qnode(dev, mcm_method="one-shot")
    def circ():
        return qp.probs()

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

    dev = qp.device("lightning.qubit", wires=1)

    @merge_rotations
    @qp.set_shots(10)
    @qp.qnode(dev, mcm_method="one-shot")
    def circ():
        return qp.sample()

    return circ()


print(test_one_shot_with_passes.mlir)


@qjit(capture=True, target="mlir")
def test_mcm_obs():
    """
    Test generation of mcm observable operation.
    """

    # CHECK:  [[m0:%.+]], {{%.+}} = quantum.measure
    # CHECK:  [[m0_tensor_i1:%.+]] = tensor.from_elements [[m0]] : tensor<i1>
    # CHECK:  [[m0_tensor_i64:%.+]] = stablehlo.convert [[m0_tensor_i1]] : (tensor<i1>) -> tensor<i64>

    # CHECK:  [[m1:%.+]], {{%.+}} = quantum.measure
    # CHECK:  [[m1_tensor_i1:%.+]] = tensor.from_elements [[m1]] : tensor<i1>
    # CHECK:  [[m1_tensor_i64:%.+]] = stablehlo.convert [[m1_tensor_i1]] : (tensor<i1>) -> tensor<i64>

    # CHECK:  [[m0_tensor_i1:%.+]] = stablehlo.convert [[m0_tensor_i64]] : (tensor<i64>) -> tensor<i1>
    # CHECK:  [[m0_i1:%.+]] = tensor.extract [[m0_tensor_i1]][] : tensor<i1>
    # CHECK:  [[expvalObs:%.+]] = quantum.mcmobs [[m0_i1]] : !quantum.obs
    # CHECK:  [[expval:%.+]] = quantum.expval [[expvalObs]] : f64

    # CHECK:  [[m0_tensor_i1:%.+]] = stablehlo.convert [[m0_tensor_i64]] : (tensor<i64>) -> tensor<i1>
    # CHECK:  [[m0_i1:%.+]] = tensor.extract [[m0_tensor_i1]][] : tensor<i1>
    # CHECK:  [[m1_tensor_i1:%.+]] = stablehlo.convert [[m1_tensor_i64]] : (tensor<i64>) -> tensor<i1>
    # CHECK:  [[m1_i1:%.+]] = tensor.extract [[m1_tensor_i1]][] : tensor<i1>
    # CHECK:  [[sampleObs:%.+]] = quantum.mcmobs [[m0_i1]], [[m1_i1]] : !quantum.obs
    # CHECK:  [[sample:%.+]] = quantum.sample [[sampleObs]] : tensor<1000x2xf64>

    # CHECK:  [[m0_tensor_i1:%.+]] = stablehlo.convert [[m0_tensor_i64]] : (tensor<i64>) -> tensor<i1>
    # CHECK:  [[m0_i1:%.+]] = tensor.extract [[m0_tensor_i1]][] : tensor<i1>
    # CHECK:  [[m1_tensor_i1:%.+]] = stablehlo.convert [[m1_tensor_i64]] : (tensor<i64>) -> tensor<i1>
    # CHECK:  [[m1_i1:%.+]] = tensor.extract [[m1_tensor_i1]][] : tensor<i1>
    # CHECK:  [[probsObs:%.+]] = quantum.mcmobs [[m0_i1]], [[m1_i1]] : !quantum.obs
    # CHECK:  [[probs:%.+]] = quantum.probs [[probsObs]] : tensor<4xf64>

    # CHECK:  [[m0_tensor_i1:%.+]] = stablehlo.convert [[m0_tensor_i64]] : (tensor<i64>) -> tensor<i1>
    # CHECK:  [[m0_i1:%.+]] = tensor.extract [[m0_tensor_i1]][] : tensor<i1>
    # CHECK:  [[varObs:%.+]] = quantum.mcmobs [[m0_i1]] : !quantum.obs
    # CHECK:  [[var:%.+]] = quantum.var [[varObs]] : f64

    # CHECK:  [[m0_tensor_i1:%.+]] = stablehlo.convert [[m0_tensor_i64]] : (tensor<i64>) -> tensor<i1>
    # CHECK:  [[m0_i1:%.+]] = tensor.extract [[m0_tensor_i1]][] : tensor<i1>
    # CHECK:  [[m1_tensor_i1:%.+]] = stablehlo.convert [[m1_tensor_i64]] : (tensor<i64>) -> tensor<i1>
    # CHECK:  [[m1_i1:%.+]] = tensor.extract [[m1_tensor_i1]][] : tensor<i1>
    # CHECK:  [[countsObs:%.+]] = quantum.mcmobs [[m0_i1]], [[m1_i1]] : !quantum.obs
    # CHECK:  [[counts:%.+]] = quantum.counts [[countsObs]] : tensor<4xf64>, tensor<4xi64>

    dev = qp.device("lightning.qubit", wires=2)

    @qp.qnode(dev, shots=1000)
    def circ():
        m0 = qp.measure(0)
        m1 = qp.measure(1)
        return (
            qp.expval(m0),
            qp.sample([m0, m1]),
            qp.probs(op=[m0, m1]),
            qp.var(m0),
            qp.counts([m0, m1]),
        )

    return circ()


print(test_mcm_obs.mlir)
