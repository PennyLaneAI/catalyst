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

from catalyst import qjit

# pylint: disable=line-too-long


# Non-root nodes have internal linkage.
# CHECK-DAG: func.func public @qnode{{.*}} {diff_method = "parameter-shift", llvm.linkage = #llvm.linkage<internal>, qnode} {
@qml.qnode(qml.device("lightning.qubit", wires=2), diff_method="parameter-shift")
def qnode(x):
    qml.RX(x, wires=0)
    return qml.state()


@qjit(target="mlir")
# The entry point has no internal linkage.
# CHECK-DAG: func.func public @jit_workload(%arg0: tensor<f64>) -> tensor<4xcomplex<f64>> attributes {llvm.emit_c_interface} {
def workload(x: float):
    y = x * qml.numpy.pi
    return qnode(y)


print(workload.mlir)
