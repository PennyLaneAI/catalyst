# Copyright 2025 Xanadu Quantum Technologies Inc.

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

# pylint: disable=missing-function-docstring

import pennylane as qml

from catalyst import qjit, switch

# ----------------------------------------------------------------------------------------------- #


# CHECK-LABEL: public @jit_no_branches
@qjit(target="mlir")
def no_branches(i: int):

    # cast index and use as arg for switch
    # CHECK:    [[i:%[a-zA-Z0-9_]+]] = arith.index_cast %{{[a-zA-Z0-9_]+}} : i64 to index
    # CHECK:    %{{[a-zA-Z0-9_]+}} = scf.index_switch [[i]] ->
    @switch(i)
    def my_switch():
        return 5

    # CHECK:    default {

    # CHECK:    return
    return my_switch()


print(no_branches.mlir)

# ----------------------------------------------------------------------------------------------- #


# CHECK-LABEL: public @jit_classical_circuit
@qjit(target="mlir")
def classical_circuit(i: int):

    # cast index and use as arg for switch
    # CHECK:    [[i:%[a-zA-Z0-9_]+]] = arith.index_cast %{{[a-zA-Z0-9_]+}} : i64 to index
    # CHECK:    %{{[a-zA-Z0-9_]+}} = scf.index_switch [[i]] ->
    @switch(i)
    def my_switch():
        return 5

    # CHECK-DAG:    case 11 {
    @my_switch.branch(11)
    def my_branch():
        return 9

    # CHECK-DAG:    case -41 {
    @my_switch.branch(-41)
    def my_branch():
        return 4

    # CHECK-DAG:    case 2048 {
    @my_switch.branch(2048)
    def my_branch():
        return 4

    # CHECK-DAG:    case -4444 {
    @my_switch.branch(-4444)
    def my_branch():
        return 4

    # default case is always last
    # CHECK:    default {

    # CHECK:    return
    return my_switch()


print(classical_circuit.mlir)

# ----------------------------------------------------------------------------------------------- #


# CHECK-LABEL: public @quantum_circuit
@qjit(target="mlir")

# CHECK: quantum.device
@qml.qnode(qml.device("lightning.qubit", wires=1))
def quantum_circuit(i: int):

    # cast index and use as arg for switch
    # CHECK:    [[i:%[a-zA-Z0-9_]+]] = arith.index_cast [[in:%[a-zA-Z0-9_]+]] : i64 to index
    # CHECK:    %{{[a-zA-Z0-9_]+}} = scf.index_switch [[i]] -> !quantum.reg
    @switch(i)
    def my_switch():
        qml.X(0)

    # CHECK-DAG: case -3 {
    @my_switch.branch(-3)
    def my_branch():
        qml.Y(0)

    # CHECK-DAG: case 0 {
    @my_switch.branch(0)
    def my_branch():
        qml.Z(0)

    # default case is always last
    # CHECK: default {

    my_switch()
    # CHECK: return

    return qml.probs()


print(quantum_circuit.mlir)
