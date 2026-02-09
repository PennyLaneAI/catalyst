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

"""Lit tests for subroutines with program capture."""

from functools import partial

import pennylane as qml
from jax import numpy as jnp


def test_basic_subroutine():
    """Test the most simple subroutine."""

    @qml.templates.Subroutine
    def f(x, wires):
        qml.RX(x, wires)

    @qml.qjit(capture=True, target="mlir")
    @qml.qnode(qml.device("null.qubit", wires=1))
    # CHECK: module @circuit
    def circuit(x):
        # CHECK: [[QREG:%.+]] = quantum.alloc
        # CHECK: [[QREG_1:%.+]] = call @f([[QREG]], %arg0, %1) : (!quantum.reg, tensor<f64>, tensor<1xi64>) -> !quantum.reg

        # CHECK: quantum.compbasis qreg [[QREG_1]] : !quantum.obs
        f(x, 0)
        return qml.probs()

    # CHECK: func.func private @f(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg
    # CHECK: [[QUBIT_1:%.+]] = quantum.custom "RX"
    # CHECK: [[REG_1:%.+]] = quantum.insert
    # CHECK-NEXT: return [[REG_1]] : !quantum.reg
    circuit(0.5)
    print(circuit.mlir)


test_basic_subroutine()


def test_multiple_metadata():
    """Test a subroutine with metadata becomes multiple functions.

    Each metadata should get its own function.
    """

    @partial(qml.templates.Subroutine, static_argnames="metadata")
    def f(wires, metadata):
        if metadata == "X":
            qml.X(wires)
        elif metadata == "Y":
            qml.Y(wires)
        else:
            qml.Z(wires)

    @qml.qjit(capture=True, target="mlir")
    @qml.qnode(qml.device("null.qubit", wires=1))
    # CHECK: module @circuit
    def circuit():
        # CHECK: [[QREG:%.+]] = quantum.alloc
        # CHECK: [[QREG_1:%.+]] = call @f([[QREG]], %1) : (!quantum.reg, tensor<1xi64>) -> !quantum.reg
        # CHECK: [[QREG_2:%.+]] = call @f_0([[QREG_1]], %3) : (!quantum.reg, tensor<1xi64>) -> !quantum.reg
        # CHECK: [[QREG_3:%.+]] = call @f_1([[QREG_2]], %5) : (!quantum.reg, tensor<1xi64>) -> !quantum.reg
        # CHECK: [[QREG_4:%.+]] = call @f([[QREG_3]], %7) : (!quantum.reg, tensor<1xi64>) -> !quantum.reg

        # CHECK: quantum.compbasis qreg [[QREG_4]] : !quantum.obs
        f(0, "X")
        f(0, "Y")
        f(0, "Z")
        f(0, "X")  # check reusing the first call to the function
        return qml.probs()

    # CHECK: func.func private @f(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg
    # CHECK: [[QUBIT_1:%.+]] = quantum.custom "PauliX"
    # CHECK: [[REG_1:%.+]] = quantum.insert
    # CHECK-NEXT: return [[REG_1]] : !quantum.reg

    # CHECK: func.func private @f_0(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg
    # CHECK: [[QUBIT_1:%.+]] = quantum.custom "PauliY"
    # CHECK: [[REG_1:%.+]] = quantum.insert
    # CHECK-NEXT: return [[REG_1]] : !quantum.reg

    # CHECK: func.func private @f_1(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg
    # CHECK: [[QUBIT_1:%.+]] = quantum.custom "PauliZ"
    # CHECK: [[REG_1:%.+]] = quantum.insert
    # CHECK-NEXT: return [[REG_1]] : !quantum.reg
    print(circuit.mlir)


test_multiple_metadata()


def test_different_shapes():
    """Test a subroutine with different shape inputs get their own function."""

    @qml.templates.Subroutine
    def my_subroutine(data, wires):
        @qml.for_loop(data.shape[0])
        def loop(i):
            qml.RX(data[i], wires[i])

        loop()

    @qml.qjit(capture=True, target="mlir")
    @qml.qnode(qml.device("null.qubit", wires=1))
    # CHECK: module @circuit
    def circuit():
        # CHECK: [[QREG:%.+]] = quantum.alloc
        # CHECK: [[QREG_1:%.+]] = call @my_subroutine([[QREG]], %arg0, %4) : (!quantum.reg, tensor<3xf64>, tensor<3xi64>) -> !quantum.reg
        # CHECK: [[QREG_2:%.+]] = call @my_subroutine([[QREG_1]], %arg1, %9) : (!quantum.reg, tensor<3xf64>, tensor<3xi64>) -> !quantum.reg
        # CHECK: [[QREG_3:%.+]] = call @my_subroutine_0([[QREG_2]], %arg2, %14) : (!quantum.reg, tensor<2xf64>, tensor<3xi64>) -> !quantum.reg

        # CHECK: quantum.compbasis qreg [[QREG_3]] : !quantum.obs
        my_subroutine(jnp.array([0.0, 0.1, 0.2]), [0, 1, 2])
        my_subroutine(jnp.array([0.0, 0.1, 0.2]), [0, 1, 2])
        my_subroutine(jnp.array([0.5, 1.2]), [0, 1, 2])
        return qml.probs()

    # CHECK: func.func private @my_subroutine(%arg0: !quantum.reg, %arg1: tensor<3xf64>, %arg2: tensor<3xi64>) -> !quantum.reg
    # CHECK: arith.constant 3 : index
    # CHECK: scf.for
    # CHECK: [[QUBIT_1:%.+]] = quantum.custom "RX"

    # CHECK: func.func private @my_subroutine_0(%arg0: !quantum.reg, %arg1: tensor<2xf64>, %arg2: tensor<3xi64>) -> !quantum.reg
    # CHECK: arith.constant 2 : index
    # CHECK: scf.for
    # CHECK: [[QUBIT_1:%.+]] = quantum.custom "RX"

    print(circuit.mlir)


test_different_shapes()
