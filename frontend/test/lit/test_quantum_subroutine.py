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

"""Lit tests for quantum subroutines"""

import jax
import pennylane as qml

from catalyst.jax_primitives import subroutine

# pylint: disable=line-too-long


def test_subroutine_classical():
    """Test that a subroutine is just jax.jit
    when used in a classical setting.

    A desirable behaviour may be to error.
    """

    @subroutine
    def add_one(x):
        return x + 1

    qml.capture.enable()

    @qml.qjit
    # CHECK: module @main
    def main():
        # CHECK: %{{.*}} = call @add_one(%{{.*}}) : (tensor<i64>) -> tensor<i64>
        return add_one(0)

    print(main.mlir)
    qml.capture.disable()


test_subroutine_classical()


def test_quantum_subroutine_identity_restore_wires():
    """Test that a subroutine receives a register as a parameter and returns a register"""

    @subroutine
    def identity(): ...

    qml.capture.enable()

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: module @main
    def main():
        qml.Hadamard(wires=[0])
        # CHECK: [[QUBIT:%.+]] = quantum.custom "Hadamard"
        # CHECK: [[QREG:%.+]] = quantum.insert {{.*}}
        # CHECK: [[QREG_1:%.+]] = call @identity([[QREG]]) : (!quantum.reg) -> !quantum.reg
        # CHECK: [[QUBIT_1:%.+]] = quantum.extract [[QREG_1]][ 0]
        # CHECK: quantum.custom "Hadamard"() [[QUBIT_1]]
        identity()
        qml.Hadamard(wires=[0])
        return qml.probs()

    # CHECK: func.func private @identity([[REG:%.+]]: !quantum.reg) -> !quantum.reg
    # CHECK-NEXT: return [[REG]] : !quantum.reg

    print(main.mlir)
    qml.capture.disable()


test_quantum_subroutine_identity_restore_wires()


def test_quantum_subroutine_identity():
    """Test that a subroutine receives a register as a parameter and returns a register"""

    @subroutine
    def identity(): ...

    qml.capture.enable()

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: module @main
    def main():
        # CHECK: [[QREG:%.+]] = quantum.alloc
        # CHECK: [[QREG_1:%.+]] = call @identity([[QREG]]) : (!quantum.reg) -> !quantum.reg
        # CHECK: quantum.compbasis qreg [[QREG_1]] : !quantum.obs
        identity()
        return qml.probs()

    # CHECK: func.func private @identity([[REG:%.+]]: !quantum.reg) -> !quantum.reg
    # CHECK-NEXT: return [[REG]] : !quantum.reg

    print(main.mlir)
    qml.capture.disable()


test_quantum_subroutine_identity()


def test_quantum_subroutine_wire_param():
    """Pass a parameter that is a wire/integer"""

    @subroutine
    def Hadamard0(wire):
        qml.Hadamard(wire)

    qml.capture.enable()

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: module @subroutine_test
    def subroutine_test(c: int):
        # CHECK: func.func public @subroutine_test([[ARG0:%.+]]
        # CHECK: [[QREG:%.+]] = quantum.alloc
        # CHECK: [[QREG_1:%.+]] = call @Hadamard0([[QREG]], [[ARG0:%.+]]) : (!quantum.reg, tensor<i64>) -> !quantum.reg
        # CHECK: quantum.compbasis qreg [[QREG_1]] : !quantum.obs
        Hadamard0(c)
        return qml.probs()

    # CHECK: func.func private @Hadamard0([[REG:%.+]]: !quantum.reg, [[WIRE_IDX_TENSOR:%.+]]: tensor<i64>) -> !quantum.reg
    # CHECK-NEXT: [[WIRE_IDX:%.+]] = tensor.extract [[WIRE_IDX_TENSOR]][] : tensor<i64>
    # CHECK-NEXT: [[QUBIT:%.+]] = quantum.extract [[REG]][[[WIRE_IDX]]] : !quantum.reg -> !quantum.bit
    # CHECK-NEXT: [[QUBIT_1:%.+]] = quantum.custom "Hadamard"() [[QUBIT]] : !quantum.bit
    # CHECK-NEXT: [[WIRE_IDX:%.+]] = tensor.extract [[WIRE_IDX_TENSOR]][] : tensor<i64>
    # CHECK-NEXT: [[REG_1:%.+]] = quantum.insert [[REG]][[[WIRE_IDX]]], [[QUBIT_1]] : !quantum.reg, !quantum.bit
    # CHECK-NEXT: return [[REG_1]] : !quantum.reg

    print(subroutine_test.mlir)

    qml.capture.disable()


test_quantum_subroutine_wire_param()


def test_quantum_subroutine_gate_param_param():
    """Test passing a regular parameter"""

    @subroutine
    def RX_on_wire_0(param):
        qml.RX(param, wires=[0])

    qml.capture.enable()

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: module @subroutine_test_2
    def subroutine_test_2():
        # CHECK-DAG: [[CST:%.+]] = stablehlo.constant dense<3.140000e+00>
        # CHECK-DAG: [[QREG:%.+]] = quantum.alloc
        # CHECK: [[QREG_1:%.+]] = call @RX_on_wire_0([[QREG]], [[CST]]) : (!quantum.reg, tensor<f64>) -> !quantum.reg
        # CHECK: quantum.compbasis qreg [[QREG_1]] : !quantum.obs
        RX_on_wire_0(3.14)
        return qml.probs()

    # CHECK: func.func private @RX_on_wire_0([[REG:%.+]]: !quantum.reg, [[PARAM_TENSOR:%.+]]: tensor<f64>) -> !quantum.reg
    # CHECK-NEXT: [[QUBIT:%.+]] = quantum.extract [[REG]][ 0] : !quantum.reg -> !quantum.bit
    # CHECK-NEXT: [[PARAM:%.+]] = tensor.extract [[PARAM_TENSOR]][] : tensor<f64>
    # CHECK-NEXT: [[QUBIT_1:%.+]] = quantum.custom "RX"([[PARAM]]) [[QUBIT]] : !quantum.bit
    # CHECK-NEXT: [[REG_1:%.+]] = quantum.insert [[REG]][ 0], [[QUBIT_1]] : !quantum.reg, !quantum.bit
    # CHECK-NEXT: return [[REG_1]] : !quantum.reg
    print(subroutine_test_2.mlir)

    qml.capture.disable()


test_quantum_subroutine_gate_param_param()


def test_quantum_subroutine_with_control_flow():
    """Test control flow inside the subroutine"""

    qml.capture.enable()

    @subroutine
    def conditional_RX(param: float):

        def true_path():
            qml.RX(param, wires=[0])

        def false_path(): ...

        qml.cond(param != 0.0, true_path, false_path)()

    @qml.qjit(autograph=False)
    @qml.qnode(qml.device("lightning.qubit", wires=1), autograph=False)
    # CHECK: module @subroutine_test_3
    def subroutine_test_3():
        # CHECK-DAG: [[CST:%.+]] = stablehlo.constant dense<3.140000e+00>
        # CHECK-DAG: [[QREG:%.+]] = quantum.alloc
        # CHECK: [[QREG_1:%.+]] = call @conditional_RX([[QREG]], [[CST]]) : (!quantum.reg, tensor<f64>) -> !quantum.reg
        # CHECK: quantum.compbasis qreg [[QREG_1]] : !quantum.obs
        conditional_RX(3.14)
        return qml.probs()

    # CHECK: func.func private @conditional_RX([[QREG:%.+]]: !quantum.reg, [[PARAM_TENSOR:%.+]]: tensor<f64>)
    # CHECK-NEXT: [[ZERO:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    # CHECK-NEXT: [[COND_TENSOR:%.+]] = stablehlo.compare  NE, [[PARAM_TENSOR]], [[ZERO]],  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    # CHECK-NEXT: [[COND:%.+]] = tensor.extract [[COND_TENSOR]][] : tensor<i1>
    # CHECK-NEXT: [[RETVAL:%.+]] = scf.if [[COND]]
    # CHECK-DAG:        [[QUBIT:%.+]] = quantum.extract [[QREG]][ 0] : !quantum.reg -> !quantum.bit
    # CHECK-DAG:        [[PARAM:%.+]] = tensor.extract [[PARAM_TENSOR]][] : tensor<f64>
    # CHECK:            [[QUBIT_0:%.+]] = quantum.custom "RX"([[PARAM]]) [[QUBIT]] : !quantum.bit
    # CHECK-NEXT:       [[QREG_0:%.+]] = quantum.insert [[QREG]][ 0], [[QUBIT_0]] : !quantum.reg, !quantum.bit
    # CHECK-NEXT:       scf.yield [[QREG_0]] : !quantum.reg
    # CHECK-NEXT: else
    # CHECK:            scf.yield [[QREG]] : !quantum.reg
    # CHECK:      return [[RETVAL]]
    print(subroutine_test_3.mlir)
    qml.capture.disable()


test_quantum_subroutine_with_control_flow()


def test_nested_subroutine_call():
    """Test nested subroutine call"""

    qml.capture.enable()

    @subroutine
    def Hadamard_subroutine():
        qml.Hadamard(wires=[0])

    @subroutine
    def Hadamard_caller():
        Hadamard_subroutine()

    @qml.qjit(autograph=False)
    @qml.qnode(qml.device("lightning.qubit", wires=1), autograph=False)
    # CHECK: module @subroutine_test_4
    def subroutine_test_4():
        # CHECK: [[QREG:%.+]] = quantum.alloc
        # CHECK: [[QREG_1:%.+]] = call @Hadamard_caller([[QREG]]) : (!quantum.reg) -> !quantum.reg
        # CHECK: quantum.compbasis qreg [[QREG_1]] : !quantum.obs
        Hadamard_caller()
        return qml.probs()

    # CHECK: func.func private @Hadamard_caller([[QREG:%.+]]: !quantum.reg) -> !quantum.reg
    # CHECK-NEXT: [[QREG_1:%.+]] = call @Hadamard_subroutine([[QREG]]) : (!quantum.reg) -> !quantum.reg
    # CHECK-NEXT: return [[QREG_1]]

    # CHECK: func.func private @Hadamard_subroutine([[QREG:%.+]]: !quantum.reg) -> !quantum.reg
    # CHECK-NEXT: [[QUBIT:%.+]] = quantum.extract [[QREG]][ 0] : !quantum.reg -> !quantum.bit
    # CHECK-NEXT: [[QUBIT_1:%.+]] = quantum.custom "Hadamard"() [[QUBIT]] : !quantum.bit
    # CHECK-NEXT: [[QREG_1:%.+]] = quantum.insert [[QREG]][ 0], [[QUBIT_1]] : !quantum.reg, !quantum.bit
    # CHECK-NEXT: return [[QREG_1]] : !quantum.reg
    print(subroutine_test_4.mlir)
    qml.capture.disable()


test_nested_subroutine_call()


def test_two_callsites():
    """Test that two calls won't give multiple definitions
    in the classical setting"""

    qml.capture.enable()

    @subroutine
    def identity(): ...

    @qml.qjit(autograph=False)
    # CHECK: module @subroutine_test_5
    def subroutine_test_5():
        identity()
        identity()

    # CHECK-NOT: func.func private @identity_0()
    print(subroutine_test_5.mlir)
    qml.capture.disable()


test_two_callsites()


def test_two_callsites_quantum():
    """Test that two calls won't give multiple definitions
    int the quantum setting"""

    qml.capture.enable()

    @subroutine
    def identity(): ...

    @qml.qjit(autograph=False)
    @qml.qnode(qml.device("lightning.qubit", wires=1), autograph=False)
    # CHECK: module @subroutine_test_6
    def subroutine_test_6():
        # CHECK: [[QREG:%.+]] = quantum.alloc
        # CHECK: [[QREG_1:%.+]] = call @identity([[QREG]]) : (!quantum.reg) -> !quantum.reg
        identity()
        # CHECK: [[QREG_2:%.+]] = call @identity([[QREG_1]]) : (!quantum.reg) -> !quantum.reg
        identity()
        # CHECK: quantum.compbasis qreg [[QREG_2]] : !quantum.obs
        return qml.probs()

    # CHECK-NOT: func.func private @identity_0
    print(subroutine_test_6.mlir)
    qml.capture.disable()


test_two_callsites_quantum()


def test_two_qnodes_one_subroutine():
    """Test whether the cache correctly cleans up after each
    qnode tracing"""

    @subroutine
    def identity(): ...

    # CHECK: module @main

    @qml.qnode(qml.device("lightning.qubit", wires=1), autograph=False)
    def subroutine_test_7():
        # CHECK: [[QREG:%.+]] = quantum.alloc
        # CHECK: [[QREG_1:%.+]] = call @identity([[QREG]]) : (!quantum.reg) -> !quantum.reg
        identity()
        # CHECK: quantum.compbasis qreg [[QREG_1]] : !quantum.obs
        return qml.probs()

        # CHECK: func.func private @identity

    @qml.qnode(qml.device("null.qubit", wires=1), autograph=False)
    def subroutine_test_8():
        # CHECK: [[QREG:%.+]] = quantum.alloc
        # CHECK: [[QREG_1:%.+]] = call @identity_0([[QREG]]) : (!quantum.reg) -> !quantum.reg
        identity()
        # CHECK: quantum.compbasis qreg [[QREG_1]] : !quantum.obs
        return qml.probs()
        # CHECK: func.func private @identity_0

    qml.capture.enable()

    @qml.qjit(autograph=False)
    def main():
        return subroutine_test_7() + subroutine_test_8()

    print(main.mlir)

    qml.capture.disable()


test_two_qnodes_one_subroutine()


def test_with_constant():
    """Test that constants are not hoisted"""

    @subroutine
    def Hadamard_plus_1(c):
        # CHECK: func.func private @Hadamard_plus_1
        # CHECK-NEXT: %c = stablehlo.constant dense<1> : tensor<i64>
        one = jax.numpy.array(1)
        qml.Hadamard(c + one)

    qml.capture.enable()

    @qml.qjit(autograph=False)
    @qml.qnode(qml.device("null.qubit", wires=2), autograph=False)
    def circ():
        Hadamard_plus_1(0)
        return qml.probs()

    print(circ.mlir)

    qml.capture.disable()


test_with_constant()
