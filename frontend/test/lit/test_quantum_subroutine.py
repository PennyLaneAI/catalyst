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

from functools import partial

import jax
import pennylane as qp
from jax import numpy as jnp
from pennylane.capture import subroutine

# pylint: disable=line-too-long


def test_subroutine_classical():
    """Test that a subroutine is just jax.jit
    when used in a classical setting.

    A desirable behaviour may be to error.
    """

    @subroutine
    def add_one(x):
        return x + 1

    qp.capture.enable()

    @qp.qjit
    # CHECK: module @main
    def main():
        # CHECK: %{{.*}} = call @add_one(%{{.*}}) : (tensor<i64>) -> tensor<i64>
        return add_one(0)

    print(main.mlir)
    qp.capture.disable()


test_subroutine_classical()


def test_quantum_subroutine_identity_restore_wires():
    """Test that a subroutine receiving a register as a parameter"""

    @subroutine
    def identity(): ...

    qp.capture.enable()

    @qp.qjit
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK: module @main
    def main():
        # CHECK: [[QREG:%.+]] = qref.alloc( 1) : !qref.reg<1>
        # CHECK: [[QUBIT:%.+]] = qref.get [[QREG]][ 0] : !qref.reg<1> -> !qref.bit
        # CHECK: qref.custom "Hadamard"() [[QUBIT]] : !qref.bit
        # CHECK: call @identity([[QREG]]) : (!qref.reg<1>) -> ()
        # CHECK: [[QUBIT:%.+]] = qref.get [[QREG]][ 0] : !qref.reg<1> -> !qref.bit
        # CHECK: qref.custom "Hadamard"() [[QUBIT]] : !qref.bit

        qp.Hadamard(wires=[0])
        identity()
        qp.Hadamard(wires=[0])
        return qp.probs()

    # CHECK: func.func private @identity([[REG:%.+]]: !qref.reg<1>)
    # CHECK-NEXT: return

    print(main.mlir)
    qp.capture.disable()


test_quantum_subroutine_identity_restore_wires()


def test_quantum_subroutine_identity():
    """Test that a subroutine receiving a register as a parameter"""

    @subroutine
    def identity(): ...

    qp.capture.enable()

    @qp.qjit
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK: module @main
    def main():
        # CHECK: [[QREG:%.+]] = qref.alloc
        # CHECK: call @identity([[QREG]]) : (!qref.reg<1>) -> ()
        # CHECK: qref.compbasis(qreg [[QREG]] : !qref.reg<1>) : !quantum.obs
        identity()
        return qp.probs()

    # CHECK: func.func private @identity([[REG:%.+]]: !qref.reg<1>)
    # CHECK-NEXT: return

    print(main.mlir)
    qp.capture.disable()


test_quantum_subroutine_identity()


def test_quantum_subroutine_wire_param():
    """Pass a parameter that is a wire/integer"""

    @subroutine
    def Hadamard0(wire):
        qp.Hadamard(wire)

    qp.capture.enable()

    @qp.qjit
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK: module @subroutine_test
    def subroutine_test(c: int):
        # CHECK: func.func public @subroutine_test([[ARG0:%.+]]
        # CHECK: [[QREG:%.+]] = qref.alloc
        # CHECK: call @Hadamard0([[QREG]], [[ARG0:%.+]]) : (!qref.reg<1>, tensor<i64>) -> ()
        # CHECK: qref.compbasis(qreg [[QREG]] : !qref.reg<1>) : !quantum.obs
        Hadamard0(c)
        return qp.probs()

    # CHECK: func.func private @Hadamard0([[REG:%.+]]: !qref.reg<1>, [[WIRE_IDX_TENSOR:%.+]]: tensor<i64>)
    # CHECK-NEXT: [[WIRE_IDX:%.+]] = tensor.extract [[WIRE_IDX_TENSOR]][] : tensor<i64>
    # CHECK-NEXT: [[QUBIT:%.+]] = qref.get [[REG]][[[WIRE_IDX]]] : !qref.reg<1>, i64 -> !qref.bit
    # CHECK-NEXT: qref.custom "Hadamard"() [[QUBIT]] : !qref.bit
    # CHECK-NEXT: return

    print(subroutine_test.mlir)

    qp.capture.disable()


test_quantum_subroutine_wire_param()


def test_quantum_subroutine_gate_param_param():
    """Test passing a regular parameter"""

    @subroutine
    def RX_on_wire_0(param):
        qp.RX(param, wires=[0])

    qp.capture.enable()

    @qp.qjit
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK: module @subroutine_test_2
    def subroutine_test_2():
        # CHECK-DAG: [[CST:%.+]] = stablehlo.constant dense<3.140000e+00>
        # CHECK-DAG: [[QREG:%.+]] = qref.alloc
        # CHECK: call @RX_on_wire_0([[QREG]], [[CST]]) : (!qref.reg<1>, tensor<f64>) -> ()
        # CHECK: qref.compbasis(qreg [[QREG]] : !qref.reg<1>) : !quantum.obs
        RX_on_wire_0(3.14)
        return qp.probs()

    # CHECK: func.func private @RX_on_wire_0([[REG:%.+]]: !qref.reg<1>, [[PARAM_TENSOR:%.+]]: tensor<f64>)
    # CHECK-NEXT: [[QUBIT:%.+]] = qref.get [[REG]][ 0] : !qref.reg<1> -> !qref.bit
    # CHECK-NEXT: [[PARAM:%.+]] = tensor.extract [[PARAM_TENSOR]][] : tensor<f64>
    # CHECK-NEXT: qref.custom "RX"([[PARAM]]) [[QUBIT]] : !qref.bit
    # CHECK-NEXT: return
    print(subroutine_test_2.mlir)

    qp.capture.disable()


test_quantum_subroutine_gate_param_param()


def test_quantum_subroutine_with_control_flow():
    """Test control flow inside the subroutine"""

    qp.capture.enable()

    @subroutine
    def conditional_RX(param: float):

        def true_path():
            qp.RX(param, wires=[0])

        def false_path(): ...

        qp.cond(param != 0.0, true_path, false_path)()

    @qp.qjit(autograph=False)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK: module @subroutine_test_3
    def subroutine_test_3():
        # CHECK-DAG: [[CST:%.+]] = stablehlo.constant dense<3.140000e+00>
        # CHECK-DAG: [[QREG:%.+]] = qref.alloc
        # CHECK: call @conditional_RX([[QREG]], [[CST]]) : (!qref.reg<1>, tensor<f64>) -> ()
        # CHECK: qref.compbasis(qreg [[QREG]] : !qref.reg<1>) : !quantum.obs
        conditional_RX(3.14)
        return qp.probs()

    # CHECK: func.func private @conditional_RX([[QREG:%.+]]: !qref.reg<1>, [[PARAM_TENSOR:%.+]]: tensor<f64>)
    # CHECK-NEXT: [[ZERO:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    # CHECK-NEXT: [[COND_TENSOR:%.+]] = stablehlo.compare  NE, [[PARAM_TENSOR]], [[ZERO]],  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    # CHECK-NEXT: [[COND:%.+]] = tensor.extract [[COND_TENSOR]][] : tensor<i1>
    # CHECK-NEXT: scf.if [[COND]]
    # CHECK:        [[QUBIT:%.+]] = qref.get [[QREG]][ 0] : !qref.reg<1> -> !qref.bit
    # CHECK:        [[PARAM:%.+]] = tensor.extract [[PARAM_TENSOR]][] : tensor<f64>
    # CHECK:        qref.custom "RX"([[PARAM]]) [[QUBIT]] : !qref.bit
    # CHECK:      return
    print(subroutine_test_3.mlir)
    qp.capture.disable()


test_quantum_subroutine_with_control_flow()


def test_nested_subroutine_call():
    """Test nested subroutine call"""

    qp.capture.enable()

    @subroutine
    def Hadamard_subroutine():
        qp.Hadamard(wires=[0])

    @subroutine
    def Hadamard_caller():
        Hadamard_subroutine()

    @qp.qjit(autograph=False)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK: module @subroutine_test_4
    def subroutine_test_4():
        # CHECK: [[QREG:%.+]] = qref.alloc
        # CHECK: call @Hadamard_caller([[QREG]]) : (!qref.reg<1>) -> ()
        # CHECK: qref.compbasis(qreg [[QREG]] : !qref.reg<1>) : !quantum.obs
        Hadamard_caller()
        return qp.probs()

    # CHECK: func.func private @Hadamard_caller([[QREG:%.+]]: !qref.reg<1>)
    # CHECK-NEXT: call @Hadamard_subroutine([[QREG]]) : (!qref.reg<1>) -> ()
    # CHECK-NEXT: return

    # CHECK: func.func private @Hadamard_subroutine([[QREG:%.+]]: !qref.reg<1>)
    # CHECK-NEXT: [[QUBIT:%.+]] = qref.get [[QREG]][ 0] : !qref.reg<1> -> !qref.bit
    # CHECK-NEXT: qref.custom "Hadamard"() [[QUBIT]] : !qref.bit
    # CHECK-NEXT: return
    print(subroutine_test_4.mlir)
    qp.capture.disable()


test_nested_subroutine_call()


def test_two_callsites():
    """Test that two calls won't give multiple definitions
    in the classical setting"""

    qp.capture.enable()

    @subroutine
    def identity(): ...

    @qp.qjit(autograph=False)
    # CHECK: module @subroutine_test_5
    def subroutine_test_5():
        identity()
        identity()

    # CHECK-NOT: func.func private @identity_0()
    print(subroutine_test_5.mlir)
    qp.capture.disable()


test_two_callsites()


def test_two_callsites_quantum():
    """Test that two calls won't give multiple definitions
    int the quantum setting"""

    qp.capture.enable()

    @subroutine
    def identity(): ...

    @qp.qjit(autograph=False)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK: module @subroutine_test_6
    def subroutine_test_6():
        # CHECK: [[QREG:%.+]] = qref.alloc
        # CHECK: call @identity([[QREG]]) : (!qref.reg<1>) -> ()
        identity()
        # CHECK: call @identity([[QREG]]) : (!qref.reg<1>) -> ()
        identity()
        # CHECK: qref.compbasis(qreg [[QREG]] : !qref.reg<1>) : !quantum.obs
        return qp.probs()

    # CHECK-NOT: func.func private @identity_0
    print(subroutine_test_6.mlir)
    qp.capture.disable()


test_two_callsites_quantum()


def test_two_qnodes_one_subroutine():
    """Test whether the cache correctly cleans up after each
    qnode tracing"""

    @subroutine
    def identity(): ...

    # CHECK: module @main

    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def subroutine_test_7():
        # CHECK: [[QREG:%.+]] = qref.alloc
        # CHECK: call @identity([[QREG]]) : (!qref.reg<1>) -> ()
        identity()
        # CHECK: qref.compbasis(qreg [[QREG]] : !qref.reg<1>) : !quantum.obs
        return qp.probs()

        # CHECK: func.func private @identity

    @qp.qnode(qp.device("null.qubit", wires=1))
    def subroutine_test_8():
        # CHECK: [[QREG:%.+]] = qref.alloc
        # CHECK: call @identity_0([[QREG]]) : (!qref.reg<1>) -> ()
        identity()
        # CHECK: qref.compbasis(qreg [[QREG]] : !qref.reg<1>) : !quantum.obs
        return qp.probs()

        # CHECK: func.func private @identity_0

    qp.capture.enable()

    @qp.qjit(autograph=False)
    def main():
        return subroutine_test_7() + subroutine_test_8()

    print(main.mlir)

    qp.capture.disable()


test_two_qnodes_one_subroutine()


def test_with_constant():
    """Test that constants are not hoisted"""

    @subroutine
    def Hadamard_plus_1(c):
        # CHECK: func.func private @Hadamard_plus_1
        # CHECK-NEXT: %c = stablehlo.constant dense<1> : tensor<i64>
        one = jax.numpy.array(1)
        qp.Hadamard(c + one)

    qp.capture.enable()

    @qp.qjit(autograph=False)
    @qp.qnode(qp.device("null.qubit", wires=2))
    def circ():
        Hadamard_plus_1(0)
        return qp.probs()

    print(circ.mlir)

    qp.capture.disable()


test_with_constant()


def test_basic_subroutine():
    """Test the most simple subroutine."""

    @qp.templates.Subroutine
    def f(x, wires):
        qp.RX(x, wires)

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=1))
    # CHECK: module @circuit
    def circuit(x):
        # CHECK: [[zero:%.+]] = stablehlo.constant dense<0> : tensor<i64>
        # CHECK: [[QREG:%.+]] = qref.alloc
        # CHECK: [[zero_tensor:%.+]] = stablehlo.broadcast_in_dim [[zero]], dims = [] : (tensor<i64>) -> tensor<1xi64>
        # CHECK: call @f([[QREG]], %arg0, [[zero_tensor]]) : (!qref.reg<1>, tensor<f64>, tensor<1xi64>)
        # CHECK: qref.compbasis(qreg [[QREG]] : !qref.reg<1>) : !quantum.obs
        f(x, 0)
        return qp.probs()

    # CHECK: func.func private @f(%arg0: !qref.reg<1>, %arg1: tensor<f64>, %arg2: tensor<1xi64>)
    # CHECK: [[IDX:%.+]] = stablehlo.slice %arg2 [0:1] : (tensor<1xi64>) -> tensor<1xi64>
    # CHECK: [[reshape:%.+]] = stablehlo.reshape [[IDX]] : (tensor<1xi64>) -> tensor<i64>
    # CHECK: [[extract:%.+]] = tensor.extract [[reshape]][] : tensor<i64>
    # CHECK: [[qubit:%.+]] = qref.get %arg0[[[extract]]] : !qref.reg<1>, i64 -> !qref.bit
    # CHECK: [[angle:%.+]] = tensor.extract %arg1[] : tensor<f64>
    # CHECK: qref.custom "RX"([[angle]]) [[qubit]] : !qref.bit
    # CHECK-NEXT: return
    circuit(0.5)
    print(circuit.mlir)


test_basic_subroutine()


def test_multiple_metadata():
    """Test a subroutine with metadata becomes multiple functions.

    Each metadata should get its own function.
    """

    @partial(qp.templates.Subroutine, static_argnames="metadata")
    def f(wires, metadata):
        if metadata == "X":
            qp.X(wires)
        elif metadata == "Y":
            qp.Y(wires)
        else:
            qp.Z(wires)

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=1))
    # CHECK: module @circuit
    def circuit():
        # CHECK: [[QREG:%.+]] = qref.alloc
        # CHECK: call @f([[QREG]], {{%.+}}) : (!qref.reg<1>, tensor<1xi64>)
        # CHECK: call @f_0([[QREG]], {{%.+}}) : (!qref.reg<1>, tensor<1xi64>)
        # CHECK: call @f_1([[QREG]], {{%.+}}) : (!qref.reg<1>, tensor<1xi64>)
        # CHECK: call @f([[QREG]], {{%.+}}) : (!qref.reg<1>, tensor<1xi64>)
        # CHECK: qref.compbasis(qreg [[QREG]] : !qref.reg<1>) : !quantum.obs
        f(0, "X")
        f(0, "Y")
        f(0, "Z")
        f(0, "X")  # check reusing the first call to the function
        return qp.probs()

    # CHECK: func.func private @f(%arg0: !qref.reg<1>, %arg1: tensor<1xi64>)
    # CHECK: qref.custom "PauliX"
    # CHECK-NEXT: return

    # CHECK: func.func private @f_0(%arg0: !qref.reg<1>, %arg1: tensor<1xi64>)
    # CHECK: qref.custom "PauliY"
    # CHECK-NEXT: return

    # CHECK: func.func private @f_1(%arg0: !qref.reg<1>, %arg1: tensor<1xi64>)
    # CHECK: qref.custom "PauliZ"
    # CHECK-NEXT: return
    print(circuit.mlir)


test_multiple_metadata()


def test_different_shapes():
    """Test a subroutine with different shape inputs get their own function."""

    @qp.templates.Subroutine
    def my_subroutine(data, wires):
        @qp.for_loop(data.shape[0])
        def loop(i):
            qp.RX(data[i], wires[i])

        loop()  # pylint: disable=no-value-for-parameter

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("null.qubit", wires=1))
    # CHECK: module @circuit
    def circuit():
        # CHECK: [[QREG:%.+]] = qref.alloc
        # CHECK: call @my_subroutine([[QREG]], %arg0, {{%.+}}) : (!qref.reg<1>, tensor<3xf64>, tensor<3xi64>)
        # CHECK: call @my_subroutine([[QREG]], %arg1, {{%.+}}) : (!qref.reg<1>, tensor<3xf64>, tensor<3xi64>)
        # CHECK: call @my_subroutine_0([[QREG]], %arg2, {{%.+}}) : (!qref.reg<1>, tensor<2xf64>, tensor<3xi64>)
        # CHECK: qref.compbasis(qreg [[QREG]] : !qref.reg<1>) : !quantum.obs
        my_subroutine(jnp.array([0.0, 0.1, 0.2]), [0, 1, 2])
        my_subroutine(jnp.array([0.0, 0.1, 0.2]), [0, 1, 2])
        my_subroutine(jnp.array([0.5, 1.2]), [0, 1, 2])
        return qp.probs()

    # CHECK: func.func private @my_subroutine(%arg0: !qref.reg<1>, %arg1: tensor<3xf64>, %arg2: tensor<3xi64>)
    # CHECK:   [[ub:%.+]] = arith.constant 3 : index
    # CHECK:   scf.for {{%.+}} = {{%.+}} to [[ub]] step {{%.+}}
    # CHECK:   qref.custom "RX"

    # CHECK: func.func private @my_subroutine_0(%arg0: !qref.reg<1>, %arg1: tensor<2xf64>, %arg2: tensor<3xi64>)
    # CHECK: arith.constant 2 : index
    # CHECK: scf.for
    # CHECK: qref.custom "RX"

    print(circuit.mlir)


test_different_shapes()
