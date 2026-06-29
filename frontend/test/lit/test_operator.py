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
"""Tests for operator in Catalyst."""

# pylint: disable = useless-parent-delegation, missing-function-docstring, missing-class-docstring

# RUN: %PYTHON %s | FileCheck %s

import numpy as np
import pennylane as qp


class NoParams(qp.core.Operator2):

    def __init__(self, wires):
        super().__init__(wires=wires)


@qp.qjit(target="mlir", capture=True)
@qp.qnode(qp.device("null.qubit", wires=2))
def c_no_params():
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}
    # CHECK: qref.operator "NoParams"() qubits([[q0]])
    # CHECK: static_data = {}
    # CHECK: param_map = {} qubit_map = {wires = [0]}
    NoParams(wires=0)

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}
    # CHECK: qref.operator "NoParams"() qubits([[q1]], [[q2]])
    # CHECK: static_data = {}
    # CHECK: param_map = {} qubit_map = {wires = [0, 1]}
    NoParams(wires=(0, 1))
    return qp.state()


print(c_no_params.mlir)


class SingleParam(qp.core.Operator2):

    dynamic_argnames = ("x",)

    def __init__(self, x, wires):
        super().__init__(x, wires=wires)


@qp.qjit(target="mlir", capture=True)
@qp.qnode(qp.device("null.qubit", wires=3))
def c_single_param(x: float):

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}

    # CHECK: qref.operator "SingleParam"({{%.+}}: tensor<f64>) qubits([[q0]])
    # CHECK: static_data = {}
    # CHECK: param_map = {x = [0]} qubit_map = {wires = [0]}
    SingleParam(x, 0)

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}
    # CHECK: qref.operator "SingleParam"({{%.+}}: tensor<4x4xf64>) qubits([[q1]], [[q2]])
    # CHECK: static_data = {}
    # CHECK: param_map = {x = [0]} qubit_map = {wires = [0, 1]}
    SingleParam(np.eye(4), (1, 2))

    return qp.state()


print(c_single_param.mlir)


class CompilableData(qp.core.Operator2):

    compilable_argnames = ("a", "b", "thing")

    def __init__(self, a, b, thing, wires):
        super().__init__(a=a, b=b, thing=thing, wires=wires)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def c_compilable():
    # CHECK: [[q1:%.+]] = qref.get {{%.+}}
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}
    # CHECK: qref.operator "CompilableData"() qubits([[q1]], [[q2]])
    # CHECK: static_data = {a = true, b = "some string", thing = [1, true, "string"]}
    # CHECK: param_map = {} qubit_map = {wires = [0, 1]}

    CompilableData(True, "some string", (1, True, "string"), wires=(0, 1))

    return qp.state()


print(c_compilable.mlir)


class MultipleRegisters(qp.core.Operator2):

    wire_argnames = ("reg1", "reg2")

    def __init__(self, reg1, reg2):
        super().__init__(reg1=reg1, reg2=reg2)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=5))
def c_multiple_registers():
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}
    # CHECK: [[q3:%.+]] = qref.get {{%.+}}
    # CHECK: [[q4:%.+]] = qref.get {{%.+}}

    # CHECK: qref.operator "MultipleRegisters"() qubits([[q0]], [[q2]], [[q3]], [[q4]])
    # CHECK: static_data = {}
    # CHECK: param_map = {} qubit_map = {reg1 = [0], reg2 = [1, 2, 3]}
    MultipleRegisters(0, (2, 3, 4))
    return qp.state()


print(c_multiple_registers.mlir)


class MultiParams(qp.core.Operator2):

    dynamic_argnames = ("a", "b", "c")

    # note also having non-standard order with dynamic inputs after wires
    def __init__(self, wires, a, b, c):
        super().__init__(wires, a, b, c)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=1))
def c_multi_params():
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}

    # pylint: disable=line-too-long
    # CHECK: qref.operator "MultiParams"({{%.+}}: tensor<f64>, {{%.+}}: tensor<4x2x1xf64>, {{%.+}}: tensor<3xi64>) qubits([[q0]])
    # CHECK: static_data = {}
    # CHECK: param_map = {a = [0], b = [1], c = [2]} qubit_map = {wires = [0]}
    MultiParams(0, 0.5, c=np.array([1, 2, 3]), b=np.zeros((4, 2, 1)))
    return qp.state()


print(c_multi_params.mlir)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=4))
def c_controlled():
    # CHECK: [[true:%.+]] = arith.constant true
    # CHECK: [[false:%.+]] = arith.constant false
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 1]

    # CHECK: qref.operator "NoParams"() qubits([[q0]])
    # CHECK-NEXT: ctrls([[q1]]) ctrl_vals([[false]])
    op0 = NoParams(wires=0)
    qp.ops.ControlledOp2(op0, [1], control_values=[False])

    # CHECK: [[q0_1:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q1_1:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q2_1:%.+]] = qref.get {{%.+}}[ 2]
    # CHECK: [[q3_1:%.+]] = qref.get {{%.+}}[ 3]

    # CHECK: qref.operator "NoParams"() qubits([[q0_1]])
    # CHECK-NEXT: ctrls([[q1_1]], [[q2_1]], [[q3_1]]) ctrl_vals([[true]], [[false]], [[true]])
    qp.ops.ControlledOp2(
        qp.ops.ControlledOp2(op0, [1], control_values=[True]), [2, 3], control_values=[False, True]
    )
    return qp.state()


print(c_controlled.mlir)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=1))
def c_adjoint():
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.operator "NoParams"() adj qubits([[q0]])
    op0 = NoParams(wires=0)
    qp.adjoint(op0)

    # CHECK: [[q0_1:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.operator "NoParams"() qubits([[q0_1]])
    qp.adjoint(qp.adjoint(op0))
    return qp.state()


print(c_adjoint.mlir)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=4))
def c_adjoint_and_controlled():
    # CHECK: [[false:%.+]] = arith.constant false

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: qref.operator "NoParams"() adj qubits([[q0]])
    # CHECK-NEXT: ctrls([[q1]]) ctrl_vals([[false]])
    op0 = NoParams(wires=0)
    qp.ops.ControlledOp2(qp.adjoint(op0), [1], [False])

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: qref.operator "NoParams"() adj qubits([[q0]])
    # CHECK-NEXT: ctrls([[q1]]) ctrl_vals([[false]])
    qp.adjoint(qp.ops.ControlledOp2(op0, [1], [False]))
    return qp.state()


print(c_adjoint_and_controlled.mlir)
