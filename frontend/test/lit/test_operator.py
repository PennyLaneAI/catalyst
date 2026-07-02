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

    # have to use different wire argnames or will in up CustomOp
    wire_argnames = ("reg",)

    def __init__(self, reg):
        super().__init__(reg=reg)


@qp.qjit(target="mlir", capture=True)
@qp.qnode(qp.device("null.qubit", wires=2))
def c_no_params():
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}
    # CHECK: qref.operator "NoParams"() qubits([[q0]])
    # CHECK: static_data = {}
    # CHECK: param_map = {} qubit_map = {reg = [0]}
    NoParams(reg=0)

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}
    # CHECK: qref.operator "NoParams"() qubits([[q1]], [[q2]])
    # CHECK: static_data = {}
    # CHECK: param_map = {} qubit_map = {reg = [0, 1]}
    NoParams(reg=(0, 1))
    return qp.state()


print(c_no_params.mlir)


class NoParamsCustomOp(qp.core.Operator2):

    def __init__(self, wires):
        super().__init__(wires=wires)


@qp.qjit(target="mlir", capture=True)
@qp.qnode(qp.device("null.qubit", wires=2))
def c_no_params_custom():
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}
    # CHECK: qref.custom "NoParamsCustomOp"() [[q0]] : !qref.bit
    NoParamsCustomOp(wires=0)

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}
    # CHECK: qref.custom "NoParamsCustomOp"() [[q1]], [[q2]] : !qref.bit, !qref.bit
    NoParamsCustomOp(wires=(0, 1))
    return qp.state()


print(c_no_params_custom.mlir)


class SingleParam(qp.core.Operator2):

    dynamic_argnames = ("x",)
    wire_argnames = ("reg",)

    def __init__(self, x, reg):
        super().__init__(x, reg=reg)


@qp.qjit(target="mlir", capture=True)
@qp.qnode(qp.device("null.qubit", wires=3))
def c_single_param(x: float):

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}

    # CHECK: qref.operator "SingleParam"({{%.+}}: tensor<f64>) qubits([[q0]])
    # CHECK: static_data = {}
    # CHECK: param_map = {x = [0]} qubit_map = {reg = [0]}
    SingleParam(x, 0)

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}
    # CHECK: qref.operator "SingleParam"({{%.+}}: tensor<4x4xf64>) qubits([[q1]], [[q2]])
    # CHECK: static_data = {}
    # CHECK: param_map = {x = [0]} qubit_map = {reg = [0, 1]}
    SingleParam(np.eye(4), (1, 2))

    return qp.state()


print(c_single_param.mlir)


class SingleParamCustomOp(qp.core.Operator2):

    dynamic_argnames = ("x",)

    def __init__(self, x, wires):
        super().__init__(x, wires=wires)


@qp.qjit(target="mlir", capture=True)
@qp.qnode(qp.device("null.qubit", wires=3))
def c_single_param_custom(x: float):

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}
    # CHECK: qref.custom "SingleParamCustomOp"({{%.+}}) [[q0]] : !qref.bit
    SingleParamCustomOp(x, 0)

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}
    # CHECK: qref.custom "SingleParamCustomOp"({{%.+}}) [[q1]], [[q2]] : !qref.bit, !qref.bit
    SingleParamCustomOp(0.5, (1, 2))

    return qp.state()


print(c_single_param_custom.mlir)


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
    wire_argnames = ("reg",)

    # note also having non-standard order with dynamic inputs after wires
    def __init__(self, reg, a, b, c):
        super().__init__(reg, a, b, c)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=1))
def c_multi_params():
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}

    # pylint: disable=line-too-long
    # CHECK: qref.operator "MultiParams"({{%.+}}: tensor<f64>, {{%.+}}: tensor<4x2x1xf64>, {{%.+}}: tensor<3xi64>) qubits([[q0]])
    # CHECK: static_data = {}
    # CHECK: param_map = {a = [0], b = [1], c = [2]} qubit_map = {reg = [0]}
    MultiParams(0, 0.5, c=np.array([1, 2, 3]), b=np.zeros((4, 2, 1)))
    return qp.state()


print(c_multi_params.mlir)


class MultiParamsCustom(qp.core.Operator2):

    dynamic_argnames = ("a", "b", "c")

    # note also having non-standard order with dynamic inputs after wires
    def __init__(self, wires, a, b, c):
        super().__init__(wires, a, b, c)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=1))
def c_multi_param_custom():
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}

    # pylint: disable=line-too-long
    # CHECK: qref.custom "MultiParamsCustom"({{%.+}}, {{%.+}}, {{%.+}}) [[q0]] : !qref.bit
    MultiParamsCustom(0, 0.5, c=0.7, b=2.4)
    return qp.state()


print(c_multi_param_custom.mlir)


class MultiRZ(qp.core.Operator2):

    dynamic_argnames = ("phi",)

    def __init__(self, phi, wires):
        super().__init__(phi, wires)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=2))
def circuit(x: float):
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}
    # CHECK: [[q1:%.+]] = qref.get {{%.+}}
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}

    # CHECK: qref.multirz({{%.+}}) [[q0]], [[q1]], [[q2]] : !qref.bit, !qref.bit, !qref.bit
    MultiRZ(x, (0, 1, 2))
    return qp.state()


print(circuit.mlir)


class PauliRot(qp.core.Operator2):

    dynamic_argnames = ("phi",)
    compilable_argnames = ("pauli_word",)

    def __init__(self, phi, pauli_word, wires):
        super().__init__(phi, pauli_word, wires)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def circuit(x: float):

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}
    # CHECK: [[q1:%.+]] = qref.get {{%.+}}
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}

    # CHECK: qref.paulirot ["X", "Y", "Z"]({{%.+}}) [[q0]], [[q1]], [[q2]] : !qref.bit, !qref.bit, !qref.bit
    PauliRot(x, "XYZ", (0, 1, 2))

    # CHECK: [[q3:%.+]] = qref.get {{%.+}}
    # CHECK: [[q4:%.+]] = qref.get {{%.+}}
    # CHECK: [[q5:%.+]] = qref.get {{%.+}}

    # CHECK: qref.paulirot ["Y", "Z", "X"]({{%.+}}) [[q3]], [[q4]], [[q5]] : !qref.bit, !qref.bit, !qref.bit
    PauliRot(x, "YZX", (0, 1, 2))

    return qp.probs(wires=(0, 1, 2))


print(circuit.mlir)


class GlobalPhase(qp.core.Operator2):

    dynamic_argnames = ("phi",)
    wire_argnames = ()

    def __init__(self, phi):
        super().__init__(phi=phi)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def circuit(x: float):

    # CHECK: qref.gphase({{%.+}})
    GlobalPhase(x)
    return qp.state()


print(circuit.mlir)


class QubitUnitary(qp.core.Operator2):

    dynamic_argnames = ("matrix",)

    def __init__(self, matrix, wires):
        super().__init__(matrix, wires)


@qp.qjit(capture=True)
@qp.qnode(qp.device("lightning.qubit", wires=3))
def c():

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}
    # CHECK: qref.unitary({{%.+}}. : tensor<2x2xcomplex<f64>>) [[q0]] : !qref.bit

    QubitUnitary(np.array([[0, 1], [1, 0]]), 0)

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}
    # CHECK: qref.unitary({{%.+}}. : tensor<4x4xcomplex<f64>>) [[q1]], [[q2]] : !qref.bit, !qref.bit

    QubitUnitary(qp.CNOT.compute_matrix(), (0, 1))
    return qp.expval(qp.Z(0)), qp.expval(qp.Z(0))


print(circuit.mlir)


class PCPhase(qp.core.Operator2):

    dynamic_argnames = ("phi", "dim")

    def __init__(self, phi, dim, wires):
        super().__init__(phi, dim, wires)


@qp.qjit(capture=True)
@qp.qnode(qp.device("lightning.qubit", wires=2))
def c(x: float, dim: int):

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}
    # CHECK: qref.pcphase({{%.+}}, {{%.+}}) [[q1]], [[q2]] :  !qref.bit, !qref.bit

    PCPhase(x, dim, (0, 1))
    return qp.state()


print(circuit.mlir)
