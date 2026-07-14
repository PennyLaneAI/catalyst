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

# pylint: disable = useless-parent-delegation, missing-function-docstring, missing-class-docstring, line-too-long

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
    # CHECK-LABEL: func.func public @c_no_params

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.operator "NoParams"() qubits([[q0]])
    # CHECK: static_data = {}
    # CHECK: qubit_map = {reg = [0]}
    NoParams(reg=0)

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: qref.operator "NoParams"() qubits([[q1]], [[q2]])
    # CHECK: static_data = {}
    # CHECK: qubit_map = {reg = [0, 1]}
    NoParams(reg=(0, 1))
    return qp.state()


print(c_no_params.mlir)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=1))
def c_adjoint():
    # CHECK-LABEL: func.func public @c_adjoint

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.operator "NoParams"() adj qubits([[q0]])
    op0 = NoParams(reg=0)
    qp.adjoint(op0)

    # CHECK: [[q0_1:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.operator "NoParams"() qubits([[q0_1]])
    qp.adjoint(qp.adjoint(op0))
    return qp.state()


print(c_adjoint.mlir)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=4))
def c_controlled():
    # CHECK-LABEL: func.func public @c_controlled

    # CHECK: [[true:%.+]] = arith.constant true
    # CHECK: [[false:%.+]] = arith.constant false
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 1]

    # CHECK: qref.operator "NoParams"() qubits([[q0]])
    # CHECK-NEXT: ctrls([[q1]]) ctrl_vals([[false]])
    # CHECK-NEXT: static_data = {}
    # CHECK-NEXT: qubit_map = {reg = [0]}
    op0 = NoParams(reg=0)
    qp.ctrl(op0, [1], control_values=[False])

    # CHECK: [[q0_1:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q1_1:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: qref.operator "NoParams"() qubits([[q0_1]])
    # CHECK-NEXT: ctrls([[q1_1]]) ctrl_vals([[false]])
    # CHECK-NEXT: static_data = {}
    # CHECK-NEXT: qubit_map = {reg = [0]}
    op0 = NoParams(reg=0)
    qp.ctrl(op0, [1], control_values=[0])

    # CHECK: [[q0_2:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q1_2:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q2_2:%.+]] = qref.get {{%.+}}[ 2]
    # CHECK: [[q3_2:%.+]] = qref.get {{%.+}}[ 3]

    # CHECK: qref.operator "NoParams"() qubits([[q0_2]])
    # CHECK-NEXT: ctrls([[q1_2]], [[q2_2]], [[q3_2]]) ctrl_vals([[false]], [[true]], [[true]])
    # CHECK-NEXT: static_data = {}
    # CHECK-NEXT: qubit_map = {reg = [0]}
    qp.ctrl(qp.ctrl(op0, [3], control_values=[True]), [1, 2], control_values=[False, True])
    return qp.state()


print(c_controlled.mlir)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=4))
def c_adjoint_and_controlled():
    # CHECK-LABEL: func.func public @c_adjoint_and_controlled

    # CHECK: [[false:%.+]] = arith.constant false

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: qref.operator "NoParams"() adj qubits([[q0]])
    # CHECK-NEXT: ctrls([[q1]]) ctrl_vals([[false]])
    # CHECK-NEXT: static_data = {}
    # CHECK-NEXT: qubit_map = {reg = [0]}
    op0 = NoParams(reg=0)
    qp.ctrl(qp.adjoint(op0), [1], [False])

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: qref.operator "NoParams"() adj qubits([[q0]])
    # CHECK-NEXT: ctrls([[q1]]) ctrl_vals([[false]])
    # CHECK-NEXT: static_data = {}
    # CHECK-NEXT: qubit_map = {reg = [0]}
    qp.adjoint(qp.ctrl(op0, [1], [False]))
    return qp.state()


print(c_adjoint_and_controlled.mlir)


class NoParamsCustomOp(qp.core.Operator2):

    def __init__(self, wires):
        super().__init__(wires=wires)


@qp.qjit(target="mlir", capture=True)
@qp.qnode(qp.device("null.qubit", wires=2))
def c_no_params_custom():
    # CHECK-LABEL: func.func public @c_no_params_custom
    # CHECK: [[false:%.+]] = arith.constant false

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.custom "NoParamsCustomOp"() [[q0]] : !qref.bit
    NoParamsCustomOp(wires=0)

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: qref.custom "NoParamsCustomOp"() [[q1]], [[q2]] : !qref.bit, !qref.bit
    NoParamsCustomOp(wires=(0, 1))

    # CHECK: [[q3:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q4:%.+]] = qref.get {{%.+}}[ 2]
    # CHECK: qref.custom "NoParamsCustomOp"() [[q3]] adj ctrls([[q4]])
    # CHECK-SAME: ctrlvals([[false]]) : !qref.bit ctrls !qref.bit
    qp.ctrl(qp.adjoint(NoParamsCustomOp(wires=(0,))), [2], [False])
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
    # CHECK-LABEL: func.func public @c_single_param

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]

    # CHECK: qref.operator "SingleParam"({{%.+}}: tensor<f64>) qubits([[q0]])
    # CHECK: static_data = {}
    # CHECK: param_map = {x = [0]} qubit_map = {reg = [0]}
    SingleParam(x, 0)

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}[ 2]
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
    # CHECK-LABEL: func.func public @c_single_param_custom

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.custom "SingleParamCustomOp"({{%.+}}) [[q0]] : !qref.bit
    SingleParamCustomOp(x, 0)

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}[ 2]
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
    # CHECK-LABEL: func.func public @c_compilable

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: qref.operator "CompilableData"() qubits([[q1]], [[q2]])
    # CHECK: static_data = {a = true, b = "some string", thing = [1, true, "string"]}
    # CHECK: qubit_map = {wires = [0, 1]}

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
    # CHECK-LABEL: func.func public @c_multiple_registers

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}[ 2]
    # CHECK: [[q3:%.+]] = qref.get {{%.+}}[ 3]
    # CHECK: [[q4:%.+]] = qref.get {{%.+}}[ 4]

    # CHECK: qref.operator "MultipleRegisters"() qubits([[q0]], [[q2]], [[q3]], [[q4]])
    # CHECK: static_data = {}
    # CHECK: qubit_map = {reg1 = [0], reg2 = [1, 2, 3]}
    MultipleRegisters(0, (2, 3, 4))

    # CHECK: [[q5:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.operator "MultipleRegisters"() qubits([[q5]])
    # CHECK: qubit_map = {reg1 = [0]}
    MultipleRegisters(0, [])

    # CHECK: qref.operator "MultipleRegisters"() qubits()
    # CHECK-NOT: qubit_map
    MultipleRegisters([], [])
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
    # CHECK-LABEL: func.func public @c_multi_params

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]

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
    # CHECK-LABEL: func.func public @c_multi_param_custom
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]

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
@qp.qnode(qp.device("null.qubit", wires=4))
def circuit_multirz(x: float):
    # CHECK-LABEL: func.func public @circuit_multirz
    # CHECK: [[false:%.+]] = arith.constant false

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}[ 2]

    # CHECK: qref.multirz({{%.+}}) [[q0]], [[q1]], [[q2]] : !qref.bit, !qref.bit, !qref.bit
    MultiRZ(x, (0, 1, 2))

    # CHECK: [[q3:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q4:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q5:%.+]] = qref.get {{%.+}}[ 3]

    # MultiRZ gets automatically canonicalized, so it will never have the `adj` attribute
    # CHECK: [[theta:%.+]] = arith.negf {{%.+}} : f64
    # CHECK: qref.multirz([[theta]]) [[q3]], [[q4]] ctrls([[q5]]) ctrlvals([[false]]) :
    # CHECK-SAME: !qref.bit, !qref.bit ctrls !qref.bit
    qp.ctrl(qp.adjoint(MultiRZ(x, wires=(0, 1))), [3], [False])
    return qp.state()


print(circuit_multirz.mlir)


class PauliRot(qp.core.Operator2):

    dynamic_argnames = ("phi",)
    compilable_argnames = ("pauli_word",)

    def __init__(self, phi, pauli_word, wires):
        super().__init__(phi, pauli_word, wires)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def circuit_paulirot(x: float):
    # CHECK-LABEL: func.func public @circuit_paulirot
    # CHECK: [[false:%.+]] = arith.constant false

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}[ 2]

    # CHECK: qref.paulirot ["X", "Y", "Z"]({{%.+}}) [[q0]], [[q1]], [[q2]] : !qref.bit, !qref.bit, !qref.bit
    PauliRot(x, "XYZ", (0, 1, 2))

    # CHECK: [[q3:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q4:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q5:%.+]] = qref.get {{%.+}}[ 2]

    # CHECK: qref.paulirot ["Y", "Z", "X"]({{%.+}}) [[q3]], [[q4]], [[q5]] : !qref.bit, !qref.bit, !qref.bit
    PauliRot(x, "YZX", (0, 1, 2))

    # CHECK: [[q3:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q4:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q5:%.+]] = qref.get {{%.+}}[ 2]

    # CHECK: qref.paulirot ["Y", "Z"]({{%.+}}) [[q3]], [[q4]] adj ctrls([[q5]])
    # CHECK-SAME: ctrlvals([[false]]) : !qref.bit, !qref.bit ctrls !qref.bit
    qp.ctrl(qp.adjoint(PauliRot(x, "YZ", (0, 1))), [2], [False])

    return qp.probs(wires=(0, 1, 2))


print(circuit_paulirot.mlir)


class GlobalPhase(qp.core.Operator2):

    dynamic_argnames = ("phi",)
    wire_argnames = ()

    def __init__(self, phi):
        super().__init__(phi=phi)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=3))
def circuit_gphase(x: float):
    # CHECK-LABEL: func.func public @circuit_gphase
    # CHECK: [[false:%.+]] = arith.constant false

    # CHECK: qref.gphase({{%.+}})
    GlobalPhase(x)

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.gphase({{%.+}}) adj ctrls([[q0]]) ctrlvals([[false]]) : ctrls !qref.bit
    qp.ctrl(qp.adjoint(GlobalPhase(x)), [0], [False])
    return qp.state()


print(circuit_gphase.mlir)


class QubitUnitary(qp.core.Operator2):

    dynamic_argnames = ("matrix",)

    def __init__(self, matrix, wires):
        super().__init__(matrix, wires)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=3))
def circuit_qubitunitary():
    # CHECK-LABEL: func.func public @circuit_qubitunitary
    # CHECK: [[false:%.+]] = arith.constant false

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.unitary({{%.+}} : tensor<2x2xcomplex<f64>>) [[q0]] : !qref.bit

    QubitUnitary(np.array([[0, 1], [1, 0]]), 0)

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: qref.unitary({{%.+}} : tensor<4x4xcomplex<f64>>) [[q1]], [[q2]] : !qref.bit, !qref.bit

    QubitUnitary(qp.CNOT.compute_matrix(), (0, 1))

    # CHECK: [[q3:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q4:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q5:%.+]] = qref.get {{%.+}}[ 2]
    # CHECK: qref.unitary({{%.+}} : tensor<4x4xcomplex<f64>>) [[q3]], [[q4]] adj ctrls([[q5]])
    # CHECK-SAME: ctrlvals([[false]]) : !qref.bit, !qref.bit ctrls !qref.bit

    qp.ctrl(qp.adjoint(QubitUnitary(qp.CNOT.compute_matrix(), (0, 1))), [2], [False])
    return qp.expval(qp.Z(0)), qp.expval(qp.Z(0))


print(circuit_qubitunitary.mlir)


class PCPhase(qp.core.Operator2):

    dynamic_argnames = ("phi", "dim")

    def __init__(self, phi, dim, wires):
        super().__init__(phi, dim, wires)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=3))
def c_pcphase(x: float, dim: int):
    # CHECK-LABEL: func.func public @c_pcphase
    # CHECK: [[false:%.+]] = arith.constant false

    # CHECK: [[q11:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q12:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: qref.pcphase({{%.+}}, {{%.+}}) [[q11]], [[q12]] : !qref.bit, !qref.bit

    PCPhase(x, dim, (0, 1))

    # CHECK: [[q3:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q4:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q5:%.+]] = qref.get {{%.+}}[ 2]

    # PCPhase gets automatically canonicalized, so it will never have the `adj` attribute
    # CHECK: [[theta:%.+]] = arith.negf {{%.+}} : f64
    # CHECK: qref.pcphase([[theta]], {{%.+}}) [[q3]], [[q4]] ctrls([[q5]]) ctrlvals([[false]]) :
    # CHECK-SAME: !qref.bit, !qref.bit ctrls !qref.bit

    qp.ctrl(qp.adjoint(PCPhase(x, dim, (0, 1))), [2], [False])
    return qp.state()


print(c_pcphase.mlir)


class StaticData(qp.core.Operator2):

    static_argnames = ("label",)
    wire_argnames = ("reg",)

    def __init__(self, label, reg):
        super().__init__(label=label, reg=reg)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=1))
def c_static_data():
    # CHECK-LABEL: func.func public @c_static_data

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.operator "StaticData"() qubits([[q0]])
    # CHECK: UID({{[0-9]+}})
    # CHECK-NOT: static_data
    # CHECK: qubit_map = {reg = [0]}
    StaticData("hello", reg=0)
    return qp.state()


print(c_static_data.mlir)


class HybridWires(qp.core.Operator2):

    hybrid_argnames = ("cwires",)
    wire_argnames = ("cwires",)

    def __init__(self, cwires):
        super().__init__(cwires=cwires)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=2))
def c_hybrid_wires():
    # CHECK-LABEL: func.func public @c_hybrid_wires

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: qref.operator "HybridWires"() qubits([[q0]], [[q1]])
    # CHECK: UID({{[0-9]+}})
    # CHECK: qubit_map = {cwires = [0, 1]}
    HybridWires(cwires=[qp.wires.Wires(0), qp.wires.Wires(1)])

    # CHECK: qref.operator "HybridWires"() qubits()
    # CHECK-NEXT: UID({{[0-9]+}})
    # CHECK-NOT: qubit_map
    HybridWires(cwires=[])
    return qp.state()


print(c_hybrid_wires.mlir)


class HybridNoOpArg(qp.core.Operator2):

    hybrid_argnames = ("angles",)

    def __init__(self, angles, wires):
        super().__init__(angles, wires)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=2))
def c_hybrid_arg_not_op():
    # CHECK-LABEL: func.func public @c_hybrid_arg_not_op

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.operator "HybridNoOpArg"({{%.+}}: tensor<f64>, {{%.+}}: tensor<f64>) qubits([[q0]])
    # CHECK: UID([[UID_A:[0-9]+]])
    # CHECK: param_map = {angles = [0, 1]} qubit_map = {wires = [0]}
    HybridNoOpArg([1.5, 2.5], wires=0)

    # CHECK: [[q0_1:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.operator "HybridNoOpArg"({{%.+}}: tensor<f64>, {{%.+}}: tensor<f64>) qubits([[q0_1]])
    # CHECK: UID([[UID_A]])
    # CHECK: param_map = {angles = [0, 1]} qubit_map = {wires = [0]}
    HybridNoOpArg([2.5, 4.5], wires=0)

    # CHECK: [[q0_2:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.operator "HybridNoOpArg"({{%.+}}: tensor<f64>, {{%.+}}: tensor<f64>, {{%.+}}: tensor<f64>) qubits([[q0_2]])
    # CHECK-NOT: UID([[UID_A]])
    # CHECK: UID({{[0-9]+}})
    # CHECK: param_map = {angles = [0, 1, 2]} qubit_map = {wires = [0]}
    HybridNoOpArg([1.5, 2.5, 3.5], wires=0)
    return qp.state()


print(c_hybrid_arg_not_op.mlir)


class HybridOpArg(qp.core.Operator2):

    dynamic_argnames = ("angle",)
    hybrid_argnames = ("op",)
    wire_argnames = ("cwires",)
    static_argnames = ("n_iters",)

    def __init__(self, angle, op, cwires, n_iters=1):
        super().__init__(angle, op, cwires, n_iters)


@qp.qjit(capture=True, target="mlir")
@qp.qnode(qp.device("null.qubit", wires=4))
def c_hybrid_op_arg(x: float, y: float):
    # CHECK-LABEL: func.func public @c_hybrid_op_arg
    # CHECK-SAME: ([[x:%.+]]: tensor<f64>, [[y:%.+]]: tensor<f64>)

    op1 = MultiRZ(x, [0, 1, 2])
    op2 = MultiRZ(x, [0, 1, 2])
    op3 = MultiRZ(x, [0, 1])

    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 3]
    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q3:%.+]] = qref.get {{%.+}}[ 2]
    # CHECK: qref.operator "HybridOpArg"([[y]]: tensor<f64>) qubits([[q0]], [[q1]], [[q2]], [[q3]])
    # CHECK-NEXT: UID([[UID_A:.+]]) forward([[x]]: tensor<f64>)
    # CHECK-NEXT: param_map = {angle = [0]} qubit_map = {cwires = [0], op = [1, 2, 3]}
    HybridOpArg(y, op1, cwires=[3])

    # CHECK: [[q4:%.+]] = qref.get {{%.+}}[ 3]
    # CHECK: [[q5:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q6:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q7:%.+]] = qref.get {{%.+}}[ 2]
    # CHECK: qref.operator "HybridOpArg"([[y]]: tensor<f64>) qubits([[q4]], [[q5]], [[q6]], [[q7]])
    # CHECK-NEXT: UID([[UID_A]]) forward([[x]]: tensor<f64>)
    # CHECK-NEXT: param_map = {angle = [0]} qubit_map = {cwires = [0], op = [1, 2, 3]}
    HybridOpArg(y, op2, cwires=[3])

    # CHECK: [[q8:%.+]] = qref.get {{%.+}}[ 3]
    # CHECK: [[q9:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q10:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: qref.operator "HybridOpArg"([[y]]: tensor<f64>) qubits([[q8]], [[q9]], [[q10]])
    # CHECK-NOT: UID([[UID_A]])
    # CHECK-NEXT: UID([[UID_B:.+]]) forward([[x]]: tensor<f64>)
    # CHECK-NEXT: param_map = {angle = [0]} qubit_map = {cwires = [0], op = [1, 2]}
    HybridOpArg(y, op3, cwires=[3])

    # CHECK: [[q11:%.+]] = qref.get {{%.+}}[ 3]
    # CHECK: [[q12:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[q13:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q14:%.+]] = qref.get {{%.+}}[ 2]
    # CHECK: qref.operator "HybridOpArg"([[y]]: tensor<f64>) qubits([[q11]], [[q12]], [[q13]], [[q14]])
    # CHECK-NOT: UID([[UID_A]])
    # CHECK-NOT: UID([[UID_B]])
    # CHECK-NEXT: UID([[UID_C:.+]]) forward([[x]]: tensor<f64>)
    # CHECK-NEXT: param_map = {angle = [0]} qubit_map = {cwires = [0], op = [1, 2, 3]}
    HybridOpArg(y, op1, cwires=[3], n_iters=2)


print(c_hybrid_op_arg.mlir)
