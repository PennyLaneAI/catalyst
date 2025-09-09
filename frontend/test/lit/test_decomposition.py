# Copyright 2022-2023 Xanadu Quantum Technologies Inc.
import os
import pathlib
import platform
from copy import deepcopy
from functools import partial

import jax
import numpy as np
import pennylane as qml
from pennylane.devices.capabilities import OperatorProperties
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

from catalyst import measure, qjit
from catalyst.compiler import get_lib_path
from catalyst.device import get_device_capabilities
from catalyst.jax_primitives import decomposition_rule

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
# pylint: disable=line-too-long


TEST_PATH = os.path.dirname(__file__)
CONFIG_CUSTOM_DEVICE = pathlib.Path(f"{TEST_PATH}/../custom_device/custom_device.toml")


def get_custom_device_without(num_wires, discards=frozenset(), force_matrix=frozenset()):
    """Generate a custom device without gates in discards."""

    class CustomDevice(qml.devices.Device):
        """Custom Gate Set Device"""

        name = "Custom Device"
        config_filepath = CONFIG_CUSTOM_DEVICE

        _to_matrix_ops = {}

        def __init__(self, wires=None):
            super().__init__(wires=wires)
            self.qjit_capabilities = deepcopy(get_device_capabilities(self))
            for gate in discards:
                self.qjit_capabilities.operations.pop(gate, None)
            for gate in force_matrix:
                self.qjit_capabilities.operations.pop(gate, None)
                self._to_matrix_ops[gate] = OperatorProperties(False, False, False)

        def apply(self, operations, **kwargs):
            """Unused"""
            raise RuntimeError("Only C/C++ interface is defined")

        @staticmethod
        def get_c_interface():
            """Returns a tuple consisting of the device name, and
            the location to the shared object with the C/C++ device implementation.
            """
            system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
            lib_path = (
                get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/librtd_null_qubit" + system_extension
            )
            return "NullQubit", lib_path

        def execute(self, circuits, execution_config):
            """Execution."""
            return circuits, execution_config

    return CustomDevice(wires=num_wires)


def test_decompose_multicontrolledx():
    """Test decomposition of MultiControlledX as an aliased gate."""
    dev = get_custom_device_without(5, discards={"MultiControlledX"})

    @qjit(target="mlir")
    @qml.qnode(dev)
    # CHECK-LABEL: public @jit_decompose_multicontrolled_x1
    def decompose_multicontrolled_x1(theta: float):
        qml.RX(theta, wires=[0])
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK:     quantum.custom "PauliX"() {{%[a-zA-Z0-9_]+}} ctrls({{%[a-zA-Z0-9_]+}}, {{%[a-zA-Z0-9_]+}}, {{%[a-zA-Z0-9_]+}})
        # CHECK-NOT: name = "MultiControlledX"
        qml.MultiControlledX(wires=[0, 1, 2, 3])
        return qml.state()

    print(decompose_multicontrolled_x1.mlir)


test_decompose_multicontrolledx()


def test_decompose_rot():
    """Test decomposition of Rot gate."""
    dev = get_custom_device_without(1, discards={"Rot", "C(Rot)"})

    @qjit(target="mlir")
    @qml.qnode(dev)
    # CHECK-LABEL: public @jit_decompose_rot
    def decompose_rot(phi: float, theta: float, omega: float):
        # CHECK-NOT: name = "Rot"
        # CHECK: [[phi:%.+]] = tensor.extract %arg0
        # CHECK-NOT: name = "Rot"
        # CHECK:  {{%.+}} = quantum.custom "RZ"([[phi]])
        # CHECK-NOT: name = "Rot"
        # CHECK: [[theta:%.+]] = tensor.extract %arg1
        # CHECK-NOT: name = "Rot"
        # CHECK: {{%.+}} = quantum.custom "RY"([[theta]])
        # CHECK-NOT: name = "Rot"
        # CHECK: [[omega:%.+]] = tensor.extract %arg2
        # CHECK-NOT: name = "Rot"
        # CHECK: {{%.+}} = quantum.custom "RZ"([[omega]])
        # CHECK-NOT: name = "Rot"
        qml.Rot(phi, theta, omega, wires=0)
        return measure(wires=0)

    print(decompose_rot.mlir)


test_decompose_rot()


def test_decompose_s():
    """Test decomposition of S gate."""
    dev = get_custom_device_without(1, discards={"S", "C(S)"})

    @qjit(target="mlir")
    @qml.qnode(dev)
    # CHECK-LABEL: public @jit_decompose_s
    def decompose_s():
        # CHECK-NOT: name="S"
        # CHECK: [[pi_div_2:%.+]] = arith.constant 1.57079{{.+}} : f64
        # CHECK-NOT: name = "S"
        # CHECK: {{%.+}} = quantum.custom "PhaseShift"([[pi_div_2]])
        # CHECK-NOT: name = "S"
        qml.S(wires=0)
        return measure(wires=0)

    print(decompose_s.mlir)


test_decompose_s()


def test_decompose_qubitunitary():
    """Test decomposition of QubitUnitary"""
    dev = get_custom_device_without(1, discards={"QubitUnitary"})

    @qjit(target="mlir")
    @qml.qnode(dev)
    # CHECK-LABEL: public @jit_decompose_qubit_unitary
    def decompose_qubit_unitary(U: jax.core.ShapedArray([2, 2], float)):
        # CHECK-NOT: name = "QubitUnitary"
        # CHECK: quantum.custom "RZ"
        # CHECK: quantum.custom "RY"
        # CHECK: quantum.custom "RZ"
        # CHECK-NOT: name = "QubitUnitary"
        qml.QubitUnitary(U, wires=0)
        return measure(wires=0)

    print(decompose_qubit_unitary.mlir)


test_decompose_qubitunitary()


def test_decompose_singleexcitation():
    """
    Test that single excitation is not decomposed.
    """
    dev = get_custom_device_without(2)

    @qjit(target="mlir")
    @qml.qnode(dev)
    # CHECK-LABEL: public @jit_decompose_singleexcitation
    def decompose_singleexcitation(theta: float):
        # CHECK: quantum.custom "SingleExcitation"

        qml.SingleExcitation(theta, wires=[0, 1])
        return measure(wires=0)

    print(decompose_singleexcitation.mlir)


test_decompose_singleexcitation()


def test_decompose_doubleexcitation():
    """
    Test that Double excitation is not decomposed.
    """
    dev = get_custom_device_without(4)

    @qjit(target="mlir")
    @qml.qnode(dev)
    # CHECK-LABEL: public @jit_decompose_doubleexcitation
    def decompose_doubleexcitation(theta: float):
        # CHECK: quantum.custom "DoubleExcitation"

        qml.DoubleExcitation(theta, wires=[0, 1, 2, 3])
        return measure(wires=0)

    print(decompose_doubleexcitation.mlir)


test_decompose_doubleexcitation()


def test_decompose_singleexcitationplus():
    """
    Test decomposition of single excitation plus.
    See
    https://github.com/PennyLaneAI/pennylane/blob/master/pennylane/ops/qubit/qchem_ops.py
    for the decomposition of qml.SingleExcitationPlus
    """
    dev = get_custom_device_without(2, discards={"SingleExcitationPlus", "C(SingleExcitationPlus)"})

    @qjit(target="mlir")
    @qml.qnode(dev)
    # CHECK-LABEL: public @jit_decompose_singleexcitationplus
    def decompose_singleexcitationplus(theta: float):
        # CHECK-NOT: "SingleExcitationPlus"
        # CHECK: quantum.custom "Hadamard"
        # CHECK: quantum.custom "CNOT"
        # CHECK: quantum.custom "RY"
        # CHECK: quantum.custom "RY"
        # CHECK: quantum.custom "CY"
        # CHECK: quantum.custom "S"
        # CHECK: quantum.custom "Hadamard"
        # CHECK: quantum.custom "RZ"
        # CHECK: quantum.custom "CNOT"
        # CHECK: quantum.gphase

        qml.SingleExcitationPlus(theta, wires=[0, 1])
        return measure(wires=0)

    print(decompose_singleexcitationplus.mlir)


test_decompose_singleexcitationplus()


def test_decompose_to_matrix():
    """Test decomposition of QubitUnitary"""
    dev = get_custom_device_without(1, force_matrix={"PauliY"})

    @qjit(target="mlir")
    @qml.qnode(dev)
    # CHECK-LABEL: public @jit_decompose_to_matrix
    def decompose_to_matrix():
        # CHECK: quantum.custom "PauliX"
        qml.PauliX(wires=0)
        # CHECK: quantum.unitary
        qml.PauliY(wires=0)
        # CHECK: quantum.custom "PauliZ"
        qml.PauliZ(wires=0)
        return measure(wires=0)

    print(decompose_to_matrix.mlir)


test_decompose_to_matrix()


def test_decomposition_rule_wire_param():
    """Test decomposition rule with passing a parameter that is a wire/integer"""

    @decomposition_rule(is_qreg=False)
    def Hadamard0(wire: WiresLike):
        qml.Hadamard(wire)

    qml.capture.enable()

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: module @circuit
    def circuit(_: float):
        # CHECK: func.func public @circuit([[ARG0:%.+]]
        # CHECK: [[QREG:%.+]] = quantum.alloc
        Hadamard0(int)
        return qml.probs()

    # CHECK: func.func public @Hadamard0([[QBIT:%.+]]: !quantum.bit) -> !quantum.bit
    # CHECK-NEXT: [[QUBIT_OUT:%.+]] = quantum.custom "Hadamard"() [[QBIT]] : !quantum.bit
    # CHECK-NEXT: return [[QUBIT_OUT]] : !quantum.bit

    print(circuit.mlir)

    qml.capture.disable()


test_decomposition_rule_wire_param()


def test_decomposition_rule_gate_param_param():
    """Test decomposition rule with passing a regular parameter"""

    @decomposition_rule(is_qreg=False, num_params=1)
    def RX_on_wire_0(param: TensorLike, w0: WiresLike):
        qml.RX(param, wires=w0)

    qml.capture.enable()

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: module @circuit_2
    def circuit_2(_: float):
        RX_on_wire_0(float, int)
        return qml.probs()

    # CHECK: func.func public @RX_on_wire_0([[PARAM_TENSOR:%.+]]: tensor<f64>, [[QUBIT:%.+]]: !quantum.bit) -> !quantum.bit
    # CHECK-NEXT: [[PARAM:%.+]] = tensor.extract [[PARAM_TENSOR]][] : tensor<f64>
    # CHECK-NEXT: [[QUBIT_1:%.+]] = quantum.custom "RX"([[PARAM]]) [[QUBIT]] : !quantum.bit
    # CHECK-NEXT: return [[QUBIT_1]] : !quantum.bit
    print(circuit_2.mlir)

    qml.capture.disable()


test_decomposition_rule_gate_param_param()


def test_multiple_decomposition_rules():
    """Test with multiple decomposition rules"""

    qml.capture.enable()

    @decomposition_rule
    def identity(): ...

    @decomposition_rule(is_qreg=True)
    def all_wires_rx(param: TensorLike, w0: WiresLike, w1: WiresLike, w2: WiresLike):
        qml.RX(param, wires=w0)
        qml.RX(param, wires=w1)
        qml.RX(param, wires=w2)

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit_3(_: float):
        # CHECK: [[QREG:%.+]] = quantum.alloc
        # CHECK-NEXT: [[QUBIT:%.+]] = quantum.extract [[QREG]][ 0] : !quantum.reg -> !quantum.bit
        # CHECK-NEXT: [[QUBIT_1:%.+]] = quantum.custom "Hadamard"() [[QUBIT]] : !quantum.bit
        # CHECK-NEXT: [[QREG_1:%.+]] = quantum.insert [[QREG]][ 0], [[QUBIT_1]] : !quantum.reg, !quantum.bit
        # CHECK-NEXT: quantum.compbasis qreg [[QREG_1]] : !quantum.obs
        identity()
        all_wires_rx(float, int, int, int)
        qml.Hadamard(0)
        return qml.probs()

    # CHECK: func.func public @identity
    # CHECK: func.func public @all_wires_rx

    print(circuit_3.mlir)
    qml.capture.disable()


test_multiple_decomposition_rules()


def test_decomposition_rule_shaped_wires():
    """Test decomposition rule with passing a shaped array of wires"""

    qml.capture.enable()

    @decomposition_rule(is_qreg=True)
    def shaped_wires_rule(param: TensorLike, wires: WiresLike):
        qml.RX(param, wires=wires[0])
        qml.RX(param, wires=wires[1])
        qml.RX(param, wires=wires[2])

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit_4(_: float):
        # CHECK: module @circuit_4
        shaped_wires_rule(float, jax.core.ShapedArray((3,), int))
        qml.Hadamard(0)
        return qml.probs()

    # CHECK: func.func public @shaped_wires_rule([[QREG:%.+]]: !quantum.reg, [[PARAM_TENSOR:%.+]]: tensor<f64>, [[QUBITS:%.+]]: tensor<3xi64>) -> !quantum.reg
    # CHECK-NEXT: [[IDX_0:%.+]] = stablehlo.slice [[QUBITS]] [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    # CHECK-NEXT: [[RIDX_0:%.+]] = stablehlo.reshape [[IDX_0]] : (tensor<1xi64>) -> tensor<i64>
    # CHECK-NEXT: [[EXTRACTED:%.+]] = tensor.extract [[RIDX_0]][] : tensor<i64>
    # CHECK-NEXT: [[QUBIT:%.+]] = quantum.extract [[QREG]][[[EXTRACTED]]] : !quantum.reg -> !quantum.bit
    # CHECK-NEXT: [[EXTRACTED_0:%.+]] = tensor.extract [[PARAM_TENSOR]][] : tensor<f64>
    # CHECK-NEXT: [[OUT_QUBITS:%.+]] = quantum.custom "RX"([[EXTRACTED_0]]) [[QUBIT]] : !quantum.bit

    print(circuit_4.mlir)
    qml.capture.disable()


test_decomposition_rule_shaped_wires()


def test_decomposition_rule_expanded_wires():
    """Test decomposition rule with passing expanding wires as a Python list"""

    qml.capture.enable()

    def shaped_wires_rule(param: TensorLike, wires: WiresLike):
        qml.RX(param, wires=wires[0])
        qml.RX(param, wires=wires[1])
        qml.RX(param, wires=wires[2])

    @decomposition_rule(is_qreg=False, num_params=1)
    def expanded_wires_rule(param: TensorLike, w1, w2, w3):
        shaped_wires_rule(param, [w1, w2, w3])

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit_5(_: float):
        # CHECK: module @circuit_5
        expanded_wires_rule(float, int, int, int)
        qml.Hadamard(0)
        return qml.probs()

    # CHECK: func.func public @expanded_wires_rule(%arg0: tensor<f64>, %arg1: !quantum.bit, %arg2: !quantum.bit, %arg3: !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.bit)

    print(circuit_5.mlir)
    qml.capture.disable()


test_decomposition_rule_expanded_wires()


def test_decomposition_rule_with_cond():
    """Test decomposition rule with a conditional path"""

    qml.capture.enable()

    @decomposition_rule(is_qreg=True)
    def cond_RX(param: TensorLike, w0: WiresLike):

        def true_path():
            qml.RX(param, wires=w0)

        def false_path(): ...

        qml.cond(param != 0.0, true_path, false_path)()

    @qml.qjit(autograph=False)
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit_6():
        # CHECK: module @circuit_6
        cond_RX(float, jax.core.ShapedArray((1,), int))
        return qml.probs()

    # CHECK: func.func public @cond_RX([[QREG:%.+]]: !quantum.reg, [[PARAM_TENSOR:%.+]]: tensor<f64>, [[QUBITS:%.+]]: tensor<1xi64>) -> !quantum.reg
    # CHECK-NEXT: [[ZERO:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    # CHECK-NEXT: [[COND_TENSOR:%.+]] = stablehlo.compare  NE, [[PARAM_TENSOR]], [[ZERO]],  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    # CHECK-NEXT: [[COND:%.+]] = tensor.extract [[COND_TENSOR]][] : tensor<i1>
    # CHECK-NEXT: [[RETVAL:%.+]] = scf.if [[COND]]
    # CHECK-DAG:        [[QUBIT:%.+]] = quantum.extract [[QREG]][%extracted_0] : !quantum.reg -> !quantum.bit
    # CHECK-DAG:        [[PARAM:%.+]] = tensor.extract [[PARAM_TENSOR]][] : tensor<f64>
    # CHECK:            [[QUBIT_0:%.+]] = quantum.custom "RX"([[PARAM]]) [[QUBIT]] : !quantum.bit
    # CHECK:            [[QREG_0:%.+]] = quantum.insert [[QREG]][%extracted_2], [[QUBIT_0]] : !quantum.reg, !quantum.bit
    # CHECK-NEXT:       scf.yield [[QREG_0]] : !quantum.reg
    # CHECK-NEXT: else
    # CHECK:            scf.yield [[QREG]] : !quantum.reg
    # CHECK:      return [[RETVAL]]

    print(circuit_6.mlir)
    qml.capture.disable()


test_decomposition_rule_with_cond()


def test_decomposition_rule_caller():
    """Test decomposition rules with a caller"""

    qml.capture.enable()

    @decomposition_rule(is_qreg=True)
    def rule_op1_decomp(_: TensorLike, wires: WiresLike):
        qml.Hadamard(wires=wires[0])
        qml.Hadamard(wires=[1])

    @decomposition_rule(is_qreg=True)
    def rule_op2_decomp(param: TensorLike, wires: WiresLike):
        qml.RX(param, wires=wires[0])

    def decomps_caller(param: TensorLike, wires: WiresLike):
        rule_op1_decomp(param, wires)
        rule_op2_decomp(param, wires)

    @qml.qjit(autograph=False)
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: module @circuit_7
    def circuit_7():
        # CHECK: [[QREG:%.+]] = quantum.alloc
        # CHECK: quantum.compbasis qreg [[QREG]] : !quantum.obs
        decomps_caller(float, jax.core.ShapedArray((2,), int))
        return qml.probs()

    # CHECK: func.func public @rule_op1_decomp(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<2xi64>) -> !quantum.reg
    # CHECK: func.func public @rule_op2_decomp(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<2xi64>) -> !quantum.reg
    print(circuit_7.mlir)
    qml.capture.disable()


test_decomposition_rule_caller()


def test_decompose_gateset_without_graph():
    """Test the decompose transform to a target gate set without the graph decomposition."""

    qml.capture.enable()

    @qml.qjit(target="mlir")
    @partial(qml.transforms.decompose, gate_set={"RX", "RZ"})
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: public @circuit_8() -> tensor<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode}
    def circuit_8():
        return qml.expval(qml.Z(0))

    print(circuit_8.mlir)

    qml.capture.disable()


test_decompose_gateset_without_graph()


def test_decompose_gateset_with_graph():
    """Test the decompose transform to a target gate set with the graph decomposition."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.qjit(target="mlir")
    @partial(qml.transforms.decompose, gate_set={"RX"})
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: public @simple_circuit_9() -> tensor<f64> attributes {decomp_gateset = ["RX"]
    def simple_circuit_9():
        return qml.expval(qml.Z(0))

    print(simple_circuit_9.mlir)

    @qml.qjit(target="mlir")
    @partial(qml.transforms.decompose, gate_set={"RX", "RZ"})
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: public @circuit_9() -> tensor<f64> attributes {decomp_gateset
    def circuit_9():
        return qml.expval(qml.Z(0))

    print(circuit_9.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


test_decompose_gateset_with_graph()


def test_decompose_gateset_operator_with_graph():
    """Test the decompose transform to a target gate set with the graph decomposition."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.qjit(target="mlir")
    @partial(qml.transforms.decompose, gate_set={qml.RX})
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: public @simple_circuit_10() -> tensor<f64> attributes {decomp_gateset = ["RX"]
    def simple_circuit_10():
        return qml.expval(qml.Z(0))

    print(simple_circuit_10.mlir)

    @qml.qjit(target="mlir")
    @partial(
        qml.transforms.decompose, gate_set={qml.RX, qml.RZ, "PauliZ", qml.PauliX, qml.Hadamard}
    )
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: public @circuit_10() -> tensor<f64> attributes {decomp_gateset
    def circuit_10():
        return qml.expval(qml.Z(0))

    print(circuit_10.mlir)

    @qml.qjit(target="mlir")
    @partial(
        qml.transforms.decompose, gate_set={qml.RX, qml.RZ, qml.PauliZ, qml.PauliX, qml.Hadamard}
    )
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: public @circuit_11() -> tensor<f64> attributes {decomp_gateset
    def circuit_11():
        return qml.expval(qml.Z(0))

    print(circuit_11.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


test_decompose_gateset_operator_with_graph()


def test_decompose_gateset_with_rotxzx():
    """Test the decompose transform with a custom operator with the graph decomposition."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.qjit(target="mlir")
    @partial(qml.transforms.decompose, gate_set={"RotXZX"})
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: public @simple_circuit_12() -> tensor<f64> attributes {decomp_gateset = ["RotXZX"]
    def simple_circuit_12():
        return qml.expval(qml.Z(0))

    print(simple_circuit_12.mlir)

    @qml.qjit(target="mlir")
    @partial(qml.transforms.decompose, gate_set={qml.ftqc.RotXZX})
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: public @circuit_12() -> tensor<f64> attributes {decomp_gateset = ["RotXZX"]
    def circuit_12():
        return qml.expval(qml.Z(0))

    print(circuit_12.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


test_decompose_gateset_with_rotxzx()


def test_decomposition_rule_name_update():
    """Test the name of the decomposition rule is updated in the MLIR output."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @decomposition_rule
    def _ry_to_rz_rx(phi, wires: WiresLike, **__):
        """Decomposition of RY gate using RZ and RX gates."""
        qml.RZ(-np.pi / 2, wires=wires)
        qml.RX(phi, wires=wires)
        qml.RZ(np.pi / 2, wires=wires)

    @decomposition_rule
    def _rot_to_rz_ry_rz(phi, theta, omega, wires: WiresLike, **__):
        """Decomposition of Rot gate using RZ and RY gates."""
        qml.RZ(phi, wires=wires)
        qml.RY(theta, wires=wires)
        qml.RZ(omega, wires=wires)

    @decomposition_rule
    def _u2_phaseshift_rot_decomposition(phi, delta, wires, **__):
        """Decomposition of U2 gate using Rot and PhaseShift gates."""
        pi_half = qml.math.ones_like(delta) * (np.pi / 2)
        qml.Rot(delta, pi_half, -delta, wires=wires)
        qml.PhaseShift(delta, wires=wires)
        qml.PhaseShift(phi, wires=wires)

    @decomposition_rule
    def _xzx_decompose(phi, theta, omega, wires, **__):
        """Decomposition of Rot gate using RX and RZ gates in XZX format."""
        qml.RX(phi, wires=wires)
        qml.RZ(theta, wires=wires)
        qml.RX(omega, wires=wires)

    @qml.qjit(target="mlir")
    @partial(qml.transforms.decompose, gate_set={"RX", "RZ", "PhaseShift"})
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    # CHECK: public @circuit_13() -> tensor<f64> attributes {decomp_gateset
    def circuit_13():
        _ry_to_rz_rx(float, int)
        _rot_to_rz_ry_rz(float, float, float, int)
        _u2_phaseshift_rot_decomposition(float, float, int)
        _xzx_decompose(float, float, float, int)
        return qml.expval(qml.Z(0))

    # CHECK: func.func public @rule_ry_to_rz_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<i64>) -> !quantum.reg
    # CHECK: func.func public @rule_rot_to_rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<i64>) -> !quantum.reg
    # CHECK: func.func public @rule_u2_phaseshift_rot_decomposition(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<i64>) -> !quantum.reg
    # CHECK: func.func public @rule_xzx_decompose(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<i64>) -> !quantum.reg
    print(circuit_13.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


test_decomposition_rule_name_update()
