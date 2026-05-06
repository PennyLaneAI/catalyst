# Copyright 2022-2025 Xanadu Quantum Technologies Inc.
import os
import pathlib
import platform
from copy import deepcopy
from functools import partial

import jax
import numpy as np
import pennylane as qp
from pennylane.devices.capabilities import OperatorProperties
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

from catalyst import measure, qjit
from catalyst.compiler import get_lib_path
from catalyst.device import get_device_capabilities
from catalyst.jax_primitives import decomposition_rule
from catalyst.passes import graph_decomposition

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
# pylint: disable=too-many-lines


TEST_PATH = os.path.dirname(__file__)
CONFIG_CUSTOM_DEVICE = pathlib.Path(f"{TEST_PATH}/../custom_device/custom_device.toml")


def get_custom_device_without(num_wires, discards=frozenset(), force_matrix=frozenset()):
    """Generate a custom device without gates in discards."""

    class CustomDevice(qp.devices.Device):
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
    @qp.qnode(dev)
    # CHECK-LABEL: @jit_decompose_multicontrolled_x1
    def decompose_multicontrolled_x1(theta: float):
        qp.RX(theta, wires=[0])
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK:     quantum.custom "PauliX"() {{%[a-zA-Z0-9_]+}} ctrls({{%[a-zA-Z0-9_]+}}, {{%[a-zA-Z0-9_]+}}, {{%[a-zA-Z0-9_]+}})
        # CHECK-NOT: name = "MultiControlledX"
        qp.MultiControlledX(wires=[0, 1, 2, 3])
        return qp.state()

    print(decompose_multicontrolled_x1.mlir)


test_decompose_multicontrolledx()


def test_decompose_rot():
    """Test decomposition of Rot gate."""
    dev = get_custom_device_without(1, discards={"Rot", "C(Rot)"})

    @qjit(target="mlir")
    @qp.qnode(dev)
    # CHECK-LABEL: @jit_decompose_rot
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
        qp.Rot(phi, theta, omega, wires=0)
        return measure(wires=0)

    print(decompose_rot.mlir)


test_decompose_rot()


def test_decompose_s():
    """Test decomposition of S gate."""
    dev = get_custom_device_without(1, discards={"S", "C(S)"})

    @qjit(target="mlir")
    @qp.qnode(dev)
    # CHECK-LABEL: @jit_decompose_s
    def decompose_s():
        # CHECK-NOT: name="S"
        # CHECK: [[pi_div_2:%.+]] = arith.constant 1.57079{{.+}} : f64
        # CHECK-NOT: name = "S"
        # CHECK: {{%.+}} = quantum.custom "PhaseShift"([[pi_div_2]])
        # CHECK-NOT: name = "S"
        qp.S(wires=0)
        return measure(wires=0)

    print(decompose_s.mlir)


test_decompose_s()


def test_decompose_qubitunitary():
    """Test decomposition of QubitUnitary"""
    dev = get_custom_device_without(1, discards={"QubitUnitary"})

    @qjit(target="mlir")
    @qp.qnode(dev)
    # CHECK-LABEL: @jit_decompose_qubit_unitary
    def decompose_qubit_unitary(U: jax.core.ShapedArray([2, 2], float)):
        # CHECK-NOT: name = "QubitUnitary"
        # CHECK: quantum.custom "RZ"
        # CHECK: quantum.custom "RY"
        # CHECK: quantum.custom "RZ"
        # CHECK-NOT: name = "QubitUnitary"
        qp.QubitUnitary(U, wires=0)
        return measure(wires=0)

    print(decompose_qubit_unitary.mlir)


test_decompose_qubitunitary()


def test_decompose_singleexcitation():
    """
    Test that single excitation is not decomposed.
    """
    dev = get_custom_device_without(2)

    @qjit(target="mlir")
    @qp.qnode(dev)
    # CHECK-LABEL: @jit_decompose_singleexcitation
    def decompose_singleexcitation(theta: float):
        # CHECK: quantum.custom "SingleExcitation"

        qp.SingleExcitation(theta, wires=[0, 1])
        return measure(wires=0)

    print(decompose_singleexcitation.mlir)


test_decompose_singleexcitation()


def test_decompose_doubleexcitation():
    """
    Test that Double excitation is not decomposed.
    """
    dev = get_custom_device_without(4)

    @qjit(target="mlir")
    @qp.qnode(dev)
    # CHECK-LABEL: @jit_decompose_doubleexcitation
    def decompose_doubleexcitation(theta: float):
        # CHECK: quantum.custom "DoubleExcitation"

        qp.DoubleExcitation(theta, wires=[0, 1, 2, 3])
        return measure(wires=0)

    print(decompose_doubleexcitation.mlir)


test_decompose_doubleexcitation()


def test_decompose_singleexcitationplus():
    """
    Test decomposition of single excitation plus.
    See
    https://github.com/PennyLaneAI/pennylane/blob/main/pennylane/ops/qubit/qchem_ops.py
    for the decomposition of qp.SingleExcitationPlus
    """
    dev = get_custom_device_without(2, discards={"SingleExcitationPlus", "C(SingleExcitationPlus)"})

    @qjit(target="mlir")
    @qp.qnode(dev)
    # CHECK-LABEL: @jit_decompose_singleexcitationplus
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

        qp.SingleExcitationPlus(theta, wires=[0, 1])
        return measure(wires=0)

    print(decompose_singleexcitationplus.mlir)


test_decompose_singleexcitationplus()


def test_decompose_to_matrix():
    """Test decomposition of QubitUnitary"""
    dev = get_custom_device_without(1, force_matrix={"PauliY"})

    @qjit(target="mlir")
    @qp.qnode(dev)
    # CHECK-LABEL: @jit_decompose_to_matrix
    def decompose_to_matrix():
        # CHECK: quantum.custom "PauliX"
        qp.PauliX(wires=0)
        # CHECK: quantum.unitary
        qp.PauliY(wires=0)
        # CHECK: quantum.custom "PauliZ"
        qp.PauliZ(wires=0)
        return measure(wires=0)

    print(decompose_to_matrix.mlir)


test_decompose_to_matrix()


def test_decomposition_rule_lowering():
    """Test that decomposition rules are lowered to private functions."""

    @decomposition_rule(is_qreg=True)
    def my_decomp():
        return

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit():
        # CHECK-LABEL: func.func private @my_decomp
        my_decomp()
        return

    print(circuit.mlir)


test_decomposition_rule_lowering()


def test_decomposition_rule_wire_param():
    """Test decomposition rule with passing a parameter that is a wire/integer"""

    @decomposition_rule(is_qreg=False)
    def Hadamard0(wire: WiresLike):
        qp.Hadamard(wire)

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK-LABEL: @circuit
    def circuit(_: float):
        # CHECK: @circuit([[ARG0:%.+]]
        # CHECK: [[QREG:%.+]] = quantum.alloc
        Hadamard0(int)
        return qp.probs()

    # CHECK: @Hadamard0([[QBIT:%.+]]: !quantum.bit) -> !quantum.bit
    # CHECK-NEXT: [[QUBIT_OUT:%.+]] = quantum.custom "Hadamard"() [[QBIT]] : !quantum.bit
    # CHECK-NEXT: return [[QUBIT_OUT]] : !quantum.bit

    print(circuit.mlir)


test_decomposition_rule_wire_param()


def test_decomposition_rule_gate_param_param():
    """Test decomposition rule with passing a regular parameter"""

    @decomposition_rule(is_qreg=False, num_params=1)
    def RX_on_wire_0(param: TensorLike, w0: WiresLike):
        qp.RX(param, wires=w0)

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK: module @circuit_2
    def circuit_2(_: float):
        RX_on_wire_0(float, int)
        return qp.probs()

    # CHECK: @RX_on_wire_0([[PARAM_TENSOR:%.+]]: tensor<f64>, [[QUBIT:%.+]]: !quantum.bit) -> !quantum.bit
    # CHECK-NEXT: [[PARAM:%.+]] = tensor.extract [[PARAM_TENSOR]][] : tensor<f64>
    # CHECK-NEXT: [[QUBIT_1:%.+]] = quantum.custom "RX"([[PARAM]]) [[QUBIT]] : !quantum.bit
    # CHECK-NEXT: return [[QUBIT_1]] : !quantum.bit
    print(circuit_2.mlir)


test_decomposition_rule_gate_param_param()


def test_multiple_decomposition_rules():
    """Test with multiple decomposition rules"""

    @decomposition_rule
    def identity(): ...

    @decomposition_rule(is_qreg=True)
    def all_wires_rx(param: TensorLike, w0: WiresLike, w1: WiresLike, w2: WiresLike):
        qp.RX(param, wires=w0)
        qp.RX(param, wires=w1)
        qp.RX(param, wires=w2)

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def circuit_3(_: float):
        # CHECK: [[QREG:%.+]] = quantum.alloc
        # CHECK-NEXT: [[QUBIT:%.+]] = quantum.extract [[QREG]][ 0] : !quantum.reg -> !quantum.bit
        # CHECK-NEXT: [[QUBIT_1:%.+]] = quantum.custom "Hadamard"() [[QUBIT]] : !quantum.bit
        # CHECK-NEXT: [[QREG_1:%.+]] = quantum.insert [[QREG]][ 0], [[QUBIT_1]] : !quantum.reg, !quantum.bit
        # CHECK-NEXT: quantum.compbasis qreg [[QREG_1]] : !quantum.obs
        identity()
        all_wires_rx(float, int, int, int)
        qp.Hadamard(0)
        return qp.probs()

    # CHECK-LABEL: @identity
    # CHECK-LABEL: @all_wires_rx

    print(circuit_3.mlir)


test_multiple_decomposition_rules()


def test_decomposition_rule_shaped_wires():
    """Test decomposition rule with passing a shaped array of wires"""

    @decomposition_rule(is_qreg=True)
    def shaped_wires_rule(param: TensorLike, wires: WiresLike):
        qp.RX(param, wires=wires[0])
        qp.RX(param, wires=wires[1])
        qp.RX(param, wires=wires[2])

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def circuit_4(_: float):
        # CHECK: module @circuit_4
        shaped_wires_rule(float, jax.core.ShapedArray((3,), int))
        qp.Hadamard(0)
        return qp.probs()

    # CHECK: @shaped_wires_rule([[QREG:%.+]]: !quantum.reg, [[PARAM_TENSOR:%.+]]: tensor<f64>, [[QUBITS:%.+]]: tensor<3xi64>) -> !quantum.reg
    # CHECK-NEXT: [[IDX_0:%.+]] = stablehlo.slice [[QUBITS]] [0:1] : (tensor<3xi64>) -> tensor<1xi64>
    # CHECK-NEXT: [[RIDX_0:%.+]] = stablehlo.reshape [[IDX_0]] : (tensor<1xi64>) -> tensor<i64>
    # CHECK-NEXT: [[EXTRACTED:%.+]] = tensor.extract [[RIDX_0]][] : tensor<i64>
    # CHECK-NEXT: [[QUBIT:%.+]] = quantum.extract [[QREG]][[[EXTRACTED]]] : !quantum.reg -> !quantum.bit
    # CHECK-NEXT: [[EXTRACTED_0:%.+]] = tensor.extract [[PARAM_TENSOR]][] : tensor<f64>
    # CHECK-NEXT: [[OUT_QUBITS:%.+]] = quantum.custom "RX"([[EXTRACTED_0]]) [[QUBIT]] : !quantum.bit

    print(circuit_4.mlir)


test_decomposition_rule_shaped_wires()


def test_decomposition_rule_expanded_wires():
    """Test decomposition rule with passing expanding wires as a Python list"""

    def shaped_wires_rule(param: TensorLike, wires: WiresLike):
        qp.RX(param, wires=wires[0])
        qp.RX(param, wires=wires[1])
        qp.RX(param, wires=wires[2])

    @decomposition_rule(is_qreg=False, num_params=1)
    def expanded_wires_rule(param: TensorLike, w1, w2, w3):
        shaped_wires_rule(param, [w1, w2, w3])

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def circuit_5(_: float):
        # CHECK: module @circuit_5
        expanded_wires_rule(float, int, int, int)
        qp.Hadamard(0)
        return qp.probs()

    # CHECK-LABEL: @expanded_wires_rule(%arg0: tensor<f64>, %arg1: !quantum.bit, %arg2: !quantum.bit, %arg3: !quantum.bit) -> (!quantum.bit, !quantum.bit, !quantum.bit)

    print(circuit_5.mlir)


test_decomposition_rule_expanded_wires()


def test_decomposition_rule_with_cond():
    """Test decomposition rule with a conditional path"""

    @decomposition_rule(is_qreg=True)
    def cond_RX(param: TensorLike, w0: WiresLike):

        def true_path():
            qp.RX(param, wires=w0)

        def false_path(): ...

        qp.cond(param != 0.0, true_path, false_path)()

    @qp.qjit(autograph=False, capture=True)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def circuit_6():
        # CHECK: module @circuit_6
        cond_RX(float, jax.core.ShapedArray((1,), int))
        return qp.probs()

    # CHECK: @cond_RX([[QREG:%.+]]: !quantum.reg, [[PARAM_TENSOR:%.+]]: tensor<f64>, [[QUBITS:%.+]]: tensor<1xi64>) -> !quantum.reg
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


test_decomposition_rule_with_cond()


def test_decomposition_rule_caller():
    """Test decomposition rules with a caller"""

    @decomposition_rule(is_qreg=True)
    def rule_op1_decomp(_: TensorLike, wires: WiresLike):
        qp.Hadamard(wires=wires[0])
        qp.Hadamard(wires=[1])

    @decomposition_rule(is_qreg=True)
    def rule_op2_decomp(param: TensorLike, wires: WiresLike):
        qp.RX(param, wires=wires[0])

    def decomps_caller(param: TensorLike, wires: WiresLike):
        rule_op1_decomp(param, wires)
        rule_op2_decomp(param, wires)

    @qp.qjit(autograph=False, capture=True)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK: module @circuit_7
    def circuit_7():
        # CHECK: [[QREG:%.+]] = quantum.alloc
        # CHECK: quantum.compbasis qreg [[QREG]] : !quantum.obs
        decomps_caller(float, jax.core.ShapedArray((2,), int))
        return qp.probs()

    # CHECK-LABEL: @rule_op1_decomp(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<2xi64>) -> !quantum.reg
    # CHECK-LABEL: @rule_op2_decomp(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<2xi64>) -> !quantum.reg
    print(circuit_7.mlir)


test_decomposition_rule_caller()


def test_decompose_gateset_with_graph():
    """Test the decompose transform to a target gate set with the graph decomposition."""

    qp.decomposition.enable_graph()

    @qp.qjit(target="mlir", capture=True)
    @partial(qp.transforms.decompose, gate_set={"RX"})
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK-LABEL: @simple_circuit_9() -> tensor<f64> attributes {decompose_gatesets
    def simple_circuit_9():
        return qp.expval(qp.Z(0))

    print(simple_circuit_9.mlir)

    @qp.qjit(target="mlir", capture=True)
    @partial(qp.transforms.decompose, gate_set={"RX", "RZ"})
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    # CHECK-LABEL: @circuit_9() -> tensor<f64> attributes {decompose_gatesets
    def circuit_9():
        return qp.expval(qp.Z(0))

    print(circuit_9.mlir)

    qp.decomposition.disable_graph()


test_decompose_gateset_with_graph()


def test_decompose_gateset_without_graph():
    """Test the decompose transform to a target gate set without the graph decomposition."""

    @qp.qjit(target="mlir", capture=True)
    @partial(qp.transforms.decompose, gate_set={"RX", "RZ"})
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK-LABEL: @circuit_8() -> tensor<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, quantum.node}
    def circuit_8():
        return qp.expval(qp.Z(0))

    print(circuit_8.mlir)


test_decompose_gateset_without_graph()


def test_decompose_gateset_operator_with_graph():
    """Test the decompose transform to a target gate set with the graph decomposition."""

    qp.decomposition.enable_graph()

    @qp.qjit(target="mlir", capture=True)
    @partial(qp.transforms.decompose, gate_set={qp.RX})
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK-LABEL: @simple_circuit_10() -> tensor<f64> attributes {decompose_gatesets
    def simple_circuit_10():
        return qp.expval(qp.Z(0))

    print(simple_circuit_10.mlir)

    @qp.qjit(target="mlir", capture=True)
    @partial(qp.transforms.decompose, gate_set={qp.RX, qp.RZ, "PauliZ", qp.PauliX, qp.Hadamard})
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK-LABEL: @circuit_10() -> tensor<f64> attributes {decompose_gatesets
    def circuit_10():
        return qp.expval(qp.Z(0))

    print(circuit_10.mlir)

    @qp.qjit(target="mlir", capture=True)
    @partial(qp.transforms.decompose, gate_set={qp.RX, qp.RZ, qp.PauliZ, qp.PauliX, qp.Hadamard})
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    # CHECK-LABEL: @circuit_11() -> tensor<f64> attributes {decompose_gatesets
    def circuit_11():
        return qp.expval(qp.Z(0))

    print(circuit_11.mlir)

    qp.decomposition.disable_graph()


test_decompose_gateset_operator_with_graph()


def test_decompose_gateset_with_rotxzx():
    """Test the decompose transform with a custom operator with the graph decomposition."""

    qp.decomposition.enable_graph()

    @qp.qjit(target="mlir", capture=True)
    @partial(qp.transforms.decompose, gate_set={"RotXZX"})
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK-LABEL: @simple_circuit_12() -> tensor<f64> attributes {decompose_gatesets
    def simple_circuit_12():
        return qp.expval(qp.Z(0))

    print(simple_circuit_12.mlir)

    @qp.qjit(target="mlir", capture=True)
    @partial(qp.transforms.decompose, gate_set={qp.ftqc.RotXZX})
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    # CHECK-LABEL: @circuit_12() -> tensor<f64> attributes {decompose_gatesets
    def circuit_12():
        return qp.expval(qp.Z(0))

    print(circuit_12.mlir)

    qp.decomposition.disable_graph()


test_decompose_gateset_with_rotxzx()


def test_decomposition_rule_name():
    """Test the name of the decomposition rule is not updated with circuit instantiation."""

    qp.decomposition.enable_graph()

    @decomposition_rule
    def _ry_to_rz_rx(phi, wires: WiresLike, **__):
        """Decomposition of RY gate using RZ and RX gates."""
        qp.RZ(-np.pi / 2, wires=wires)
        qp.RX(phi, wires=wires)
        qp.RZ(np.pi / 2, wires=wires)

    @decomposition_rule
    def _rot_to_rz_ry_rz(phi, theta, omega, wires: WiresLike, **__):
        """Decomposition of Rot gate using RZ and RY gates."""
        qp.RZ(phi, wires=wires)
        qp.RY(theta, wires=wires)
        qp.RZ(omega, wires=wires)

    @decomposition_rule
    def _u2_phaseshift_rot_decomposition(phi, delta, wires, **__):
        """Decomposition of U2 gate using Rot and PhaseShift gates."""
        pi_half = qp.math.ones_like(delta) * (np.pi / 2)
        qp.Rot(delta, pi_half, -delta, wires=wires)
        qp.PhaseShift(delta, wires=wires)
        qp.PhaseShift(phi, wires=wires)

    @decomposition_rule
    def _xzx_decompose(phi, theta, omega, wires, **__):
        """Decomposition of Rot gate using RX and RZ gates in XZX format."""
        qp.RX(phi, wires=wires)
        qp.RZ(theta, wires=wires)
        qp.RX(omega, wires=wires)

    @qp.qjit(target="mlir", capture=True)
    @partial(qp.transforms.decompose, gate_set={"RX", "RZ", "PhaseShift"})
    @qp.qnode(qp.device("lightning.qubit", wires=3))
    # CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    # CHECK-LABEL: @circuit_13() -> tensor<f64> attributes {decompose_gatesets
    def circuit_13():
        _ry_to_rz_rx(float, int)
        _rot_to_rz_ry_rz(float, float, float, int)
        _u2_phaseshift_rot_decomposition(float, float, int)
        _xzx_decompose(float, float, float, int)
        return qp.expval(qp.Z(0))

    # CHECK-LABEL: @_ry_to_rz_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<i64>) -> !quantum.reg
    # CHECK-LABEL: @_rot_to_rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<i64>) -> !quantum.reg
    # CHECK-LABEL: @_u2_phaseshift_rot_decomposition(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<i64>) -> !quantum.reg
    # CHECK-LABEL: @_xzx_decompose(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<i64>) -> !quantum.reg
    print(circuit_13.mlir)

    qp.decomposition.disable_graph()


test_decomposition_rule_name()


def test_decomposition_rule_name_update():
    """Test the name of the decomposition rule is updated in the MLIR output."""

    qp.decomposition.enable_graph()

    @qp.register_resources({qp.RZ: 2, qp.RX: 1})
    def rz_rx(phi, wires: WiresLike, **__):
        """Decomposition of RY gate using RZ and RX gates."""
        qp.RZ(-np.pi / 2, wires=wires)
        qp.RX(phi, wires=wires)
        qp.RZ(np.pi / 2, wires=wires)

    @qp.register_resources({qp.RZ: 2, qp.RY: 1})
    def rz_ry_rz(phi, theta, omega, wires: WiresLike, **__):
        """Decomposition of Rot gate using RZ and RY gates."""
        qp.RZ(phi, wires=wires)
        qp.RY(theta, wires=wires)
        qp.RZ(omega, wires=wires)

    @qp.register_resources({qp.RY: 1, qp.GlobalPhase: 1})
    def ry_gp(wires: WiresLike, **__):
        """Decomposition of PauliY gate using RY and GlobalPhase gates."""
        qp.RY(np.pi, wires=wires)
        qp.GlobalPhase(-np.pi / 2, wires=wires)

    @qp.qjit(target="mlir", capture=True)
    @partial(
        qp.transforms.decompose,
        gate_set={"RX", "RZ", "GlobalPhase"},
        fixed_decomps={
            qp.RY: rz_rx,
            qp.Rot: rz_ry_rz,
            qp.PauliY: ry_gp,
        },
    )
    @qp.qnode(qp.device("lightning.qubit", wires=3))
    # CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    # CHECK-LABEL: @circuit_14() -> tensor<f64> attributes {decompose_gatesets
    def circuit_14():
        qp.RY(0.5, wires=0)
        qp.Rot(0.1, 0.2, 0.3, wires=1)
        qp.PauliY(wires=2)
        return qp.expval(qp.Z(0))

    # CHECK-DAG: @rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) -> !quantum.reg
    # CHECK-DAG: @rz_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg
    # CHECK-DAG: @ry_gp(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg
    print(circuit_14.mlir)

    qp.decomposition.disable_graph()


test_decomposition_rule_name_update()


def test_decomposition_inside_subroutine():
    """Test that operators inside subroutines can be decomposed."""

    qp.decomposition.enable_graph()

    @qp.templates.Subroutine
    def f(x, wires):
        qp.IsingXX(x, wires)

    @qp.qjit(capture=True, target="mlir")
    @qp.decompose(gate_set=qp.gate_sets.ROTATIONS_PLUS_CNOT)
    @qp.qnode(qp.device("lightning.qubit", wires=5))
    # CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    def subroutine_circuit():
        # CHECK-DAG: [[FIRST_CONST:%.+]] = stablehlo.constant dense<5.000000e-01> : tensor<f64>
        # CHECK-DAG: [[SECOND_CONST:%.+]] = stablehlo.constant dense<1.200000e+00> : tensor<f64>

        # CHECK: [[QREG:%.+]] = quantum.alloc
        # CHECK: [[QREG_1:%.+]] = call @f([[QREG]], [[FIRST_CONST]], {{%.+}}) : (!quantum.reg, tensor<f64>, tensor<2xi64>) -> !quantum.reg
        # CHECK: [[QREG_2:%.+]] = call @f([[QREG_1]], [[SECOND_CONST]], {{%.+}}) : (!quantum.reg, tensor<f64>, tensor<2xi64>) -> !quantum.reg

        f(0.5, (0, 1))
        f(1.2, (2, 3))
        return qp.probs(wires=0)

    # CHECK-DAG: @_isingxx_to_cnot_rx_cnot(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<2xi64>)
    print(subroutine_circuit.mlir)
    qp.decomposition.disable_graph()


test_decomposition_inside_subroutine()


def test_decomposition_rule_name_update_multi_qubits():
    """Test the name of the decomposition rule with multi-qubit gates."""

    qp.decomposition.enable_graph()

    @qp.qjit(target="mlir", capture=True)
    @partial(
        qp.transforms.decompose,
        gate_set={"RY", "RX", "CNOT", "Hadamard", "GlobalPhase"},
    )
    @qp.qnode(qp.device("lightning.qubit", wires=4))
    # CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    # CHECK-LABEL: @circuit_15() -> tensor<f64> attributes {decompose_gatesets
    def circuit_15():
        qp.SingleExcitation(0.5, wires=[0, 1])
        qp.SingleExcitationPlus(0.5, wires=[0, 1])
        qp.SingleExcitationMinus(0.5, wires=[0, 1])
        qp.DoubleExcitation(0.5, wires=[0, 1, 2, 3])
        return qp.expval(qp.Z(0))

    # CHECK-DAG: @_cry(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "CRY"}
    # CHECK-DAG: @_s_phaseshift(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "S"}
    # CHECK-DAG: @_phaseshift_to_rz_gp(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "PhaseShift"}
    # CHECK-DAG: @_rz_to_ry_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RZ"}
    # CHECK-DAG: @_rot_to_rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    # CHECK-DAG: @_doublexcit(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<4xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 4 : i64, target_gate = "DoubleExcitation"}
    # CHECK-DAG: @_single_excitation_decomp(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "SingleExcitation"}
    print(circuit_15.mlir)

    qp.decomposition.disable_graph()


test_decomposition_rule_name_update_multi_qubits()


def test_decomposition_rule_name_adjoint():
    """Test decomposition rule with qp.adjoint."""

    qp.decomposition.enable_graph()

    @qp.qjit(target="mlir", capture=True)
    @partial(
        qp.transforms.decompose,
        gate_set={"RY", "RX", "CZ", "GlobalPhase", "Adjoint(SingleExcitation)"},
    )
    @qp.qnode(qp.device("lightning.qubit", wires=4))
    # CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    def circuit_16(x: float):
        # CHECK-DAG: %1 = quantum.adjoint(%0) : !quantum.reg
        # CHECK-DAG: %2 = quantum.adjoint(%1) : !quantum.reg
        # CHECK-DAG: %3 = quantum.adjoint(%2) : !quantum.reg
        # CHECK-DAG: %4 = quantum.adjoint(%3) : !quantum.reg
        qp.adjoint(qp.CNOT)(wires=[0, 1])
        qp.adjoint(qp.Hadamard)(wires=2)
        qp.adjoint(qp.RZ)(0.5, wires=3)
        qp.adjoint(qp.SingleExcitation)(0.1, wires=[0, 1])
        qp.adjoint(qp.SingleExcitation(x, wires=[0, 1]))
        return qp.expval(qp.Z(0))

    # CHECK-DAG: @_single_excitation_decomp(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "SingleExcitation"}
    # CHECK-DAG: @_hadamard_to_rz_ry(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Hadamard"}
    # CHECK-DAG: @_rz_to_ry_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RZ"}
    # CHECK-DAG: @_rot_to_rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    # CHECK-DAG: @_cnot_to_cz_h(%arg0: !quantum.reg, %arg1: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "CNOT"}
    print(circuit_16.mlir)

    qp.decomposition.disable_graph()


test_decomposition_rule_name_adjoint()


# TODO: Reenable this once the underlying non-determinism issue is resolved
def test_decomposition_rule_name_ctrl():
    """Test decomposition rule with qp.ctrl."""

    qp.decomposition.enable_graph()

    @qp.qjit(target="mlir", capture=True)
    @partial(
        qp.transforms.decompose,
        gate_set={"RX", "RZ", "H", "CZ"},
    )
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    # SKIP-CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    # SKIP-CHECK{LITERAL}: @circuit_17() -> tensor<f64> attributes {decompose_gatesets
    def circuit_17():
        # SKIP-CHECK: %out_qubits:2 = quantum.custom "CRY"(%cst) %1, %2 : !quantum.bit, !quantum.bit
        # SKIP-CHECK-NEXT: %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits#0, %out_qubits#1 : !quantum.bit, !quantum.bit
        qp.ctrl(qp.RY, control=0)(0.5, 1)
        qp.ctrl(qp.PauliX, control=0)(1)
        return qp.expval(qp.Z(0))

    # SKIP-CHECK-DAG: @_cnot_to_cz_h(%arg0: !quantum.reg, %arg1: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "CNOT"}
    # SKIP-CHECK-DAG: @_cry(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "CRY"}
    # SKIP-CHECK-DAG: @_ry_to_rz_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RY"}
    # SKIP-CHECK-DAG: @_rot_to_rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    # print(circuit_17.mlir)

    qp.decomposition.disable_graph()


test_decomposition_rule_name_ctrl()


# TODO: Reenable this once the underlying non-determinism issue is resolved
def test_qft_decomposition():
    """Test the decomposition of the QFT"""

    qp.decomposition.enable_graph()

    @qp.qjit(autograph=True, target="mlir", capture=True)
    @partial(
        qp.transforms.decompose,
        gate_set={"RX", "RY", "CNOT", "GlobalPhase"},
    )
    @qp.qnode(qp.device("lightning.qubit", wires=4))
    # SKIP-CHECK: %0 = transform.apply_registered_pass "decompose-lowering"
    # SKIP-CHECK: @circuit_18(%arg0: tensor<3xf64>) -> tensor<f64> attributes {decompose_gatesets
    def circuit_18():
        # %6 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %0) -> (!quantum.reg) {
        # %23 = scf.for %arg3 = %c0 to %22 step %c1 iter_args(%arg4 = %21) -> (!quantum.reg) {
        # %7 = scf.for %arg1 = %c0 to %c2 step %c1 iter_args(%arg2 = %6) -> (!quantum.reg) {
        qp.QFT(wires=[0, 1, 2, 3])
        return qp.expval(qp.Z(0))

    # SKIP-CHECK-DAG: @ag___cphase_to_rz_cnot(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "ControlledPhaseShift"}
    # SKIP-CHECK-DAG: @ag___rz_to_ry_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RZ"}
    # SKIP-CHECK-DAG: @ag___rot_to_rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    # SKIP-CHECK-DAG: @ag___swap_to_cnot(%arg0: !quantum.reg, %arg1: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "SWAP"}
    # SKIP-CHECK-DAG: @ag___hadamard_to_rz_ry(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Hadamard"}
    # print(circuit_18.mlir)

    qp.decomposition.disable_graph()


test_qft_decomposition()


def test_decompose_lowering_with_other_passes():
    """Test the decompose lowering pass with other passes in a pass pipeline."""

    qp.decomposition.enable_graph()

    @qp.qjit(target="mlir", capture=True)
    @qp.transforms.merge_rotations
    @qp.transforms.cancel_inverses
    @partial(
        qp.transforms.decompose,
        gate_set={"RZ", "RY", "CNOT", "GlobalPhase"},
    )
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK: module attributes {transform.with_named_sequence} {
    # CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
    # CHECK-NEXT:   [[ONE:%.+]] = transform.apply_registered_pass "decompose-lowering" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
    # CHECK-NEXT:   [[TWO:%.+]] = transform.apply_registered_pass "cancel-inverses" to [[ONE]] : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
    # CHECK-NEXT:   {{%.+}} = transform.apply_registered_pass "merge-rotations" to [[TWO]] : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
    # CHECK-NEXT:   transform.yield
    # CHECK-NEXT:   }
    def circuit_19():

        # CHECK: [[OUT_0:%.+]] = quantum.custom "PauliX"() %1 : !quantum.bit
        # CHECK-NEXT: [[OUT_1:%.+]] = quantum.custom "PauliX"() [[OUT_0]] : !quantum.bit
        # CHECK-NEXT: [[OUT_2:%.+]] = quantum.custom "RX"(%cst_0) [[OUT_1]] : !quantum.bit
        # CHECK-NEXT: {{%.+}} = quantum.custom "RX"(%cst) [[OUT_2]] : !quantum.bit
        qp.PauliX(0)
        qp.PauliX(0)
        qp.RX(0.1, wires=0)
        qp.RX(-0.1, wires=0)
        return qp.expval(qp.PauliX(0))

    # CHECK-DAG: @_paulix_to_rx(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "PauliX"}
    # CHECK-DAG: @_rx_to_rz_ry(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RX"}
    print(circuit_19.mlir)

    qp.decomposition.disable_graph()


test_decompose_lowering_with_other_passes()


def test_decompose_lowering_multirz():
    """Test the decompose lowering pass with MultiRZ in the gate set."""

    qp.decomposition.enable_graph()

    @qp.qjit(target="mlir", capture=True)
    @partial(
        qp.transforms.decompose,
        gate_set={"CNOT", "RZ"},
    )
    @qp.qnode(qp.device("lightning.qubit", wires=3))
    # CHECK:  %0 = transform.apply_registered_pass "decompose-lowering"
    def circuit_20(x: float):
        # CHECK:  [[EXTRACTED:%.+]] = tensor.extract %arg0[] : tensor<f64>
        # CHECK-NEXT:  [[OUT_QUBITS:%.+]] = quantum.multirz([[EXTRACTED]]) %1 : !quantum.bit
        # CHECK-NEXT:  [[BIT_1:%.+]] = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
        # CHECK-NEXT:  [[EXTRACTED_0:%.+]] = tensor.extract %arg0[] : tensor<f64>
        # CHECK-NEXT:  [[OUT_QUBITS_1:%.+]] = quantum.multirz([[EXTRACTED_0]]) [[OUT_QUBITS]], [[BIT_1]] : !quantum.bit, !quantum.bit
        # CHECK-NEXT:  [[BIT_2:%.+]] = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
        # CHECK-NEXT:  [[EXTRACTED_2:%.+]] = tensor.extract %arg0[] : tensor<f64>
        # CHECK-NEXT:  {{%.+}} = quantum.multirz([[EXTRACTED_2]]) {{%.+}}, {{%.+}}, [[BIT_2]] : !quantum.bit, !quantum.bit, !quantum.bit
        qp.MultiRZ(x, wires=[0])
        qp.MultiRZ(x, wires=[0, 1])
        qp.MultiRZ(x, wires=[1, 0, 2])
        return qp.expval(qp.PauliX(0))

    # CHECK-DAG: @_multi_rz_decomposition_wires_1(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "MultiRZ"}
    # CHECK-DAG: @_multi_rz_decomposition_wires_2(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "MultiRZ"}
    # CHECK-DAG: @_multi_rz_decomposition_wires_3(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<3xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 3 : i64, target_gate = "MultiRZ"}
    # CHECK-DAG: %0 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %arg0) -> (!quantum.reg)
    # CHECK-DAG:   %5 = scf.for %arg3 = %c1 to %c3 step %c1 iter_args(%arg4 = %4) -> (!quantum.reg)
    print(circuit_20.mlir)

    qp.decomposition.disable_graph()


test_decompose_lowering_multirz()


def test_decompose_lowering_with_ordered_passes():
    """Test the decompose lowering pass with other passes in a specific order in a pass pipeline."""

    qp.decomposition.enable_graph()

    @qp.qjit(target="mlir", capture=True)
    @partial(
        qp.transforms.decompose,
        gate_set={"RZ", "RY", "CNOT", "GlobalPhase"},
    )
    @qp.transforms.merge_rotations
    @qp.transforms.cancel_inverses
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK: module attributes {transform.with_named_sequence} {
    # CHECK-NEXT:     transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
    # CHECK-NEXT:     [[FIRST:%.+]] = transform.apply_registered_pass "cancel-inverses" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
    # CHECK-NEXT:     [[SECOND:%.+]] = transform.apply_registered_pass "merge-rotations" to [[FIRST]] : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
    # CHECK-NEXT:     {{%.+}} = transform.apply_registered_pass "decompose-lowering" to [[SECOND]] : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
    # CHECK-NEXT:     transform.yield
    # CHECK-NEXT: }
    def circuit_21(x: float):
        # CHECK: [[OUT:%.+]] = quantum.custom "PauliX"() %1 : !quantum.bit
        # CHECK-NEXT: [[OUT_0:%.+]] = quantum.custom "PauliX"() [[OUT]] : !quantum.bit
        # CHECK-NEXT: [[EXTRACTED:%.+]] = tensor.extract %arg0[] : tensor<f64>
        # CHECK-NEXT: [[OUT_1:%.+]] = quantum.custom "RX"([[EXTRACTED]]) [[OUT_0]] : !quantum.bit
        # CHECK-NEXT: [[NEGATED:%.+]] = stablehlo.negate %arg0 : tensor<f64>
        # CHECK-NEXT: [[EXTRACTED_2:%.+]] = tensor.extract [[NEGATED]][] : tensor<f64>
        # CHECK-NEXT: {{%.+}} = quantum.custom "RX"([[EXTRACTED_2]]) [[OUT_1]] : !quantum.bit
        qp.PauliX(0)
        qp.PauliX(0)
        qp.RX(x, wires=0)
        qp.RX(-x, wires=0)
        return qp.expval(qp.PauliX(0))

    # CHECK-DAG: @_paulix_to_rx(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "PauliX"}
    # CHECK-DAG: @_rx_to_rz_ry(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RX"}
    # CHECK-DAG: @_rot_to_rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    print(circuit_21.mlir)

    qp.decomposition.disable_graph()


test_decompose_lowering_with_ordered_passes()


def test_decompose_lowering_with_gphase():
    """Test the decompose lowering pass with GlobalPhase."""

    qp.decomposition.enable_graph()

    @qp.qjit(target="mlir", capture=True)
    @partial(
        qp.transforms.decompose,
        gate_set={"RX", "RY", "GlobalPhase"},
    )
    @qp.qnode(qp.device("lightning.qubit", wires=3))
    # CHECK:  %0 = transform.apply_registered_pass "decompose-lowering"
    def circuit_22():
        # CHECK:  quantum.gphase(%cst_0)
        # CHECK-NEXT:  [[EXTRACTED:%.+]] = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
        # CHECK-NEXT:  [[OUT_QUBITS:%.+]] = quantum.custom "PhaseShift"(%cst) [[EXTRACTED]] : !quantum.bit
        # CHECK-NEXT:  {{%.+}} = quantum.custom "PhaseShift"(%cst) [[OUT_QUBITS]] : !quantum.bit
        qp.GlobalPhase(0.5)
        qp.ctrl(qp.GlobalPhase, control=0)(0.3)
        qp.ctrl(qp.GlobalPhase, control=0)(phi=0.3, wires=[1, 2])
        return qp.expval(qp.PauliX(0))

    # CHECK-DAG: @_phaseshift_to_rz_gp(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "PhaseShift"}
    # CHECK-DAG: @_rz_to_ry_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RZ"}
    print(circuit_22.mlir)

    qp.decomposition.disable_graph()


test_decompose_lowering_with_gphase()


def test_decompose_lowering_alt_decomps():
    """Test the decompose lowering pass with alternative decompositions."""

    qp.decomposition.enable_graph()

    @qp.register_resources({qp.RY: 1})
    def custom_rot_cheap(params, wires: WiresLike):
        qp.RY(params[1], wires=wires)

    @qp.qjit(target="mlir", capture=True)
    @partial(
        qp.transforms.decompose,
        gate_set={"RY", "RZ"},
        alt_decomps={qp.Rot: [custom_rot_cheap]},
    )
    @qp.qnode(qp.device("lightning.qubit", wires=3), shots=1000)
    def circuit_23(x: float, y: float):
        qp.Rot(x, y, x + y, wires=1)
        return qp.expval(qp.PauliZ(0))

    # CHECK-DAG: @custom_rot_cheap(%arg0: !quantum.reg, %arg1: tensor<3xf64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    print(circuit_23.mlir)

    qp.decomposition.disable_graph()


test_decompose_lowering_alt_decomps()


def test_decompose_lowering_with_tensorlike():
    """Test the decompose lowering pass with fixed decompositions
    using TensorLike parameters."""

    qp.decomposition.enable_graph()

    @qp.register_resources({qp.RZ: 2, qp.RY: 1})
    def custom_rot(params: TensorLike, wires: WiresLike):
        qp.RZ(params[0], wires=wires)
        qp.RY(params[1], wires=wires)
        qp.RZ(params[2], wires=wires)

    @qp.register_resources({qp.RZ: 1, qp.CNOT: 4})
    def custom_multirz(params: TensorLike, wires: WiresLike):
        qp.CNOT(wires=(wires[2], wires[1]))
        qp.CNOT(wires=(wires[1], wires[0]))
        qp.RZ(params[0], wires=wires[0])
        qp.CNOT(wires=(wires[1], wires[0]))
        qp.CNOT(wires=(wires[2], wires[1]))

    @qp.qjit(target="mlir", capture=True)
    @partial(
        qp.transforms.decompose,
        gate_set={"RY", "RX", qp.CNOT},
        fixed_decomps={qp.Rot: custom_rot, qp.MultiRZ: custom_multirz},
    )
    @qp.qnode(qp.device("lightning.qubit", wires=3), shots=1000)
    def circuit_24(x: float, y: float):
        qp.Rot(x, y, x + y, wires=1)
        qp.MultiRZ(x + y, wires=[0, 1, 2])
        return qp.expval(qp.PauliZ(0))

    # CHECK-DAG: @custom_multirz_wires_3(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<3xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 3 : i64, target_gate = "MultiRZ"}
    # CHECK-DAG: @_rz_to_ry_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RZ"}
    # CHECK-DAG: @custom_rot(%arg0: !quantum.reg, %arg1: tensor<3xf64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    print(circuit_24.mlir)

    qp.decomposition.disable_graph()


test_decompose_lowering_with_tensorlike()


def test_decompose_lowering_fallback():
    """Test the decompose lowering pass when the graph is failed."""

    qp.decomposition.enable_graph()

    @qp.qjit(target="mlir", capture=True)
    @partial(qp.transforms.decompose, gate_set={qp.RX, qp.RZ})
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    # CHECK-LABEL: @circuit_25()
    def circuit_25():
        # CHECK: [[OUT_QUBIT:%.+]] = quantum.custom "RZ"(%cst) {{%.+}} : !quantum.bit
        # CHECK-NEXT: [[OUT_QUBIT_0:%.+]] = quantum.custom "RX"(%cst) [[OUT_QUBIT]] : !quantum.bit
        # CHECK-NEXT: [[OUT_QUBIT_1:%.+]] = quantum.custom "RZ"(%cst) [[OUT_QUBIT_0]] : !quantum.bit
        qp.Hadamard(0)
        return qp.state()

    print(circuit_25.mlir)

    qp.decomposition.disable_graph()


test_decompose_lowering_fallback()


def test_decompose_lowering_params_ordering():
    """Test the order of params and wires in the captured decomposition rule."""

    qp.decomposition.enable_graph()

    @qjit(target="mlir", capture=True)
    @partial(qp.transforms.decompose, gate_set=[qp.RX, qp.RY, qp.RZ])
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    # CHECK-LABEL: @circuit_26(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>)
    def circuit_26(x: float, y: float, z: float):
        qp.Rot(x, y, z, wires=0)
        return qp.expval(qp.PauliZ(0))

    # CHECK-LABEL: @_rot_to_rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    # CHECK:  [[EXTRACTED_0:%.+]] = tensor.extract %arg1[] : tensor<f64>
    # CHECK-NEXT:  [[OUT_QUBITS:%.+]] = quantum.custom "RZ"([[EXTRACTED_0]]) {{%.+}} : !quantum.bit
    # CHECK:  [[EXTRACTED_3:%.+]] = tensor.extract %arg2[] : tensor<f64>
    # CHECK-NEXT:  [[OUT_QUBITS_4:%.+]] = quantum.custom "RY"([[EXTRACTED_3]]) {{%.+}} : !quantum.bit
    # CHECK:  [[EXTRACTED_7:%.+]] = tensor.extract %arg3[] : tensor<f64>
    # CHECK-NEXT:  [[OUT_QUBITS_8:%.+]] = quantum.custom "RZ"([[EXTRACTED_7]]) {{%.+}} : !quantum.bit
    # CHECK:  return {{%.+}} : !quantum.reg
    print(circuit_26.mlir)

    qp.decomposition.disable_graph()


test_decompose_lowering_params_ordering()


def test_decomposition_rule_with_allocation():
    """Test decomposition rule with dynamic qubit allocation"""

    @decomposition_rule(is_qreg=True)
    def Hadamard0_with_alloc(wire: WiresLike):
        with qp.allocate(1) as q:
            qp.X(q[0])
            qp.CNOT(wires=[q[0], wire])

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    # CHECK: module @circuit_27
    def circuit_27():
        Hadamard0_with_alloc(int)
        return qp.probs()

    # CHECK-LABEL: @Hadamard0_with_alloc(%arg0: !quantum.reg, %arg1: tensor<i64>) -> !quantum.reg
    # CHECK:  [[dynalloc_qreg:%.+]] = quantum.alloc( 1)
    # CHECK:  [[dynalloc_bit0:%.+]] = quantum.extract [[dynalloc_qreg]][ 0]
    # CHECK:  [[xout:%.+]] = quantum.custom "PauliX"() [[dynalloc_bit0]]
    # CHECK:  [[detensor:%.+]] = tensor.extract %arg1[]
    # CHECK:  [[glob_bit:%.+]] = quantum.extract %arg0[[[detensor]]]
    # CHECK:  [[cnot_out:%.+]]:2 = quantum.custom "CNOT"() [[xout]], [[glob_bit]]
    # CHECK:  [[dynalloc_qreg_inserted:%.+]] = quantum.insert [[dynalloc_qreg]][ 0], [[cnot_out]]#0
    # CHECK:  quantum.dealloc [[dynalloc_qreg_inserted]] : !quantum.reg
    # CHECK:  [[detensor:%.+]] = tensor.extract %arg1[]
    # CHECK:  [[glob_insert:%.+]] = quantum.insert %arg0[[[detensor]]], [[cnot_out]]#1
    # CHECK:  return [[glob_insert]] : !quantum.reg

    print(circuit_27.mlir)


test_decomposition_rule_with_allocation()


def test_decompose_autograph_multi_blocks():
    """Test the decompose lowering pass with autograph in the program and rule."""

    qp.decomposition.enable_graph()

    def _multi_rz_decomposition_resources(num_wires):
        """Resources required for MultiRZ decomposition."""
        return {qp.RZ: 1, qp.CNOT: 2 * (num_wires - 1)}

    @qp.register_resources(_multi_rz_decomposition_resources)
    @qp.capture.run_autograph
    def _multi_rz_decomposition(theta: TensorLike, wires: WiresLike, **__):
        """Decomposition of MultiRZ using CNOTs and RZs."""
        for i in range(len(wires) - 1):
            qp.CNOT(wires=(wires[i], wires[i + 1]))
        qp.RZ(theta, wires=wires[0])
        for i in range(len(wires) - 1, 0, -1):
            qp.CNOT(wires=(wires[i], wires[i - 1]))

    @qp.qjit(target="mlir", capture=True)
    @partial(
        qp.transforms.decompose,
        gate_set={"RZ", "CNOT"},
        fixed_decomps={qp.MultiRZ: _multi_rz_decomposition},
    )
    @qp.qnode(qp.device("lightning.qubit", wires=5))
    def circuit_29(n: int):

        # CHECK: {{%.+}} = scf.for %arg1 = {{%.+}} to %1 step {{%.+}} iter_args(%arg2 = %0) -> (!quantum.reg) {
        @qp.for_loop(n)
        def f(i):  # pylint: disable=unused-argument
            qp.MultiRZ(0.5, wires=[0, 1, 2, 3, 4])

        f()  # pylint: disable=no-value-for-parameter

        return qp.expval(qp.Z(0))

    # CHECK-LABEL: @ag___multi_rz_decomposition_wires_5(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<5xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 5 : i64, target_gate = "MultiRZ"}
    # CHECK: {{%.+}} = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %arg0) -> (!quantum.reg)
    # CHECK: {{%.+}} = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %4) -> (!quantum.reg)
    print(circuit_29.mlir)

    qp.decomposition.disable_graph()


test_decompose_autograph_multi_blocks()


def test_decompose_work_wires_context_manager():
    """
    Test that decomposition with work wires is correctly applied when allocating with the context
    manager.
    """

    @decomposition_rule(is_qreg=True, op_type="PauliZ")
    def my_decomp(wires):
        with qp.allocate(2, restored=False) as work_wires:
            qp.X(wires[0])
            qp.X(wires[1])
            qp.H(work_wires[0])
            qp.H(work_wires[1])

    @qp.qjit(capture=True)
    @qp.transform(pass_name="decompose-lowering")
    @qp.qnode(qp.device("lightning.qubit", wires=3))
    def my_circuit():
        my_decomp(jax.core.ShapedArray((2,), int))
        qp.Z(0)
        return qp.probs()

    # check that decomp arrives properly
    # CHECK-LABEL: @my_decomp({{.*}}) -> !quantum.reg attributes {{{.*}} target_gate = "PauliZ"}
    print(my_circuit.mlir)

    # check that decomp is applied properly
    # CHECK-NOT: PauliZ
    # CHECK-NOT: my_decomp

    # two allocates, one for main register and one for decomp register
    # CHECK: allocate
    # CHECK: allocate
    # CHECK: PauliX
    # CHECK: PauliX
    # CHECK: Hadamard
    # CHECK: Hadamard
    # CHECK: release
    # CHECK: release
    print(my_circuit.mlir_opt)


test_decompose_work_wires_context_manager()


def test_decompose_work_wires_alloc_dealloc():
    """
    Test that decomposition with work wires is correctly applied when allocating/deallocating
    explicitly.
    """

    @decomposition_rule(is_qreg=True, op_type="RY")
    def my_decomp(angle, wires):
        work_wires = qp.allocate(2)
        qp.CNOT((work_wires[0], wires[0]))
        qp.RX(-np.pi / 2, wires[0])
        qp.RZ(angle, wires[0])
        qp.RX(np.pi / 2, wires[0])
        qp.CNOT((work_wires[1], wires[1]))
        qp.deallocate(work_wires)

    @qp.qjit(capture=True)
    @qp.transform(pass_name="decompose-lowering")
    @qp.qnode(qp.device("lightning.qubit", wires=3))
    def my_circuit(angle: float):
        my_decomp(float, jax.core.ShapedArray((2,), int))
        qp.RY(angle, 0)
        return qp.probs()

    # check that decomp arrives properly
    # CHECK-LABEL: @my_decomp({{.*}}) -> !quantum.reg attributes {{{.*}} target_gate = "RY"}
    print(my_circuit.mlir)

    # check that the decomposition applies properly
    # CHECK-NOT: my_decomp
    # CHECK-NOT: RY

    # two allocates, one for main register and one for decomp register
    # CHECK: allocate
    # CHECK: allocate
    # CHECK: CNOT
    # CHECK: RX
    # CHECK: RZ
    # CHECK: RX
    # CHECK: CNOT
    # CHECK: release
    # CHECK: release
    print(my_circuit.mlir_opt)


test_decompose_work_wires_alloc_dealloc()


def test_decompose_work_wires_control_flow():
    """Test that decomposition with work wires + control flow is correctly applied."""

    @decomposition_rule(is_qreg=True, op_type="CRX")
    def my_decomp(angle, wires, **_):
        def true_func():
            qp.CNOT(wires)

            with qp.allocate(2, state="any", restored=True) as w:
                for _ in range(2):
                    qp.H(w[0])
                    qp.X(w[1])

        def false_func():
            with qp.allocate(1, state="any", restored=False) as w:
                qp.H(w)

                m = qp.measure(wires[0])

                qp.cond(m, qp.CNOT)(wires)

        qp.cond(angle > 1.2, true_func, false_func)()

    @qp.qjit(capture=True)
    @qp.transform(pass_name="decompose-lowering")
    @qp.qnode(qp.device("lightning.qubit", wires=4))
    def circuit():
        my_decomp(float, jax.core.ShapedArray((2,), int))
        qp.CRX(1.7, wires=[0, 1])
        qp.CRX(-7.2, wires=[0, 1])
        return qp.state()

    # target_gate attribute is correctly applied
    # CHECK: my_decomp([[args:.*]]) -> !quantum.reg attributes {[[other_attributes:.*]] target_gate = "CRX"}
    print(circuit.mlir)

    # test that the decomposition is applied correctly
    # CHECK-NOT: CRX
    # CHECK-NOT: my_decomp

    # allocate for main register, subsequent allocates+releases for decomp registers
    # CHECK: allocate

    # first CRX: true branch
    # CHECK: CNOT
    # CHECK: allocate
    # CHECK: Hadamard
    # CHECK: PauliX
    # CHECK: Hadamard
    # CHECK: PauliX
    # CHECK: release

    # second CRX: false branch
    # CHECK: allocate
    # CHECK: Hadamard
    # CHECK: Measure
    # CHECK: cond
    # CHECK: CNOT
    # CHECK: release

    # release main register
    # CHECK: release

    print(circuit.mlir_opt)


test_decompose_work_wires_control_flow()


def test_decompose_work_wires_with_decompose_transform():
    """Test that work wires are correctly lowered and decomposed by the decompose transform."""

    qp.decomposition.enable_graph()

    @qp.register_resources({qp.X: 1, qp.Z: 1})
    def my_decomp(wire):
        with qp.allocate(1) as work_wire:
            qp.X(work_wire)
            qp.Z(wire)
            qp.X(work_wire)

    @qjit(capture=True)
    @partial(
        qp.transforms.decompose,
        gate_set={
            "X",
            "Z",
        },
        fixed_decomps={
            qp.Y: my_decomp,
        },
    )
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def my_circuit():
        qp.Y(0)
        return qp.probs()

    # CHECK-NOT: Y
    # CHECK-NOT: my_decomp

    # two allocates, one for main register and one for decomp register
    # CHECK: allocate
    # CHECK: allocate
    # CHECK: X
    # CHECK: Z
    # CHECK: X
    # CHECK: release
    # CHECK: release
    print(my_circuit.mlir_opt)

    qp.decomposition.disable_graph()


test_decompose_work_wires_with_decompose_transform()


def test_num_work_wires():
    """Test that num_work_wires can be passed and is correctly used in solving the graph."""

    qp.decomposition.enable_graph()

    @qp.register_resources(
        {qp.CNOT: 3, qp.H: 1, qp.X: 1, qp.ops.op_math.Conditional: 2},
        work_wires={"borrowed": 2, "garbage": 1},
    )
    def my_decomp(angle, wires, **_):
        def true_func():
            qp.CNOT(wires)

            with qp.allocate(2, state="any", restored=True) as w:
                qp.H(w[0])
                qp.H(w[0])
                qp.X(w[1])
                qp.X(w[1])

            return

        def false_func():
            with qp.allocate(1, state="any", restored=False) as w:
                qp.H(w)

            m = qp.measure(wires[0])

            qp.cond(m, qp.CNOT)(wires)

            return

        qp.cond(angle > 1.2, true_func, false_func)()

    @qp.qjit(capture=True)
    @partial(
        qp.transforms.decompose,
        gate_set={qp.CNOT, qp.H, qp.X, "Conditional", "MidMeasure"},
        fixed_decomps={qp.CRX: my_decomp},
        num_work_wires=3,
    )
    @qp.qnode(qp.device("lightning.qubit", wires=5))
    def circuit():
        qp.CRX(1.7, wires=[0, 1])
        qp.CRX(-7.2, wires=[0, 1])
        return qp.state()

    # CHECK-NOT: CRX
    # CHECK-NOT: my_decomp

    # CHECK: allocate
    # CHECK: allocate
    # CHECK: CNOT
    # CHECK: Hadamard
    # CHECK: Hadamard
    # CHECK: PauliX
    # CHECK: PauliX
    # CHECK: Hadamard
    # CHECK: Measure
    # CHECK: CNOT
    # CHECK: release
    # CHECK: release
    print(circuit.mlir_opt)

    qp.decomposition.disable_graph()


test_num_work_wires()


def test_default_decomps():
    """Test that default decompositions are correctly applied with qjit."""
    qp.decomposition.enable_graph()

    # Toffoli's decomposition to this gateset includes a wire allocation
    @qp.qjit(target="mlir", capture=True)
    @partial(
        qp.transforms.decompose,
        gate_set={qp.ops.ChangeOpBasis},
        num_work_wires=1,
    )
    @qp.qnode(qp.device("lightning.qubit", wires=4))
    def circuit():
        qp.Toffoli(wires=[0, 1, 2])
        return qp.state()

    # CHECK-NOT: toffoli_elbow
    # CHECK-NOT: Toffoli

    # two allocates/releases, for default register + work wires
    # CHECK: allocate
    # CHECK: allocate
    # CHECK: TemporaryAND
    # CHECK: release
    # CHECK: release
    print(circuit.mlir_opt)

    qp.decomposition.disable_graph()


test_default_decomps()


def test_graph_decomp_registered():
    """Test that the `graph_decomposition` pass is registered correctly."""

    @qjit(target="mlir", capture=True)
    # CHECK: transform.apply_registered_pass "graph-decomposition"
    @graph_decomposition(gate_set={qp.RX})
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def catalyst_circuit():
        return

    print(catalyst_circuit.mlir)

    my_transform = qp.transform(pass_name="graph-decomposition")

    @qjit(target="mlir", capture=True)
    # CHECK: transform.apply_registered_pass "graph-decomposition"
    @my_transform(gate_set=["RX"])
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def pennylane_circuit():
        return

    print(pennylane_circuit.mlir)


test_graph_decomp_registered()


def test_cpp_decomp_args():
    """Test that the `graph_decomposition` pass lowers arguments to mlir correctly."""

    def x_to_rx(wire):
        qp.RX(np.pi, wire)

    def y_to_ry(wire):
        qp.RY(np.pi, wire)

    def h_to_rx_ry(wire):
        qp.RX(np.pi / 2, wire)
        qp.RY(np.pi / 2, wire)

    @qjit(target="mlir")
    # CHECK: "graph-decomposition" with options = {
    # CHECK-DAG: "gate-set" = {Hadamard = 1.000000e+00 : f64, RX = 1.000000e+00 : f64, RY = 1.000000e+00 : f64}
    # CHECK-DAG: "fixed-decomps" = {PauliX = "x_to_rx", PauliY = "y_to_ry"}
    # CHECK-DAG: "alt-decomps" = {Hadamard = ["h_to_rx_ry"]}
    # CHECK-DAG: "bytecode-rules" = "{{.*}}decomposition_rules_{{.*}}.mlirbc"
    # CHECK: } to {{%.+}} : (!transform.op<"builtin.module">)
    @graph_decomposition(
        gate_set={qp.RX, qp.H, qp.RY},
        fixed_decomps={qp.X: x_to_rx, qp.Y: y_to_ry},
        alt_decomps={qp.H: [h_to_rx_ry]},
        _builtin_rule_path="/decomp_rules.mlirbc",
    )
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def circuit():
        return

    print(circuit.mlir)


test_cpp_decomp_args()


def test_cpp_decomp_empty_args():
    """
    Test that the `graph_decomposition` pass correctly handled arg lowering when no values are
    supplied.
    """

    @qjit(target="mlir", capture=True)
    # CHECK: transform.apply_registered_pass "graph-decomposition"
    # CHECK-NOT: fixed-decomps
    # CHECK-NOT: alt-decomps
    # CHECK: "bytecode-rules" = "{{.*}}/decomposition_rules{{.*}}.mlirbc"
    @graph_decomposition(gate_set={qp.RX})
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def circuit():
        return

    print(circuit.mlir)

    @qjit(target="mlir", capture=True)
    # CHECK: transform.apply_registered_pass "graph-decomposition"
    # CHECK-NOT: fixed-decomps
    # CHECK-NOT: alt-decomps
    # CHECK: "bytecode-rules" = "{{.*}}/decomposition_rules{{.*}}.mlirbc"
    @graph_decomposition(gate_set={qp.RX}, fixed_decomps={}, alt_decomps={})
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def circuit2():
        return

    print(circuit2.mlir)


test_cpp_decomp_empty_args()


def test_cpp_decomp_string_op_names():
    """Test that cpp decomp args work with string op names."""

    def y_to_xz(wires):
        qp.RX(np.pi, wires)
        qp.RZ(np.pi, wires)

    @qjit(target="mlir", capture=True)
    # CHECK: transform.apply_registered_pass "graph-decomposition" with options = {
    # CHECK-DAG: "fixed-decomps" = {PauliX = "{{.*}}", PauliZ = "{{.*}}"}
    # CHECK-DAG: "alt-decomps" = {PauliY = ["{{.*}}", "y_to_xz"]}
    # CHECK: } to {{%.+}} : (!transform.op<"builtin.module">)
    @graph_decomposition(
        gate_set={"RX", "RY", "RZ"},
        fixed_decomps={
            "X": lambda wires: qp.RX(np.pi, wires),
            "PauliZ": lambda wires: qp.RZ(np.pi, wires),
        },
        alt_decomps={
            "PauliY": [
                lambda wires: qp.RY(np.pi, wires),
                y_to_xz,
            ]
        },
    )
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def circuit():
        return

    print(circuit.mlir)


test_cpp_decomp_string_op_names()


def test_cpp_decomp_builtin_rules():
    """Test that cpp decomp applies builtin rules."""

    @qjit(target="mlir", capture=True)
    @graph_decomposition(
        gate_set={qp.RX, qp.RY, qp.RZ, qp.GlobalPhase},
    )
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def circuit():
        # CHECK-NOT: PauliX
        # CHECK-NOT: PauliY
        # CHECK-NOT: PauliZ
        # CHECK-DAG: RX
        # CHECK-DAG: RY
        # CHECK-DAG: RZ
        qp.X(0)
        qp.Y(1)
        qp.Z(0)
        return qp.probs()

    print(circuit.mlir_opt)


test_cpp_decomp_builtin_rules()


def test_cpp_decomp_user_rules():
    """Test that cpp decomp applies user rules."""

    @decomposition_rule(is_qreg=True, op_type="PauliY")
    def y_to_rx(wire):
        qp.RX(np.pi, wire)

    @decomposition_rule(is_qreg=True, op_type="PauliZ")
    def z_to_rx(wire):
        qp.RX(np.pi, wire)

    @qp.qjit(target="mlir", capture=True)
    @graph_decomposition(
        gate_set={qp.RX}, fixed_decomps={qp.Y: y_to_rx}, alt_decomps={qp.Z: [z_to_rx]}
    )
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit():
        y_to_rx(jax.core.ShapedArray((1,), int))
        z_to_rx(jax.core.ShapedArray((1,), int))
        # CHECK-NOT: PauliY
        # CHECK-NOT: PauliZ
        # CHECK: RX
        # CHECK: RX
        # CHECK: return
        qp.Y(0)
        qp.Z(0)
        return qp.probs()

    print(circuit.mlir_opt)


test_cpp_decomp_user_rules()


def test_cpp_decomp_user_rule_cleanup():
    """Test that user rules do not pollute the IR after the quantum compilation stage."""

    @decomposition_rule(is_qreg=True, op_type="PauliX")
    def x_to_h(wire):
        return qp.H(wire)

    @qjit(capture=True)
    @graph_decomposition(gate_set={qp.H}, fixed_decomps={qp.X: x_to_h})
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit():
        # CHECK-NOT: PauliX
        # CHECK-NOT: x_to_h
        x_to_h(jax.core.ShapedArray((1,), int))
        qp.X(0)

    print(circuit.mlir_opt)


test_cpp_decomp_user_rule_cleanup()
