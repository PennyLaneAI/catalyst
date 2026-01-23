# Copyright 2022-2025 Xanadu Quantum Technologies Inc.
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
# pylint: disable=too-many-lines


# Helper to skip tests that fail due to PauliRot type annotation issue
# TODO: Remove this once PennyLane fixes the PauliRot decomposition type annotations
def skip_if_pauli_rot_issue(test_func):
    """Wrapper to skip tests that fail due to PauliRot type annotation issues."""

    def wrapper():
        try:
            test_func()
        except (ValueError, IndexError) as e:
            error_msg = str(e)
            if (
                "Unsupported type annotation None for parameter pauli_word" in error_msg
                or "Unsupported type annotation <class 'str'> for parameter pauli_word" in error_msg
                or "index is out of bounds for axis" in error_msg
            ):
                print(f"# SKIPPED {test_func.__name__}: PauliRot type annotation issue")
            else:
                raise

    return wrapper


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


def test_decompose_gateset_with_graph():
    """Test the decompose transform to a target gate set with the graph decomposition."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.qjit(target="mlir")
    @partial(qml.transforms.decompose, gate_set={"RX"})
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: public @simple_circuit_9() -> tensor<f64> attributes {decompose_gatesets
    def simple_circuit_9():
        return qml.expval(qml.Z(0))

    print(simple_circuit_9.mlir)

    @qml.qjit(target="mlir")
    @partial(qml.transforms.decompose, gate_set={"RX", "RZ"})
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    # CHECK: public @circuit_9() -> tensor<f64> attributes {decompose_gatesets
    def circuit_9():
        return qml.expval(qml.Z(0))

    print(circuit_9.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


test_decompose_gateset_with_graph()


def test_decompose_gateset_without_graph():
    """Test the decompose transform to a target gate set without the graph decomposition."""

    qml.capture.enable()

    @qml.qjit(target="mlir")
    @partial(qml.transforms.decompose, gate_set={"RX", "RZ"})
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: func.func public @circuit_8() -> tensor<f64> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode}
    def circuit_8():
        return qml.expval(qml.Z(0))

    print(circuit_8.mlir)
    qml.capture.disable()


test_decompose_gateset_without_graph()


def test_decompose_gateset_operator_with_graph():
    """Test the decompose transform to a target gate set with the graph decomposition."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.qjit(target="mlir")
    @partial(qml.transforms.decompose, gate_set={qml.RX})
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: public @simple_circuit_10() -> tensor<f64> attributes {decompose_gatesets
    def simple_circuit_10():
        return qml.expval(qml.Z(0))

    print(simple_circuit_10.mlir)

    @qml.qjit(target="mlir")
    @partial(
        qml.transforms.decompose, gate_set={qml.RX, qml.RZ, "PauliZ", qml.PauliX, qml.Hadamard}
    )
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: public @circuit_10() -> tensor<f64> attributes {decompose_gatesets
    def circuit_10():
        return qml.expval(qml.Z(0))

    print(circuit_10.mlir)

    @qml.qjit(target="mlir")
    @partial(
        qml.transforms.decompose, gate_set={qml.RX, qml.RZ, qml.PauliZ, qml.PauliX, qml.Hadamard}
    )
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    # CHECK: public @circuit_11() -> tensor<f64> attributes {decompose_gatesets
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
    # CHECK: public @simple_circuit_12() -> tensor<f64> attributes {decompose_gatesets
    def simple_circuit_12():
        return qml.expval(qml.Z(0))

    print(simple_circuit_12.mlir)

    @qml.qjit(target="mlir")
    @partial(qml.transforms.decompose, gate_set={qml.ftqc.RotXZX})
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    # CHECK: public @circuit_12() -> tensor<f64> attributes {decompose_gatesets
    def circuit_12():
        return qml.expval(qml.Z(0))

    print(circuit_12.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


test_decompose_gateset_with_rotxzx()


def test_decomposition_rule_name():
    """Test the name of the decomposition rule is not updated with circuit instantiation."""

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
    # CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    # CHECK: public @circuit_13() -> tensor<f64> attributes {decompose_gatesets
    def circuit_13():
        _ry_to_rz_rx(float, int)
        _rot_to_rz_ry_rz(float, float, float, int)
        _u2_phaseshift_rot_decomposition(float, float, int)
        _xzx_decompose(float, float, float, int)
        return qml.expval(qml.Z(0))

    # CHECK: func.func public @_ry_to_rz_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<i64>) -> !quantum.reg
    # CHECK: func.func public @_rot_to_rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<i64>) -> !quantum.reg
    # CHECK: func.func public @_u2_phaseshift_rot_decomposition(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<i64>) -> !quantum.reg
    # CHECK: func.func public @_xzx_decompose(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<i64>) -> !quantum.reg
    print(circuit_13.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


test_decomposition_rule_name()


def test_decomposition_rule_name_update():
    """Test the name of the decomposition rule is updated in the MLIR output."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.register_resources({qml.RZ: 2, qml.RX: 1})
    def rz_rx(phi, wires: WiresLike, **__):
        """Decomposition of RY gate using RZ and RX gates."""
        qml.RZ(-np.pi / 2, wires=wires)
        qml.RX(phi, wires=wires)
        qml.RZ(np.pi / 2, wires=wires)

    @qml.register_resources({qml.RZ: 2, qml.RY: 1})
    def rz_ry_rz(phi, theta, omega, wires: WiresLike, **__):
        """Decomposition of Rot gate using RZ and RY gates."""
        qml.RZ(phi, wires=wires)
        qml.RY(theta, wires=wires)
        qml.RZ(omega, wires=wires)

    @qml.register_resources({qml.RY: 1, qml.GlobalPhase: 1})
    def ry_gp(wires: WiresLike, **__):
        """Decomposition of PauliY gate using RY and GlobalPhase gates."""
        qml.RY(np.pi, wires=wires)
        qml.GlobalPhase(-np.pi / 2, wires=wires)

    @qml.qjit(target="mlir")
    @partial(
        qml.transforms.decompose,
        gate_set={"RX", "RZ", "GlobalPhase"},
        fixed_decomps={
            qml.RY: rz_rx,
            qml.Rot: rz_ry_rz,
            qml.PauliY: ry_gp,
        },
    )
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    # CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    # CHECK: public @circuit_14() -> tensor<f64> attributes {decompose_gatesets
    def circuit_14():
        qml.RY(0.5, wires=0)
        qml.Rot(0.1, 0.2, 0.3, wires=1)
        qml.PauliY(wires=2)
        return qml.expval(qml.Z(0))

    # CHECK-DAG: func.func public @rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) -> !quantum.reg
    # CHECK-DAG: func.func public @rz_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg
    # CHECK-DAG: func.func public @ry_gp(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg
    print(circuit_14.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


test_decomposition_rule_name_update()


def test_decomposition_rule_name_update_multi_qubits():
    """Test the name of the decomposition rule with multi-qubit gates."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.qjit(target="mlir")
    @partial(
        qml.transforms.decompose,
        gate_set={"RY", "RX", "CNOT", "Hadamard", "GlobalPhase"},
    )
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    # CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    # CHECK: public @circuit_15() -> tensor<f64> attributes {decompose_gatesets
    def circuit_15():
        qml.SingleExcitation(0.5, wires=[0, 1])
        qml.SingleExcitationPlus(0.5, wires=[0, 1])
        qml.SingleExcitationMinus(0.5, wires=[0, 1])
        qml.DoubleExcitation(0.5, wires=[0, 1, 2, 3])
        return qml.expval(qml.Z(0))

    # CHECK-DAG: func.func public @_cry(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "CRY"}
    # CHECK-DAG: func.func public @_s_phaseshift(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "S"}
    # CHECK-DAG: func.func public @_phaseshift_to_rz_gp(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "PhaseShift"}
    # CHECK-DAG: func.func public @_rz_to_ry_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RZ"}
    # CHECK-DAG: func.func public @_rot_to_rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    # CHECK-DAG: func.func public @_doublexcit(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<4xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 4 : i64, target_gate = "DoubleExcitation"}
    # CHECK-DAG: func.func public @_single_excitation_decomp(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "SingleExcitation"}
    print(circuit_15.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


skip_if_pauli_rot_issue(test_decomposition_rule_name_update_multi_qubits)()


def test_decomposition_rule_name_adjoint():
    """Test decomposition rule with qml.adjoint."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.qjit(target="mlir")
    @partial(
        qml.transforms.decompose,
        gate_set={"RY", "RX", "CZ", "GlobalPhase", "Adjoint(SingleExcitation)"},
    )
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    # CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    def circuit_16(x: float):
        # CHECK-DAG: %1 = quantum.adjoint(%0) : !quantum.reg
        # CHECK-DAG: %2 = quantum.adjoint(%1) : !quantum.reg
        # CHECK-DAG: %3 = quantum.adjoint(%2) : !quantum.reg
        # CHECK-DAG: %4 = quantum.adjoint(%3) : !quantum.reg
        qml.adjoint(qml.CNOT)(wires=[0, 1])
        qml.adjoint(qml.Hadamard)(wires=2)
        qml.adjoint(qml.RZ)(0.5, wires=3)
        qml.adjoint(qml.SingleExcitation)(0.1, wires=[0, 1])
        qml.adjoint(qml.SingleExcitation(x, wires=[0, 1]))
        return qml.expval(qml.Z(0))

    # CHECK-DAG: func.func public @_single_excitation_decomp(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "SingleExcitation"}
    # CHECK-DAG: func.func public @_hadamard_to_rz_ry(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Hadamard"}
    # CHECK-DAG: func.func public @_rz_to_ry_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RZ"}
    # CHECK-DAG: func.func public @_rot_to_rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    # CHECK-DAG: func.func public @_cnot_to_cz_h(%arg0: !quantum.reg, %arg1: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "CNOT"}
    print(circuit_16.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


skip_if_pauli_rot_issue(test_decomposition_rule_name_adjoint)()


# TODO: Reenable this once the underlying non-determinism issue is resolved'
def test_decomposition_rule_name_ctrl():
    """Test decomposition rule with qml.ctrl."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.qjit(target="mlir")
    @partial(
        qml.transforms.decompose,
        gate_set={"RX", "RZ", "H", "CZ"},
    )
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    # SKIP-CHECK-DAG: %0 = transform.apply_registered_pass "decompose-lowering"
    # SKIP-CHECK{LITERAL}: func.func public @circuit_17() -> tensor<f64> attributes {decompose_gatesets
    def circuit_17():
        # SKIP-CHECK: %out_qubits:2 = quantum.custom "CRY"(%cst) %1, %2 : !quantum.bit, !quantum.bit
        # SKIP-CHECK-NEXT: %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits#0, %out_qubits#1 : !quantum.bit, !quantum.bit
        qml.ctrl(qml.RY, control=0)(0.5, 1)
        qml.ctrl(qml.PauliX, control=0)(1)
        return qml.expval(qml.Z(0))

    # SKIP-CHECK-DAG: func.func public @_cnot_to_cz_h(%arg0: !quantum.reg, %arg1: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "CNOT"}
    # SKIP-CHECK-DAG: func.func public @_cry(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "CRY"}
    # SKIP-CHECK-DAG: func.func public @_ry_to_rz_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RY"}
    # SKIP-CHECK-DAG: func.func public @_rot_to_rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    # print(circuit_17.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


skip_if_pauli_rot_issue(test_decomposition_rule_name_ctrl)()


# TODO: Reenable this once the underlying non-determinism issue is resolved
def test_qft_decomposition():
    """Test the decomposition of the QFT"""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.qjit(autograph=True, target="mlir")
    @partial(
        qml.transforms.decompose,
        gate_set={"RX", "RY", "CNOT", "GlobalPhase"},
    )
    @qml.qnode(qml.device("lightning.qubit", wires=4))
    # SKIP-CHECK: %0 = transform.apply_registered_pass "decompose-lowering"
    # SKIP-CHECK: func.func public @circuit_18(%arg0: tensor<3xf64>) -> tensor<f64> attributes {decompose_gatesets
    def circuit_18():
        # %6 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %0) -> (!quantum.reg) {
        # %23 = scf.for %arg3 = %c0 to %22 step %c1 iter_args(%arg4 = %21) -> (!quantum.reg) {
        # %7 = scf.for %arg1 = %c0 to %c2 step %c1 iter_args(%arg2 = %6) -> (!quantum.reg) {
        qml.QFT(wires=[0, 1, 2, 3])
        return qml.expval(qml.Z(0))

    # SKIP-CHECK-DAG: func.func public @ag___cphase_to_rz_cnot(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "ControlledPhaseShift"}
    # SKIP-CHECK-DAG: func.func public @ag___rz_to_ry_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RZ"}
    # SKIP-CHECK-DAG: func.func public @ag___rot_to_rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    # SKIP-CHECK-DAG: func.func public @ag___swap_to_cnot(%arg0: !quantum.reg, %arg1: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "SWAP"}
    # SKIP-CHECK-DAG: func.func public @ag___hadamard_to_rz_ry(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Hadamard"}
    # print(circuit_18.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


skip_if_pauli_rot_issue(test_qft_decomposition)()


def test_decompose_lowering_with_other_passes():
    """Test the decompose lowering pass with other passes in a pass pipeline."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.qjit(target="mlir")
    @qml.transforms.merge_rotations
    @qml.transforms.cancel_inverses
    @partial(
        qml.transforms.decompose,
        gate_set={"RZ", "RY", "CNOT", "GlobalPhase"},
    )
    @qml.qnode(qml.device("lightning.qubit", wires=1))
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
        qml.PauliX(0)
        qml.PauliX(0)
        qml.RX(0.1, wires=0)
        qml.RX(-0.1, wires=0)
        return qml.expval(qml.PauliX(0))

    # CHECK-DAG: func.func public @_paulix_to_rx(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "PauliX"}
    # CHECK-DAG: func.func public @_rx_to_rz_ry(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RX"}
    print(circuit_19.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


skip_if_pauli_rot_issue(test_decompose_lowering_with_other_passes)()


def test_decompose_lowering_multirz():
    """Test the decompose lowering pass with MultiRZ in the gate set."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.qjit(target="mlir")
    @partial(
        qml.transforms.decompose,
        gate_set={"CNOT", "RZ"},
    )
    @qml.qnode(qml.device("lightning.qubit", wires=3))
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
        qml.MultiRZ(x, wires=[0])
        qml.MultiRZ(x, wires=[0, 1])
        qml.MultiRZ(x, wires=[1, 0, 2])
        return qml.expval(qml.PauliX(0))

    # CHECK-DAG: func.func public @_multi_rz_decomposition_wires_1(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "MultiRZ"}
    # CHECK-DAG: func.func public @_multi_rz_decomposition_wires_2(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<2xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 2 : i64, target_gate = "MultiRZ"}
    # CHECK-DAG: func.func public @_multi_rz_decomposition_wires_3(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<3xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 3 : i64, target_gate = "MultiRZ"}
    # CHECK-DAG: %0 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %arg0) -> (!quantum.reg)
    # CHECK-DAG:   %5 = scf.for %arg3 = %c1 to %c3 step %c1 iter_args(%arg4 = %4) -> (!quantum.reg)
    print(circuit_20.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


test_decompose_lowering_multirz()


def test_decompose_lowering_with_ordered_passes():
    """Test the decompose lowering pass with other passes in a specific order in a pass pipeline."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.qjit(target="mlir")
    @partial(
        qml.transforms.decompose,
        gate_set={"RZ", "RY", "CNOT", "GlobalPhase"},
    )
    @qml.transforms.merge_rotations
    @qml.transforms.cancel_inverses
    @qml.qnode(qml.device("lightning.qubit", wires=1))
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
        qml.PauliX(0)
        qml.PauliX(0)
        qml.RX(x, wires=0)
        qml.RX(-x, wires=0)
        return qml.expval(qml.PauliX(0))

    # CHECK-DAG: func.func public @_paulix_to_rx(%arg0: !quantum.reg, %arg1: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "PauliX"}
    # CHECK-DAG: func.func public @_rx_to_rz_ry(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RX"}
    # CHECK-DAG: func.func public @_rot_to_rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    print(circuit_21.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


skip_if_pauli_rot_issue(test_decompose_lowering_with_ordered_passes)()


def test_decompose_lowering_with_gphase():
    """Test the decompose lowering pass with GlobalPhase."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.qjit(target="mlir")
    @partial(
        qml.transforms.decompose,
        gate_set={"RX", "RY", "GlobalPhase"},
    )
    @qml.qnode(qml.device("lightning.qubit", wires=3))
    # CHECK:  %0 = transform.apply_registered_pass "decompose-lowering"
    def circuit_22():
        # CHECK:  quantum.gphase(%cst_0) :
        # CHECK-NEXT:  [[EXTRACTED:%.+]] = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
        # CHECK-NEXT:  [[OUT_QUBITS:%.+]] = quantum.custom "PhaseShift"(%cst) [[EXTRACTED]] : !quantum.bit
        # CHECK-NEXT:  {{%.+}} = quantum.custom "PhaseShift"(%cst) [[OUT_QUBITS]] : !quantum.bit
        qml.GlobalPhase(0.5)
        qml.ctrl(qml.GlobalPhase, control=0)(0.3)
        qml.ctrl(qml.GlobalPhase, control=0)(phi=0.3, wires=[1, 2])
        return qml.expval(qml.PauliX(0))

    # CHECK-DAG: func.func public @_phaseshift_to_rz_gp(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "PhaseShift"}
    # CHECK-DAG: func.func public @_rz_to_ry_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RZ"}
    print(circuit_22.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


skip_if_pauli_rot_issue(test_decompose_lowering_with_gphase)()


def test_decompose_lowering_alt_decomps():
    """Test the decompose lowering pass with alternative decompositions."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.register_resources({qml.RY: 1})
    def custom_rot_cheap(params, wires: WiresLike):
        qml.RY(params[1], wires=wires)

    @qml.qjit(target="mlir")
    @partial(
        qml.transforms.decompose,
        gate_set={"RY", "RZ"},
        alt_decomps={qml.Rot: [custom_rot_cheap]},
    )
    @qml.qnode(qml.device("lightning.qubit", wires=3), shots=1000)
    def circuit_23(x: float, y: float):
        qml.Rot(x, y, x + y, wires=1)
        return qml.expval(qml.PauliZ(0))

    # CHECK-DAG: func.func public @custom_rot_cheap(%arg0: !quantum.reg, %arg1: tensor<3xf64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    print(circuit_23.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


test_decompose_lowering_alt_decomps()


def test_decompose_lowering_with_tensorlike():
    """Test the decompose lowering pass with fixed decompositions
    using TensorLike parameters."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.register_resources({qml.RZ: 2, qml.RY: 1})
    def custom_rot(params: TensorLike, wires: WiresLike):
        qml.RZ(params[0], wires=wires)
        qml.RY(params[1], wires=wires)
        qml.RZ(params[2], wires=wires)

    @qml.register_resources({qml.RZ: 1, qml.CNOT: 4})
    def custom_multirz(params: TensorLike, wires: WiresLike):
        qml.CNOT(wires=(wires[2], wires[1]))
        qml.CNOT(wires=(wires[1], wires[0]))
        qml.RZ(params[0], wires=wires[0])
        qml.CNOT(wires=(wires[1], wires[0]))
        qml.CNOT(wires=(wires[2], wires[1]))

    @qml.qjit(target="mlir")
    @partial(
        qml.transforms.decompose,
        gate_set={"RY", "RX", qml.CNOT},
        fixed_decomps={qml.Rot: custom_rot, qml.MultiRZ: custom_multirz},
    )
    @qml.qnode(qml.device("lightning.qubit", wires=3), shots=1000)
    def circuit_24(x: float, y: float):
        qml.Rot(x, y, x + y, wires=1)
        qml.MultiRZ(x + y, wires=[0, 1, 2])
        return qml.expval(qml.PauliZ(0))

    # CHECK-DAG: func.func public @custom_multirz_wires_3(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<3xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 3 : i64, target_gate = "MultiRZ"}
    # CHECK-DAG: func.func public @_rz_to_ry_rx(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "RZ"}
    # CHECK-DAG: func.func public @custom_rot(%arg0: !quantum.reg, %arg1: tensor<3xf64>, %arg2: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    print(circuit_24.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


skip_if_pauli_rot_issue(test_decompose_lowering_with_tensorlike)()


def test_decompose_lowering_fallback():
    """Test the decompose lowering pass when the graph is failed."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qml.qjit(target="mlir")
    @partial(qml.transforms.decompose, gate_set={qml.RX, qml.RZ})
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    # CHECK: func.func public @circuit_25() -> tensor<4xcomplex<f64>> attributes {diff_method = "adjoint", llvm.linkage = #llvm.linkage<internal>, qnode}
    def circuit_25():
        # CHECK: [[OUT_QUBIT:%.+]] = quantum.custom "RZ"(%cst) {{%.+}} : !quantum.bit
        # CHECK-NEXT: [[OUT_QUBIT_0:%.+]] = quantum.custom "RX"(%cst) [[OUT_QUBIT]] : !quantum.bit
        # CHECK-NEXT: [[OUT_QUBIT_1:%.+]] = quantum.custom "RZ"(%cst) [[OUT_QUBIT_0]] : !quantum.bit
        qml.Hadamard(0)
        return qml.state()

    print(circuit_25.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


test_decompose_lowering_fallback()


def test_decompose_lowering_params_ordering():
    """Test the order of params and wires in the captured decomposition rule."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    @qjit(target="mlir")
    @partial(qml.transforms.decompose, gate_set=[qml.RX, qml.RY, qml.RZ])
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    # CHECK: func.func public @circuit_26(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>)
    def circuit_26(x: float, y: float, z: float):
        qml.Rot(x, y, z, wires=0)
        return qml.expval(qml.PauliZ(0))

    # CHECK: func.func public @_rot_to_rz_ry_rz(%arg0: !quantum.reg, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>, %arg4: tensor<1xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 1 : i64, target_gate = "Rot"}
    # CHECK:  [[EXTRACTED_0:%.+]] = tensor.extract %arg1[] : tensor<f64>
    # CHECK-NEXT:  [[OUT_QUBITS:%.+]] = quantum.custom "RZ"([[EXTRACTED_0]]) {{%.+}} : !quantum.bit
    # CHECK:  [[EXTRACTED_3:%.+]] = tensor.extract %arg2[] : tensor<f64>
    # CHECK-NEXT:  [[OUT_QUBITS_4:%.+]] = quantum.custom "RY"([[EXTRACTED_3]]) {{%.+}} : !quantum.bit
    # CHECK:  [[EXTRACTED_7:%.+]] = tensor.extract %arg3[] : tensor<f64>
    # CHECK-NEXT:  [[OUT_QUBITS_8:%.+]] = quantum.custom "RZ"([[EXTRACTED_7]]) {{%.+}} : !quantum.bit
    # CHECK:  return {{%.+}} : !quantum.reg
    print(circuit_26.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


test_decompose_lowering_params_ordering()


def test_decomposition_rule_with_allocation():
    """Test decomposition rule with dynamic qubit allocation"""

    qml.capture.enable()

    @decomposition_rule(is_qreg=True)
    def Hadamard0_with_alloc(wire: WiresLike):
        with qml.allocate(1) as q:
            qml.X(q[0])
            qml.CNOT(wires=[q[0], wire])

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: module @circuit_27
    def circuit_27():
        Hadamard0_with_alloc(int)
        return qml.probs()

    # CHECK: func.func public @Hadamard0_with_alloc(%arg0: !quantum.reg, %arg1: tensor<i64>) -> !quantum.reg
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

    qml.capture.disable()


test_decomposition_rule_with_allocation()


def test_decompose_autograph_multi_blocks():
    """Test the decompose lowering pass with autograph in the program and rule."""

    qml.capture.enable()
    qml.decomposition.enable_graph()

    def _multi_rz_decomposition_resources(num_wires):
        """Resources required for MultiRZ decomposition."""
        return {qml.RZ: 1, qml.CNOT: 2 * (num_wires - 1)}

    @qml.register_resources(_multi_rz_decomposition_resources)
    @qml.capture.run_autograph
    def _multi_rz_decomposition(theta: TensorLike, wires: WiresLike, **__):
        """Decomposition of MultiRZ using CNOTs and RZs."""
        for i in range(len(wires) - 1):
            qml.CNOT(wires=(wires[i], wires[i + 1]))
        qml.RZ(theta, wires=wires[0])
        for i in range(len(wires) - 1, 0, -1):
            qml.CNOT(wires=(wires[i], wires[i - 1]))

    @qml.qjit(target="mlir")
    @partial(
        qml.transforms.decompose,
        gate_set={"RZ", "CNOT"},
        fixed_decomps={qml.MultiRZ: _multi_rz_decomposition},
    )
    @qml.qnode(qml.device("lightning.qubit", wires=5))
    def circuit_29(n: int):

        # CHECK: {{%.+}} = scf.for %arg1 = {{%.+}} to %1 step {{%.+}} iter_args(%arg2 = %0) -> (!quantum.reg) {
        for _ in range(n):
            qml.MultiRZ(0.5, wires=[0, 1, 2, 3, 4])

        return qml.expval(qml.Z(0))

    # CHECK: func.func public @ag___multi_rz_decomposition(%arg0: !quantum.reg, %arg1: tensor<1xf64>, %arg2: tensor<5xi64>) -> !quantum.reg attributes {llvm.linkage = #llvm.linkage<internal>, num_wires = 5 : i64, target_gate = "MultiRZ"}
    # CHECK: {{%.+}} = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %arg0) -> (!quantum.reg)
    # CHECK: {{%.+}} = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %4) -> (!quantum.reg)
    print(circuit_29.mlir)

    qml.decomposition.disable_graph()
    qml.capture.disable()


test_decompose_autograph_multi_blocks()