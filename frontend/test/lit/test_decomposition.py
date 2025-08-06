# Copyright 2022-2023 Xanadu Quantum Technologies Inc.
import os
import pathlib
import platform
from copy import deepcopy

import jax
import pennylane as qml
from pennylane.devices.capabilities import OperatorProperties

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

        def __init__(self, shots=None, wires=None):
            super().__init__(wires=wires, shots=shots)
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

    @decomposition_rule
    def Hadamard0(wire):
        qml.Hadamard(wire)

    qml.capture.enable()

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: module @circuit
    def circuit(c: int):
        # CHECK: func.func public @circuit([[ARG0:%.+]]
        # CHECK: [[QREG:%.+]] = quantum.alloc
        Hadamard0(int)
        return qml.probs()

    # CHECK: func.func private @Hadamard0([[QBIT:%.+]]: !quantum.bit) -> !quantum.bit
    # CHECK-NEXT: [[QUBIT_OUT:%.+]] = quantum.custom "Hadamard"() [[QBIT]] : !quantum.bit
    # CHECK-NEXT: return [[QUBIT_OUT]] : !quantum.bit

    print(circuit.mlir)

    qml.capture.disable()


test_decomposition_rule_wire_param()


def test_decomposition_rule_gate_param_param():
    """Test decomposition rule with passing a regular parameter"""

    @decomposition_rule(num_params=1)
    def RX_on_wire_0(param, w0):
        qml.RX(param, wires=w0)

    qml.capture.enable()

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: module @circuit_2
    def circuit_2(param: float):
        RX_on_wire_0(float, int)
        return qml.probs()

    # CHECK: func.func private @RX_on_wire_0([[PARAM_TENSOR:%.+]]: tensor<f64>, [[QUBIT:%.+]]: !quantum.bit) -> !quantum.bit
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

    @decomposition_rule(num_params=1)
    def all_wires_rx(param, w0, w1, w2):
        qml.RX(param, wires=w0)
        qml.RX(param, wires=w1)
        qml.RX(param, wires=w2)

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    # CHECK: module @circuit_3
    def circuit_3(_: float):
        # CHECK-DAG: [[QREG:%.+]] = quantum.alloc
        # CHECK-NEXT: [[QUBIT:%.+]] = quantum.extract [[QREG]][ 0] : !quantum.reg -> !quantum.bit
        # CHECK-NEXT: [[QUBIT_1:%.+]] = quantum.custom "Hadamard"() [[QUBIT]] : !quantum.bit
        # CHECK-NEXT: [[QREG_1:%.+]] = quantum.insert [[QREG]][ 0], [[QUBIT_1]] : !quantum.reg, !quantum.bit
        # CHECK-NEXT: quantum.compbasis qreg [[QREG_1]] : !quantum.obs
        identity()
        all_wires_rx(int, float, float, float)
        qml.Hadamard(0)
        return qml.probs()

    # CHECK: func.func private @identity
    # CHECK: func.func private @all_wires_rx

    print(circuit_3.mlir)
    qml.capture.disable()


test_multiple_decomposition_rules()
