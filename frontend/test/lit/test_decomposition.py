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

# RUN: %PYTHON %s | FileCheck %s

import jax
import pennylane as qml

from catalyst import cond, for_loop, measure, qjit, while_loop

# This is used just for internal testing
from catalyst.pennylane_extensions import qfunc


def get_custom_device_without(num_wires, discards):
    lightning = qml.device("lightning.qubit", wires=3)
    copy = lightning.operations.copy()
    for discard in discards:
        copy.discard(discard)

    class CustomDevice(qml.QubitDevice):
        name = "Device without some operations"
        short_name = "dummy.device"
        pennylane_requires = "0.1.0"
        version = "0.0.1"
        author = "CV quantum"

        operations = copy
        observables = discard

        def __init__(self, shots=None, wires=None, backend=None):
            self.backend = backend if backend else "default"
            super().__init__(wires=wires, shots=shots)

        def apply(self, operations, **kwargs):
            pass

    return CustomDevice(wires=num_wires)


def test_decompose_multicontrolledx():
    dev = get_custom_device_without(5, {"MultiControlledX"})

    @qjit(target="mlir")
    @qfunc(5, device=dev)
    # CHECK-LABEL: public @jit_decompose_multicontrolled_x1
    def decompose_multicontrolled_x1(theta: float):
        qml.RX(theta, wires=[0])
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state0:%.+]]:3 = "quantum.custom"([[q2:%.+]], [[q4:%.+]], [[q3:%.+]]) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state1:%.+]]:3 = "quantum.custom"([[q0:%.+]], [[q1:%.+]], [[state0]]#1) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state2:%.+]]:3 = "quantum.custom"([[state0]]#0, [[state1]]#2, [[state0]]#2) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state3:%.+]]:3 = "quantum.custom"([[state1]]#0, [[state1]]#1, [[state2]]#1) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        qml.MultiControlledX(wires=[0, 1, 2, 3], work_wires=[4])
        return qml.state()

    print(decompose_multicontrolled_x1.mlir)


test_decompose_multicontrolledx()


def test_decompose_multicontrolledx_in_conditional():
    dev = get_custom_device_without(5, {"MultiControlledX"})

    @qjit(target="mlir")
    @qfunc(5, device=dev)
    # CHECK-LABEL: @jit_decompose_multicontrolled_x2
    def decompose_multicontrolled_x2(theta: float, n: int):
        qml.RX(theta, wires=[0])

        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state0:%.+]]:3 = "quantum.custom"([[q2:%.+]], [[q4:%.+]], [[q3:%.+]]) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state1:%.+]]:3 = "quantum.custom"([[q0:%.+]], [[q1:%.+]], [[state0]]#1) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state2:%.+]]:3 = "quantum.custom"([[state0]]#0, [[state1]]#2, [[state0]]#2) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state3:%.+]]:3 = "quantum.custom"([[state1]]#0, [[state1]]#1, [[state2]]#1) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        @cond(n > 1)
        def cond_fn():
            qml.MultiControlledX(wires=[0, 1, 2, 3], work_wires=[4])
            return

        cond_fn()
        return qml.state()

    print(decompose_multicontrolled_x2.mlir)


test_decompose_multicontrolledx_in_conditional()


def test_decompose_multicontrolledx_in_while_loop():
    dev = get_custom_device_without(5, {"MultiControlledX"})

    @qjit(target="mlir")
    @qfunc(5, device=dev)
    # CHECK-LABEL: @jit_decompose_multicontrolled_x3
    def decompose_multicontrolled_x3(theta: float, n: int):
        qml.RX(theta, wires=[0])

        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state0:%[0-9]+]]{{:3}} = "quantum.custom"([[q2:%[0-9]+]], [[q4:%[0-9]+]], [[q3:%[0-9]+]]) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state1:%[0-9]+]]{{:3}} = "quantum.custom"([[q0:%[0-9]+]], [[q1:%[0-9]+]], [[state0]]{{#1}}) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state2:%[0-9]+]]{{:3}} = "quantum.custom"([[state0]]{{#0}}, [[state1]]{{#2}}, [[state0]]{{#2}}) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state3:%[0-9]+]]{{:3}} = "quantum.custom"([[state1]]{{#0}}, [[state1]]{{#1}}, [[state2]]{{#1}}) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        @while_loop(lambda v: v[0] < 10)
        def loop(v):
            qml.MultiControlledX(wires=[0, 1, 2, 3], work_wires=[4])
            return v[0] + 1, v[1]

        loop((0, n))
        return qml.state()

    print(decompose_multicontrolled_x3.mlir)


test_decompose_multicontrolledx_in_while_loop()


def test_decompose_multicontrolledx_in_for_loop():
    dev = get_custom_device_without(5, {"MultiControlledX"})

    @qjit(target="mlir")
    @qfunc(5, device=dev)
    # CHECK-LABEL: @jit_decompose_multicontrolled_x4
    def decompose_multicontrolled_x4(theta: float, n: int):
        qml.RX(theta, wires=[0])

        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state0:%[0-9]+]]{{:3}} = "quantum.custom"([[q2:%[0-9]+]], [[q4:%[0-9]+]], [[q3:%[0-9]+]]) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state1:%[0-9]+]]{{:3}} = "quantum.custom"([[q0:%[0-9]+]], [[q1:%[0-9]+]], [[state0]]{{#1}}) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state2:%[0-9]+]]{{:3}} = "quantum.custom"([[state0]]{{#0}}, [[state1]]{{#2}}, [[state0]]{{#2}}) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state3:%[0-9]+]]{{:3}} = "quantum.custom"([[state1]]{{#0}}, [[state1]]{{#1}}, [[state2]]{{#1}}) {gate_name = "Toffoli"
        # CHECK-NOT: name = "MultiControlledX"
        @for_loop(0, n, 1)
        def loop(i):
            qml.MultiControlledX(wires=[0, 1, 2, 3], work_wires=[4])

        loop()
        return qml.state()

    print(decompose_multicontrolled_x4.mlir)


test_decompose_multicontrolledx_in_for_loop()


def test_decompose_rot():
    dev = get_custom_device_without(1, {"Rot"})

    @qjit(target="mlir")
    @qfunc(1, device=dev)
    # CHECK-LABEL: public @jit_decompose_rot
    def decompose_rot(phi: float, theta: float, omega: float):
        # CHECK-NOT: name = "Rot"
        # CHECK: [[phi:%.+]] = "tensor.extract"(%arg0)
        # CHECK-NOT: name = "Rot"
        # CHECK:  {{%.+}} = "quantum.custom"([[phi]], {{%.+}}) {gate_name = "RZ"
        # CHECK-NOT: name = "Rot"
        # CHECK: [[theta:%.+]] = "tensor.extract"(%arg1)
        # CHECK-NOT: name = "Rot"
        # CHECK: {{%.+}} = "quantum.custom"([[theta]], {{%.+}}) {gate_name = "RY"
        # CHECK-NOT: name = "Rot"
        # CHECK: [[omega:%.+]] = "tensor.extract"(%arg2)
        # CHECK-NOT: name = "Rot"
        # CHECK: {{%.+}} = "quantum.custom"([[omega]], {{%.+}}) {gate_name = "RZ"
        # CHECK-NOT: name = "Rot"
        qml.Rot(phi, theta, omega, wires=0)
        return measure(wires=0)

    print(decompose_rot.mlir)


test_decompose_rot()


def test_decompose_s():
    dev = get_custom_device_without(1, {"S"})

    @qjit(target="mlir")
    @qfunc(1, device=dev)
    # CHECK-LABEL: public @jit_decompose_s
    def decompose_s():
        # CHECK-NOT: name="S"
        # CHECK: [[pi_div_2_t:%.+]] = stablehlo.constant dense<1.57079{{.+}}> : tensor<f64>
        # CHECK-NOT: name = "S"
        # CHECK: [[pi_div_2:%.+]] = "tensor.extract"([[pi_div_2_t]])
        # CHECK-NOT: name = "S"
        # CHECK: {{%.+}} = "quantum.custom"([[pi_div_2]], {{%.+}}) {gate_name = "PhaseShift"
        # CHECK-NOT: name = "S"
        qml.S(wires=0)
        return measure(wires=0)

    print(decompose_s.mlir)


test_decompose_s()


def test_decompose_qubitunitary():
    dev = get_custom_device_without(1, {"QubitUnitary"})

    @qjit(target="mlir")
    @qfunc(1, device=dev)
    # CHECK-LABEL: public @jit_decompose_qubit_unitary
    def decompose_qubit_unitary(U: jax.core.ShapedArray([2, 2], float)):
        # CHECK-NOT: name = "QubitUnitary"
        # CHECK: name = "RZ"
        # CHECK: name = "RY"
        # CHECK: name = "RZ"
        # CHECK-NOT: name = "QubitUnitary"
        qml.QubitUnitary(U, wires=0)
        return measure(wires=0)

    print(decompose_qubit_unitary.mlir)


test_decompose_qubitunitary()


def test_decompose_singleexcitationplus():
    dev = get_custom_device_without(2, {"SingleExcitationPlus"})

    @qjit(target="mlir")
    @qfunc(2, device=dev)
    # CHECK-LABEL: public @jit_decompose_singleexcitationplus
    def decompose_singleexcitationplus(theta: float):
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[a_scalar_tensor_float_2:%.+]] = stablehlo.constant dense<2.{{[0]+}}e+00>
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[a_theta_div_2:%.+]] = stablehlo.divide %arg0, [[a_scalar_tensor_float_2]]
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[b_scalar_tensor_float_2:%.+]] = stablehlo.constant dense<2.{{[0]+}}e+00>
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[b_theta_div_2:%.+]] = stablehlo.divide %arg0, [[b_scalar_tensor_float_2]]
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s0q1:%.+]] = "quantum.custom"({{%.+}}) {gate_name = "PauliX"
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s0q0:%.+]] = "quantum.custom"({{%.+}}) {gate_name = "PauliX"
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[a_theta_div_2_scalar:%.+]] = "tensor.extract"([[a_theta_div_2]])
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s1:%.+]]:2 = "quantum.custom"([[a_theta_div_2_scalar]], [[s0q0]], [[s0q1]]) {gate_name = "ControlledPhaseShift"
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s2q1:%.+]] = "quantum.custom"([[s1]]#1) {gate_name = "PauliX"
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s2q0:%.+]] = "quantum.custom"([[s1]]#0) {gate_name = "PauliX"
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[b_theta_div_2_scalar:%.+]] = "tensor.extract"([[b_theta_div_2]])
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s3:%.+]]:2 = "quantum.custom"([[b_theta_div_2_scalar]], [[s2q1]], [[s2q0]]) {gate_name = "ControlledPhaseShift"
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s4:%.+]]:2 = "quantum.custom"([[s3]]#0, [[s3]]#1) {gate_name = "CNOT"
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[theta_scalar:%.+]] = "tensor.extract"(%arg0)
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s5:%.+]]:2 = "quantum.custom"([[theta_scalar]], [[s4]]#1, [[s4]]#0) {gate_name = "CRY"
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s6:%.+]]:2 = "quantum.custom"([[s5]]#1, [[s5]]#0) {gate_name = "CNOT"
        # CHECK-NOT: name = "SingleExcitationPlus"
        qml.SingleExcitationPlus(theta, wires=[0, 1])
        return measure(wires=0)

    print(decompose_singleexcitationplus.mlir)


test_decompose_singleexcitationplus()
