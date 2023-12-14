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
from catalyst.compiler import get_lib_path

# This is used just for internal testing
from catalyst.pennylane_extensions import qfunc


def get_custom_device_without(num_wires, discards):
    """Generate a custom device without gates in discards."""

    lightning = qml.device("lightning.qubit", wires=3)
    copy = lightning.operations.copy()
    observables_copy = lightning.observables.copy()
    for discard in discards:
        copy.discard(discard)

    class CustomDevice(qml.QubitDevice):
        """Custom Device"""

        name = "Device without some operations"
        short_name = "dummy.device"
        pennylane_requires = "0.1.0"
        version = "0.0.1"
        author = "CV quantum"

        operations = copy
        observables = observables_copy

        # pylint: disable=too-many-arguments
        def __init__(
            self, shots=None, wires=None, backend_name=None, backend_lib=None, backend_kwargs=None
        ):
            self.backend_name = backend_name if backend_name else "default"
            self.backend_lib = backend_lib if backend_lib else "default"
            self.backend_kwargs = backend_kwargs if backend_kwargs else ""
            super().__init__(wires=wires, shots=shots)

        def apply(self, operations, **kwargs):
            pass

        @staticmethod
        def get_c_interface():
            """Location to shared object with C/C++ implementation"""
            return get_lib_path("runtime", "RUNTIME_LIB_DIR") + "/libdummy_device.so"

    return CustomDevice(wires=num_wires)


def test_decompose_multicontrolledx():
    """Test decomposition of MultiControlledX."""
    dev = get_custom_device_without(5, {"MultiControlledX"})

    @qjit(target="mlir")
    @qfunc(device=dev)
    # CHECK-LABEL: public @jit_decompose_multicontrolled_x1
    def decompose_multicontrolled_x1(theta: float):
        qml.RX(theta, wires=[0])
        # pylint: disable=line-too-long
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state0:%.+]]:3 = quantum.custom "Toffoli"() [[q2:%.+]], [[q4:%.+]], [[q3:%.+]]
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state1:%.+]]:3 = quantum.custom "Toffoli"() [[q0:%.+]], [[q1:%.+]], [[state0]]#1
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state2:%.+]]:3 = quantum.custom "Toffoli"() [[state0]]#0, [[state1]]#2, [[state0]]#2
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state3:%.+]]:3 = quantum.custom "Toffoli"() [[state1]]#0, [[state1]]#1, [[state2]]#1
        # CHECK-NOT: name = "MultiControlledX"
        qml.MultiControlledX(wires=[0, 1, 2, 3], work_wires=[4])
        return qml.state()

    print(decompose_multicontrolled_x1.mlir)


test_decompose_multicontrolledx()


def test_decompose_multicontrolledx_in_conditional():
    """Test decomposition of MultiControlledX in conditional."""
    dev = get_custom_device_without(5, {"MultiControlledX"})

    @qjit(target="mlir")
    @qfunc(device=dev)
    # CHECK-LABEL: @jit_decompose_multicontrolled_x2
    def decompose_multicontrolled_x2(theta: float, n: int):
        qml.RX(theta, wires=[0])

        # pylint: disable=line-too-long
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state0:%.+]]:3 = quantum.custom "Toffoli"() [[q2:%.+]], [[q4:%.+]], [[q3:%.+]]
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state1:%.+]]:3 = quantum.custom "Toffoli"() [[q0:%.+]], [[q1:%.+]], [[state0]]#1
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state2:%.+]]:3 = quantum.custom "Toffoli"() [[state0]]#0, [[state1]]#2, [[state0]]#2
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state3:%.+]]:3 = quantum.custom "Toffoli"() [[state1]]#0, [[state1]]#1, [[state2]]#1
        # CHECK-NOT: name = "MultiControlledX"
        @cond(n > 1)
        def cond_fn():
            qml.MultiControlledX(wires=[0, 1, 2, 3], work_wires=[4])

        cond_fn()
        return qml.state()

    print(decompose_multicontrolled_x2.mlir)


test_decompose_multicontrolledx_in_conditional()


def test_decompose_multicontrolledx_in_while_loop():
    """Test decomposition of MultiControlledX in while loop."""
    dev = get_custom_device_without(5, {"MultiControlledX"})

    @qjit(target="mlir")
    @qfunc(device=dev)
    # CHECK-LABEL: @jit_decompose_multicontrolled_x3
    def decompose_multicontrolled_x3(theta: float, n: int):
        qml.RX(theta, wires=[0])

        # pylint: disable=line-too-long
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state0:%[0-9]+]]{{:3}} = quantum.custom "Toffoli"() [[q2:%[0-9]+]], [[q4:%[0-9]+]], [[q3:%[0-9]+]]
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state1:%[0-9]+]]{{:3}} = quantum.custom "Toffoli"() [[q0:%[0-9]+]], [[q1:%[0-9]+]], [[state0]]{{#1}}
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state2:%[0-9]+]]{{:3}} = quantum.custom "Toffoli"() [[state0]]{{#0}}, [[state1]]{{#2}}, [[state0]]{{#2}}
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state3:%[0-9]+]]{{:3}} = quantum.custom "Toffoli"() [[state1]]{{#0}}, [[state1]]{{#1}}, [[state2]]{{#1}}
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
    """Test decomposition of MultiControlledX in for loop."""
    dev = get_custom_device_without(5, {"MultiControlledX"})

    @qjit(target="mlir")
    @qfunc(device=dev)
    # CHECK-LABEL: @jit_decompose_multicontrolled_x4
    def decompose_multicontrolled_x4(theta: float, n: int):
        qml.RX(theta, wires=[0])

        # pylint: disable=line-too-long
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state0:%[0-9]+]]{{:3}} = quantum.custom "Toffoli"() [[q2:%[0-9]+]], [[q4:%[0-9]+]], [[q3:%[0-9]+]]
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state1:%[0-9]+]]{{:3}} = quantum.custom "Toffoli"() [[q0:%[0-9]+]], [[q1:%[0-9]+]], [[state0]]{{#1}}
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state2:%[0-9]+]]{{:3}} = quantum.custom "Toffoli"() [[state0]]{{#0}}, [[state1]]{{#2}}, [[state0]]{{#2}}
        # CHECK-NOT: name = "MultiControlledX"
        # CHECK: [[state3:%[0-9]+]]{{:3}} = quantum.custom "Toffoli"() [[state1]]{{#0}}, [[state1]]{{#1}}, [[state2]]{{#1}}
        # CHECK-NOT: name = "MultiControlledX"
        @for_loop(0, n, 1)
        def loop(_):
            qml.MultiControlledX(wires=[0, 1, 2, 3], work_wires=[4])

        loop()
        return qml.state()

    print(decompose_multicontrolled_x4.mlir)


test_decompose_multicontrolledx_in_for_loop()


def test_decompose_rot():
    """Test decomposition of Rot gate."""
    dev = get_custom_device_without(1, {"Rot"})

    @qjit(target="mlir")
    @qfunc(device=dev)
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
    dev = get_custom_device_without(1, {"S"})

    @qjit(target="mlir")
    @qfunc(device=dev)
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
    dev = get_custom_device_without(1, {"QubitUnitary"})

    @qjit(target="mlir")
    @qfunc(device=dev)
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
    """Test decomposition of single excitation plus."""
    dev = get_custom_device_without(2, {"SingleExcitationPlus"})

    @qjit(target="mlir")
    @qfunc(device=dev)
    # CHECK-LABEL: public @jit_decompose_singleexcitationplus
    def decompose_singleexcitationplus(theta: float):
        # pylint: disable=line-too-long
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[a_scalar_tensor_float_2:%.+]] = stablehlo.constant dense<2.{{[0]+}}e+00>
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[b_theta_div_2:%.+]] = stablehlo.divide %arg0, [[a_scalar_tensor_float_2]]
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[a_theta_div_2:%.+]] = stablehlo.divide %arg0, [[a_scalar_tensor_float_2]]
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s0q1:%.+]] = quantum.custom "PauliX"
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s0q0:%.+]] = quantum.custom "PauliX"
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[a_theta_div_2_scalar:%.+]] = tensor.extract [[a_theta_div_2]]
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s1:%.+]]:2 = quantum.custom "ControlledPhaseShift"([[a_theta_div_2_scalar]]) [[s0q0]], [[s0q1]]
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s2q1:%.+]] = quantum.custom "PauliX"() [[s1]]#1
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s2q0:%.+]] = quantum.custom "PauliX"() [[s1]]#0
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[b_theta_div_2_scalar:%.+]] = tensor.extract [[b_theta_div_2]]
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s3:%.+]]:2 = quantum.custom "ControlledPhaseShift"([[b_theta_div_2_scalar]]) [[s2q1]], [[s2q0]]
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s4:%.+]]:2 = quantum.custom "CNOT"() [[s3]]#0, [[s3]]#1
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[theta_scalar:%.+]] = tensor.extract %arg0
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s5:%.+]]:2 = quantum.custom "CRY"([[theta_scalar]]) [[s4]]#1, [[s4]]#0
        # CHECK-NOT: name = "SingleExcitationPlus"
        # CHECK: [[s6:%.+]]:2 = quantum.custom "CNOT"() [[s5]]#1, [[s5]]#0
        # CHECK-NOT: name = "SingleExcitationPlus"
        qml.SingleExcitationPlus(theta, wires=[0, 1])
        return measure(wires=0)

    print(decompose_singleexcitationplus.mlir)


test_decompose_singleexcitationplus()
