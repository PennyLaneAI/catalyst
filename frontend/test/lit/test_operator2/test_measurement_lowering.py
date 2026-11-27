# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test generic and specialized lowering of mid-circuit measurements."""

# RUN: %PYTHON %s | FileCheck %s

import pennylane as qp

Operator = getattr(qp.core, "Operator2", None) or qp.core.Operator


class RotatedMeasurement(qp.ops.MidMeasure):
    """A measurement with a dynamic parameter and compilable static data."""

    dynamic_argnames = ("angle",)
    compilable_argnames = ("plane",)

    def __init__(self, angle, wires, plane="XY"):
        Operator.__init__(self, angle, wires=wires, plane=plane)


class BareMeasurement(qp.ops.MidMeasure):
    """A measurement whose signature would otherwise qualify for a custom gate."""

    compilable_argnames = ()

    def __init__(self, wires):
        Operator.__init__(self, wires=wires)


@qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
@qp.qnode(qp.device("null.qubit", wires=2))
def direct_measurements():
    # CHECK-LABEL: func.func public @direct_measurements
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.measurement "MidMeasure"() qubits([[q0]])
    # CHECK-SAME: static_data = {meas_uid = "mid", postselect = 1 : i64, reset = true}
    # CHECK-SAME: qubit_map = {wires = [0]} : ()
    qp.ops.MidMeasure(0, reset=True, postselect=1, meas_uid="mid")

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.measurement "PauliMeasure"() qubits([[q1]], [[q2]])
    # CHECK-SAME: static_data = {meas_uid = none, pauli_word = "XZ", postselect = none}
    # CHECK-SAME: qubit_map = {wires = [0, 1]} : ()
    qp.ops.PauliMeasure("XZ", (1, 0))
    return qp.state()


print(direct_measurements.mlir)


@qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
@qp.qnode(qp.device("null.qubit", wires=1))
def custom_measurements(angle: float):
    # CHECK-LABEL: func.func public @custom_measurements
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.measurement "RotatedMeasurement"({{%.+}}: tensor<f64>) qubits([[q0]])
    # CHECK-SAME: static_data = {plane = "YZ"}
    # CHECK-SAME: param_map = {angle = [0]} qubit_map = {wires = [0]} : ()
    RotatedMeasurement(angle, 0, plane="YZ")

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.measurement "BareMeasurement"() qubits([[q1]]) static_data = {}
    # CHECK-SAME: qubit_map = {wires = [0]} : ()
    BareMeasurement(0)
    return qp.state()


print(custom_measurements.mlir)


@qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
@qp.qnode(qp.device("null.qubit", wires=2))
def specialized_measurements(angle: float):
    # CHECK-LABEL: func.func public @specialized_measurements
    # CHECK-NOT: qref.measurement
    # CHECK: qref.measure {{%.+}} postselect 1 : i1
    qp.measure(0, reset=True, postselect=1)
    # CHECK-NOT: qref.measurement
    # CHECK: mbqc.ref.measure_in_basis
    qp.ftqc.measure_arbitrary_basis(0, angle, plane="XY")
    return qp.state()


print(specialized_measurements.mlir)


@qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
@qp.qnode(qp.device("null.qubit", wires=2))
def pauli_measurements():
    # CHECK-LABEL: func.func public @pauli_measurements
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[result:%.+]] = qref.measurement "PauliMeasure"() qubits([[q0]], [[q1]])
    # CHECK-SAME: static_data = {pauli_word = "XZ", postselect = none}
    # CHECK-SAME: qubit_map = {wires = [0, 1]} : (i1)
    result = qp.pauli_measure("XZ", (1, 0))
    qp.cond(result, qp.X)(0)
    return qp.state()


print(pauli_measurements.mlir)
