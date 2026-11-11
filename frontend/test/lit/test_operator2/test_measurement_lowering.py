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

# RUN: %PYTHON %s > %t
# RUN: FileCheck %s < %t
# RUN: quantum-opt --convert-to-value-semantics --canonicalize %t | FileCheck %s --check-prefix=VALUE

import pennylane as qp


class RotatedMeasurement(qp.ops.MidMeasure):
    """A measurement with a dynamic parameter and compilable static data."""

    dynamic_argnames = ("angle",)
    compilable_argnames = ("plane",)

    def __init__(self, angle, wires, plane="XY"):
        qp.core.Operator2.__init__(self, angle, wires=wires, plane=plane)


class BareMeasurement(qp.ops.MidMeasure):
    """A measurement whose signature would otherwise qualify for a custom gate."""

    compilable_argnames = ()

    def __init__(self, wires):
        qp.core.Operator2.__init__(self, wires=wires)


@qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
@qp.qnode(qp.device("null.qubit", wires=2))
def direct_measurements():
    # VALUE-LABEL: func.func public @direct_measurements
    # VALUE: quantum.measurement "MidMeasure"()
    # VALUE: quantum.measurement "PauliMeasure"()
    # CHECK-LABEL: func.func public @direct_measurements
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.measurement "MidMeasure"() qubits([[q0]])
    # CHECK-SAME: static_data = {meas_uid = "mid", postselect = 1 : i64, reset = true} : ()
    qp.ops.MidMeasure(0, reset=True, postselect=1, meas_uid="mid")

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q2:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.measurement "PauliMeasure"() qubits([[q1]], [[q2]])
    # CHECK-SAME: static_data = {meas_uid = none, pauli_word = "XZ", postselect = -1 : si64} : ()
    qp.ops.PauliMeasure("XZ", (1, 0), postselect=-1)
    return qp.state()


print(direct_measurements.mlir)


@qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
@qp.qnode(qp.device("null.qubit", wires=1))
def custom_measurements(angle: float):
    # VALUE-LABEL: func.func public @custom_measurements
    # VALUE: quantum.measurement "RotatedMeasurement"({{%.+}}: tensor<f64>)
    # VALUE: quantum.measurement "BareMeasurement"()
    # CHECK-LABEL: func.func public @custom_measurements
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.measurement "RotatedMeasurement"({{%.+}}: tensor<f64>) qubits([[q0]])
    # CHECK-SAME: static_data = {plane = "YZ"} : ()
    RotatedMeasurement(angle, 0, plane="YZ")

    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: qref.measurement "BareMeasurement"() qubits([[q1]]) static_data = {} : ()
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
    # CHECK-NOT: qref.measurement
    return qp.state()


print(specialized_measurements.mlir)


@qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
@qp.qnode(qp.device("null.qubit", wires=2))
def pauli_measurements():
    # VALUE-LABEL: func.func public @pauli_measurements
    # VALUE: quantum.measurement "PauliMeasure"()
    # VALUE-SAME: static_data = {pauli_word = "XZ", postselect = -1 : si64} : i1,
    # CHECK-LABEL: func.func public @pauli_measurements
    # CHECK-NOT: pbc.
    # CHECK: [[q0:%.+]] = qref.get {{%.+}}[ 1]
    # CHECK: [[q1:%.+]] = qref.get {{%.+}}[ 0]
    # CHECK: [[result:%.+]] = qref.measurement "PauliMeasure"() qubits([[q0]], [[q1]])
    # CHECK-SAME: static_data = {pauli_word = "XZ", postselect = -1 : si64} : i1
    result = qp.pauli_measure("XZ", (1, 0), postselect=-1)
    # CHECK: tensor.from_elements [[result]] : tensor<i1>
    # CHECK: scf.if
    qp.cond(result, qp.X)(0)
    # CHECK-NOT: pbc.
    return qp.state()


print(pauli_measurements.mlir)
