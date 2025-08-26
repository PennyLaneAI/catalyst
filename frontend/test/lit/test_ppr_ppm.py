# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file performs the frontend tests that the PPR and PPM passes are correctly lowered."""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable=line-too-long

import pennylane as qml

from catalyst import measure, qjit
from catalyst.passes import commute_ppr, merge_ppr_ppm, ppm_compilation, ppr_to_ppm, to_ppr


def test_convert_clifford_to_ppr():
    """
    Test the `to_ppr` pass.
    Check that the original qnode is correctly kept and untransformed.
    """

    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipe, target="mlir")
    @to_ppr
    @qml.qnode(qml.device("null.qubit", wires=2))
    def circuit():
        qml.H(0)
        qml.S(1)
        qml.T(0)
        qml.CNOT([0, 1])
        return measure(0)

    print(circuit.mlir_opt)


# CHECK-NOT: transform.apply_registered_pass "to-ppr"
# CHECK: qec.ppr
# CHECK: qec.ppm
test_convert_clifford_to_ppr()


def test_commute_ppr():
    """
    Test the `commute_ppr` pass.
    Ensure that the `qec.ppr` with pi/8 rotations are moved to the beginning of the circuit.
    """

    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipe, target="mlir")
    @commute_ppr
    @to_ppr
    @qml.qnode(qml.device("null.qubit", wires=2))
    def cir_commute_ppr():
        qml.H(0)
        qml.S(1)
        qml.T(0)
        qml.T(1)
        qml.CNOT([0, 1])
        return measure(0), measure(1)

    print(cir_commute_ppr.mlir_opt)


# CHECK-LABEL: public @cir_commute_ppr
# CHECK: qec.ppr ["X"](8)
# CHECK: qec.ppr ["Z"](8)
# CHECK: qec.ppr ["Z"](4)
# CHECK: qec.ppr ["Z"](4)
test_commute_ppr()


def test_commute_ppr_max_pauli_size():
    """
    Test the `commute_ppr` pass with max_pauli_size.
    The Pauli string should not be larger than max_pauli_size.
    """

    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipe, target="mlir")
    @commute_ppr(max_pauli_size=2)
    @to_ppr
    @qml.qnode(qml.device("null.qubit", wires=2))
    def cir_commute_ppr_max_pauli_size():
        qml.CNOT([0, 2])
        qml.T(0)
        qml.T(1)
        qml.CNOT([0, 1])
        qml.S(0)
        qml.H(0)
        qml.T(0)
        return measure(0), measure(1)

    print(cir_commute_ppr_max_pauli_size.mlir_opt)


# CHECK-LABEL: public @cir_commute_ppr_max_pauli_size
# CHECK-NOT: qec.ppr ["Y", "X", "X"](-8)
# CHECK: qec.ppr ["X", "X"](8)
test_commute_ppr_max_pauli_size()


def test_merge_ppr_ppm():
    """
    Test the `merge_ppr_ppm` pass.
    `qec.ppr` should be merged into `qec.ppm`, thus no `qec.ppr` should be left.
    """

    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipe, target="mlir")
    @merge_ppr_ppm
    @to_ppr
    @qml.qnode(qml.device("null.qubit", wires=2))
    def cir_merge_ppr_ppm():
        qml.H(0)
        qml.S(1)
        qml.CNOT([0, 1])
        return measure(0), measure(1)

    print(cir_merge_ppr_ppm.mlir_opt)


# CHECK-LABEL: public @cir_merge_ppr_ppm
# CHECK-NOT: qec.ppr
# CHECK: qec.ppm ["Z", "X"]
# CHECK: qec.ppm ["X"]
test_merge_ppr_ppm()


def test_merge_ppr_ppm_max_pauli_size():
    """
    Test the `merge_ppr_ppm` pass with max_pauli_size.
    The Pauli string should not be larger than max_pauli_size.
    """

    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipe, target="mlir")
    @merge_ppr_ppm(max_pauli_size=1)
    @to_ppr
    @qml.qnode(qml.device("null.qubit", wires=2))
    def cir_merge_ppr_ppm_max_pauli_size():
        qml.CNOT([0, 2])
        qml.T(0)
        qml.T(1)
        qml.CNOT([0, 1])
        return measure(0), measure(1)

    print(cir_merge_ppr_ppm_max_pauli_size.mlir_opt)


# CHECK-LABEL: public @cir_merge_ppr_ppm_max_pauli_size
# CHECK-NOT: qec.ppm ["Z", "Z"]
# CHECK:  qec.ppm ["Y"](-1)
test_merge_ppr_ppm_max_pauli_size()


def test_ppr_to_ppm():
    """
    Test the pipeline `ppr_to_ppm` pass.
    Check that the `qec.ppr` is correctly decomposed into `qec.ppm`.
    """

    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    device = qml.device("null.qubit", wires=2)

    @qjit(pipelines=pipe, target="mlir")
    def circuit_ppr_to_ppm():

        @ppr_to_ppm
        @to_ppr
        @qml.qnode(device)
        def cir_default():
            qml.S(0)

        @ppr_to_ppm(decompose_method="clifford-corrected", avoid_y_measure=True)
        @to_ppr
        @qml.qnode(device)
        def cir_inject_magic_state():
            qml.T(0)
            qml.CNOT([0, 1])

        return cir_default(), cir_inject_magic_state()

    print(circuit_ppr_to_ppm.mlir_opt)


# CHECK-LABEL: public @cir_default_0

# PPR Z(pi/4) should be decomposed.
# CHECK-NOT: qec.ppr ["Z"](4)
# CHECK: quantum.alloc( 2)
# CHECK: quantum.alloc_qb
# CHECK: qec.ppm ["Z", "Y"] {{.+}}, {{.+}} : !quantum.bit, !quantum.bit
# CHECK: qec.ppm ["X"] {{.+}} : !quantum.bit
# CHECK: arith.xori
# CHECK: qec.ppr ["Z"](2) {{.+}} cond({{.+}})

# CHECK-LABEL: public @cir_inject_magic_state_0

# FOR T gate
# CHECK: qec.fabricate  magic
# CHECK: qec.ppm ["Z", "Z"](-1) {{.+}}, {{.+}} cond({{.+}})
# CHECK: qec.ppm ["X"] {{.+}} cond({{.+}})

# FOR CNOT gate
# CHECK: qec.fabricate  plus_i
# Avoid Y-measurement, so Z-measurement should be used
# CHECK: ["Z", "X", "Z"](-1) {{.+}}, {{.+}}, {{.+}} :
# CHECK: qec.ppm ["X"] {{.+}} : !quantum.bit
# CHECK: arith.xori
# CHECK: qec.ppr ["Z", "X"](2) {{.+}},{{.+}} cond({{.+}})
test_ppr_to_ppm()


def test_clifford_to_ppm():
    """
    Test the pipeline `to_ppm` pass.
    Check whole pipeline of PPM's sub-passes.
    The Pauli string should not be larger than max_pauli_size, but in ppr_to_ppm,
    the Pauli string can increase by one because of an additional auxiliary qubit.
    """

    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipe, target="mlir")
    def test_clifford_to_ppm_workflow():

        @ppm_compilation
        @qml.qnode(qml.device("null.qubit", wires=2))
        def cir_clifford_to_ppm():
            qml.H(0)
            qml.CNOT(wires=[0, 1])
            qml.T(0)
            qml.T(1)
            return [measure(0), measure(1)]

        @ppm_compilation(
            decompose_method="clifford-corrected", avoid_y_measure=True, max_pauli_size=2
        )
        @qml.qnode(qml.device("null.qubit", wires=5))
        def cir_clifford_to_ppm_with_params():
            for idx in range(5):
                qml.H(idx)
                qml.CNOT(wires=[idx, idx + 1])
                qml.CNOT(wires=[idx + 1, (idx + 2) % 5])
                qml.T(idx)
                qml.T(idx + 1)
            return [measure(idx) for idx in range(5)]

        return cir_clifford_to_ppm(), cir_clifford_to_ppm_with_params()

    print(test_clifford_to_ppm_workflow.mlir_opt)


# CHECK-LABEL: public @cir_clifford_to_ppm
# decompose Clifford to PPM
# CHECK: qec.select.ppm({{.+}}, ["X"], ["Z"])
# CHECK: qec.ppm ["X", "Z", "Z"]
# CHECK: qec.ppm ["Z", "Y"]
# CHECK: qec.ppm ["X"]
# CEHCK: qec.select.ppm({{.+}}, ["X"], ["Z"])

# CHECK-LABEL: public @cir_clifford_to_ppm_with_params
# decompose Clifford to PPM with params
# CHECK-NOT: qec.ppm [{{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}}]
# CHECK-NOT: qec.ppm [{{.+}}, {{.+}}, {{.+}}, {{.+}}]
# It can be decomposed to three pauli strings in decomposing ppr to ppm
# CHECK: qec.ppm [{{.+}}, {{.+}}, {{.+}}]
test_clifford_to_ppm()
