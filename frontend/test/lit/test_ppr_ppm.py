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

import numpy as np
import pennylane as qml

from catalyst import measure, qjit
from catalyst.passes import (
    commute_ppr,
    merge_ppr_ppm,
    ppm_compilation,
    ppr_to_mbqc,
    ppr_to_ppm,
    reduce_t_depth,
    to_ppr,
)


def test_convert_clifford_to_ppr():
    """
    Test the `to_ppr` pass.
    Check that the original qnode is correctly kept and untransformed.
    """

    pipe = [("pipe", ["quantum-compilation-stage"])]

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

    pipe = [("pipe", ["quantum-compilation-stage"])]

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
# CHECK: %0 = quantum.alloc( 2)
# CHECK: %1 = quantum.extract %0[ 0]
# CHECK: %2 = qec.ppr ["X"](8) %1
# CHECK: %3 = qec.ppr ["Z"](4) %2
# CHECK: %4 = qec.ppr ["X"](4) %3
# CHECK: %5 = qec.ppr ["Z"](4) %4
# CHECK: %6 = quantum.extract %0[ 1]
# CHECK: %7 = qec.ppr ["Z"](8) %6
# CHECK: %8 = qec.ppr ["Z"](4) %7
test_commute_ppr()


def test_commute_ppr_max_pauli_size():
    """
    Test the `commute_ppr` pass with max_pauli_size.
    The Pauli string should not be larger than max_pauli_size.
    """

    pipe = [("pipe", ["quantum-compilation-stage"])]

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

    pipe = [("pipe", ["quantum-compilation-stage"])]

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

    pipe = [("pipe", ["quantum-compilation-stage"])]

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
    Test the pipeline `ppr_to_ppm` pass with decompose method ``"clifford-corrected"`` and `""pauli-corrected"``.
    Check that the `qec.ppr` is correctly decomposed into `qec.ppm`.
    """

    pipe = [("pipe", ["quantum-compilation-stage"])]

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

        @ppr_to_ppm(decompose_method="pauli-corrected")
        @to_ppr
        @qml.qnode(device)
        def cir_pauli_corrected():
            qml.T(0)
            qml.CNOT([0, 1])

        @ppr_to_ppm(decompose_method="pauli-corrected", avoid_y_measure=True)
        @to_ppr
        @qml.qnode(device)
        def cir_pauli_corrected_avoid_y():
            qml.T(0)
            qml.CNOT([0, 1])

        return (
            cir_default(),
            cir_inject_magic_state(),
            cir_pauli_corrected(),
            cir_pauli_corrected_avoid_y(),
        )

    print(circuit_ppr_to_ppm.mlir_opt)


# CHECK-LABEL: public @cir_default_0

# PPR Z(pi/4) should be decomposed.
# CHECK-NOT: qec.ppr ["Z"](4)
# CHECK: quantum.alloc( 2)
# CHECK: quantum.alloc_qb
# CHECK: qec.ppm ["Z", "Y"](-1) {{.+}}, {{.+}} : i1, !quantum.bit, !quantum.bit
# CHECK: qec.ppm ["X"] {{.+}} : i1, !quantum.bit
# CHECK: arith.xori
# CHECK: qec.ppr ["Z"](2) {{.+}} cond({{.+}})

# CHECK-LABEL: public @cir_inject_magic_state_0

# FOR T gate
# CHECK: qec.fabricate  magic
# CHECK: qec.ppm ["Z", "Z"] {{.+}}, {{.+}} cond({{.+}})
# CHECK: qec.ppm ["X"] {{.+}} cond({{.+}})

# FOR CNOT gate
# CHECK: qec.fabricate  plus_i
# Avoid Y-measurement, so Z-measurement should be used
# CHECK: ["Z", "X", "Z"] {{.+}}, {{.+}}, {{.+}}
# CHECK: qec.ppm ["X"] {{.+}} : i1, !quantum.bit
# CHECK: arith.xori
# CHECK: qec.ppr ["Z", "X"](2) {{.+}},{{.+}} cond({{.+}})

# CHECK-LABEL: public @cir_pauli_corrected_0

# FOR T gate
# CHECK: qec.fabricate  magic
# CHECK: qec.ppm ["Z", "Z"] {{.+}}, {{.+}}
# CHECK: qec.select.ppm({{.+}}, ["Y"], ["X"])
# CHECK: qec.ppr ["Z"](2) {{.+}} cond({{.+}})

# FOR CNOT gate
# CHECK: ["Z", "X", "Y"](-1) {{.+}}, {{.+}}, {{.+}}
# CHECK: qec.ppm ["X"] {{.+}}
# CHECK: arith.xori
# CHECK: qec.ppr ["Z", "X"](2) {{.+}},{{.+}} cond({{.+}})

# CHECK-LABEL: public @cir_pauli_corrected_avoid_y_0

# FOR T gate
# CHECK: qec.fabricate  magic
# CHECK: qec.ppm ["Z", "Z"] {{.+}}, {{.+}}
# CHECK: scf.if {{.+}}
# CHECK: qec.fabricate  plus_i
# CHECK: qec.ppm ["Z", "Z"]
# CHECK: qec.ppm ["X", "X"]
# CHECK: qec.ppr ["Z"](2) {{.+}} cond({{.+}})
# CHECK: quantum.dealloc_qb {{.+}}
# CHECK: quantum.dealloc_qb {{.+}}
# CHECK: scf.yield {{.+}}
# CHECK: else
# CHECK: qec.ppm ["X"]
# CHECK: qec.ppr ["Z"](2)
# CHECK: quantum.dealloc_qb {{.+}}
# CHECK: scf.yield {{.+}}

# FOR CNOT gate
# CHECK: qec.fabricate  plus_i
# Avoid Y-measurement, so Z-measurement should be used
# CHECK: ["Z", "X", "Z"] {{.+}}, {{.+}}, {{.+}}
# CHECK: qec.ppm ["X"] {{.+}} : i1, !quantum.bit
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

    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipe, target="mlir")
    def test_clifford_to_ppm_workflow():

        @ppm_compilation(decompose_method="auto-corrected")
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
# CHECK: qec.ppm ["Z", "Y"](-1)
# CHECK: qec.ppm ["X"]
# CEHCK: qec.select.ppm({{.+}}, ["X"], ["Z"])

# CHECK-LABEL: public @cir_clifford_to_ppm_with_params
# decompose Clifford to PPM with params
# CHECK-NOT: qec.ppm [{{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}}]
# CHECK-NOT: qec.ppm [{{.+}}, {{.+}}, {{.+}}, {{.+}}]
# It can be decomposed to three pauli strings in decomposing ppr to ppm
# CHECK: qec.ppm [{{.+}}, {{.+}}, {{.+}}]
test_clifford_to_ppm()


def test_reduce_t_depth():
    """
    Test the `reduce_t_depth` pass.
    """

    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipe, target="mlir")
    @reduce_t_depth
    @merge_ppr_ppm
    @commute_ppr
    @to_ppr
    @qml.qnode(qml.device("null.qubit", wires=3))
    def test_reduce_t_depth_workflow():
        n = 3
        for i in range(n):
            qml.H(wires=i)
            qml.S(wires=i)
            qml.CNOT(wires=[i, (i + 1) % n])
            qml.T(wires=i)
            qml.H(wires=i)
            qml.T(wires=i)

        return [measure(wires=i) for i in range(n)]

    print(test_reduce_t_depth_workflow.mlir_opt)


# CHECK: qec.ppr ["X"](8)
# CHECK: qec.ppr ["X"](8)
# CHECK: qec.ppr ["Y", "X"](8)
# CHECK: qec.ppr ["X", "Y", "X"](8)
# CHECK: qec.ppr ["X"](8)
# CHECK: qec.ppr ["X", "X", "Y"](8)
test_reduce_t_depth()


def test_ppr_to_mbqc():
    """
    Test the `ppr_to_mbqc` pass.
    """

    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipe, target="mlir")
    @ppr_to_mbqc
    @to_ppr
    @qml.qnode(qml.device("null.qubit", wires=2))
    def test_ppr_to_mbqc_workflow():
        qml.H(0)
        qml.CNOT([0, 1])
        return measure(1)

    print(test_ppr_to_mbqc_workflow.mlir_opt)


# CHECK: quantum.custom "Hadamard"
# CHECK: quantum.custom "RZ"
# CHECK: quantum.custom "Hadamard"
# CHECK: quantum.custom "RZ"

# CHECK: quantum.custom "Hadamard"
# CHECK: quantum.custom "CNOT"
# CHECK: quantum.custom "RZ"
# CHECK: quantum.custom "CNOT"
# CHECK: quantum.custom "Hadamard"
# CHECK-NOT: qec.ppr
# CHECK-NOT: qec.ppm
# CHECK: quantum.measure
test_ppr_to_mbqc()


def test_decompose_arbitrary_ppr():
    """
    Test the `decompose_arbitrary_ppr` pass.
    """

    qml.capture.enable()

    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipe, target="mlir")
    @qml.transform(pass_name="decompose-arbitrary-ppr")
    @qml.transform(pass_name="to-ppr")
    @qml.qnode(qml.device("null.qubit", wires=3))
    def test_decompose_arbitrary_ppr_workflow():
        qml.PauliRot(np.pi / 4, pauli_word="Y", wires=0)
        qml.PauliRot(0.123, pauli_word="XYZ", wires=[0, 1, 2])

    print(test_decompose_arbitrary_ppr_workflow.mlir_opt)
    qml.capture.disable()


# CHECK: qec.ppr ["Y"](8)
# CHECK-NOT: qec.ppr ["Z"](2)
# CHECK: qec.prepare  plus
# CHECK: qec.ppm ["X", "Y", "Z", "Z"]
# CHECK: qec.ppr ["X"](2)
# CHECK: qec.ppr.arbitrary ["Z"]
# CHECK: qec.ppm ["X"]
# CHECK: qec.ppr ["X", "Y", "Z"](2)
# CHECK: quantum.dealloc_qb
test_decompose_arbitrary_ppr()
