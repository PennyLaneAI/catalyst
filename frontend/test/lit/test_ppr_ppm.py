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
import pennylane as qp

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
    @qp.qnode(qp.device("null.qubit", wires=2))
    def circuit():
        qp.H(0)
        qp.S(1)
        qp.T(0)
        qp.CNOT([0, 1])
        return measure(0)

    print(circuit.mlir_opt)


# CHECK-NOT: transform.apply_registered_pass "to-ppr"
# CHECK: pbc.ppr
# CHECK: pbc.ppm
test_convert_clifford_to_ppr()


def test_commute_ppr():
    """
    Test the `commute_ppr` pass.
    Ensure that the `pbc.ppr` with pi/8 rotations are moved to the beginning of the circuit.
    """

    pipe = [("pipe", ["quantum-compilation-stage"])]

    # CHECK-LABEL: public @cir_commute_ppr
    @qjit(pipelines=pipe, target="mlir")
    @commute_ppr
    @to_ppr
    @qp.qnode(qp.device("null.qubit", wires=2))
    def cir_commute_ppr():
        # CHECK: %0 = quantum.alloc( 2)
        # CHECK: %1 = quantum.extract %0[ 0]
        # CHECK: %2 = pbc.ppr ["X"](8) %1
        # CHECK: %3 = pbc.ppr ["Z"](4) %2
        # CHECK: %4 = pbc.ppr ["X"](4) %3
        # CHECK: %5 = pbc.ppr ["Z"](4) %4
        # CHECK: %6 = quantum.extract %0[ 1]
        # CHECK: %7 = pbc.ppr ["Z"](8) %6
        # CHECK: %8 = pbc.ppr ["Z"](4) %7
        qp.H(0)
        qp.S(1)
        qp.T(0)
        qp.T(1)
        qp.CNOT([0, 1])
        return measure(0), measure(1)

    print(cir_commute_ppr.mlir_opt)


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
    @qp.qnode(qp.device("null.qubit", wires=2))
    def cir_commute_ppr_max_pauli_size():
        qp.CNOT([0, 2])
        qp.T(0)
        qp.T(1)
        qp.CNOT([0, 1])
        qp.S(0)
        qp.H(0)
        qp.T(0)
        return measure(0), measure(1)

    print(cir_commute_ppr_max_pauli_size.mlir_opt)


# CHECK-LABEL: public @cir_commute_ppr_max_pauli_size
# CHECK-NOT: pbc.ppr ["Y", "X", "X"](-8)
# CHECK: pbc.ppr ["X", "X"](8)
test_commute_ppr_max_pauli_size()


def test_merge_ppr_ppm():
    """
    Test the `merge_ppr_ppm` pass.
    `pbc.ppr` should be merged into `pbc.ppm`, thus no `pbc.ppr` should be left.
    """

    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipe, target="mlir")
    @merge_ppr_ppm
    @to_ppr
    @qp.qnode(qp.device("null.qubit", wires=2))
    def cir_merge_ppr_ppm():
        qp.H(0)
        qp.S(1)
        qp.CNOT([0, 1])
        return measure(0), measure(1)

    print(cir_merge_ppr_ppm.mlir_opt)


# CHECK-LABEL: public @cir_merge_ppr_ppm
# CHECK-NOT: pbc.ppr
# CHECK: pbc.ppm ["Z", "X"]
# CHECK: pbc.ppm ["X"]
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
    @qp.qnode(qp.device("null.qubit", wires=2))
    def cir_merge_ppr_ppm_max_pauli_size():
        qp.CNOT([0, 2])
        qp.T(0)
        qp.T(1)
        qp.CNOT([0, 1])
        return measure(0), measure(1)

    print(cir_merge_ppr_ppm_max_pauli_size.mlir_opt)


# CHECK-LABEL: public @cir_merge_ppr_ppm_max_pauli_size
# CHECK-NOT: pbc.ppm ["Z", "Z"]
# CHECK:  pbc.ppm ["Y"](-)
test_merge_ppr_ppm_max_pauli_size()


def test_ppr_to_ppm():
    """
    Test the pipeline `ppr_to_ppm` pass with decompose method ``"clifford-corrected"`` and `""pauli-corrected"``.
    Check that the `pbc.ppr` is correctly decomposed into `pbc.ppm`.
    """

    pipe = [("pipe", ["quantum-compilation-stage"])]

    device = qp.device("null.qubit", wires=2)

    @qjit(pipelines=pipe, target="mlir")
    def circuit_ppr_to_ppm():

        # CHECK-LABEL: public @cir_default_0
        @ppr_to_ppm
        @to_ppr
        @qp.qnode(device)
        def cir_default():
            # PPR Z(pi/4) should be decomposed.
            # CHECK-NOT: pbc.ppr ["Z"](4)
            # CHECK: quantum.alloc( 2)
            # CHECK: quantum.alloc_qb
            # CHECK: pbc.ppm ["Z", "Y"](-)
            # CHECK: pbc.ppm ["X"] {{.+}} : i1, !quantum.bit
            # CHECK: [[pred:%.+]] = arith.xori
            # CHECK: pbc.ppr ["Z"](2) {{.+}} cond([[pred]])
            qp.S(0)

        # CHECK-LABEL: public @cir_inject_magic_state_0
        @ppr_to_ppm(decompose_method="clifford-corrected", avoid_y_measure=True)
        @to_ppr
        @qp.qnode(device)
        def cir_inject_magic_state():
            # CHECK: quantum.custom "Hadamard"
            # CHECK: quantum.custom "T"
            # CHECK: [[m:%.+]], {{.+}} = pbc.ppm ["Z", "Z"]
            # CHECK: scf.if [[m]]
            # CHECK:   quantum.custom "Hadamard"
            # CHECK:   quantum.custom "S"
            # CHECK:   pbc.ppm ["Z", "Z"]
            # CHECK:   pbc.ppm ["X"] {{.+}}
            # CHECK:   scf.yield
            qp.T(0)

            # CHECK: quantum.custom "Hadamard"
            # CHECK: quantum.custom "S"
            # Avoid Y-measurement, so Z-measurement should be used
            # CHECK: pbc.ppm ["Z", "X", "Z"]
            # CHECK: pbc.ppm ["X"]
            # CHECK: [[pred:%.+]] = arith.xori
            # CHECK: pbc.ppr ["Z", "X"](2) {{.+}}, {{.+}} cond([[pred]])
            qp.CNOT([0, 1])

        # CHECK-LABEL: public @cir_pauli_corrected_0
        @ppr_to_ppm(decompose_method="pauli-corrected")
        @to_ppr
        @qp.qnode(device)
        def cir_pauli_corrected():
            # CHECK: quantum.custom "Hadamard"
            # CHECK: quantum.custom "T"
            # CHECK: [[m:%.+]], {{.+}} = pbc.ppm ["Z", "Z"]
            # CHECK: pbc.select.ppm ([[m]] ? ["Y"] : ["X"])
            # CHECK: pbc.ppr ["Z"](2) {{.+}} cond(
            qp.T(0)

            # CHECK: pbc.ppm ["Z", "X", "Y"](-)
            # CHECK: pbc.ppm ["X"]
            # CHECK: [[pred:%.+]] = arith.xori
            # CHECK: pbc.ppr ["Z", "X"](2) {{.+}},{{.+}} cond([[pred]])
            qp.CNOT([0, 1])

        # CHECK-LABEL: public @cir_pauli_corrected_avoid_y_0
        @ppr_to_ppm(decompose_method="pauli-corrected", avoid_y_measure=True)
        @to_ppr
        @qp.qnode(device)
        def cir_pauli_corrected_avoid_y():
            # CHECK: quantum.custom "Hadamard"
            # CHECK: quantum.custom "T"
            # CHECK: [[m:%.+]], {{.+}} = pbc.ppm ["Z", "Z"]
            # CHECK: scf.if [[m]]
            # CHECK:   quantum.custom "Hadamard"
            # CHECK:   quantum.custom "S"
            # CHECK:   pbc.ppm ["Z", "Z"]
            # CHECK:   pbc.ppm ["X", "X"]
            # CHECK:   pbc.ppr ["Z"](2) {{.+}} cond(
            # CHECK:   quantum.dealloc_qb
            # CHECK:   quantum.dealloc_qb
            # CHECK:   scf.yield
            # CHECK: else
            # CHECK:   pbc.ppm ["X"]
            # CHECK:   pbc.ppr ["Z"](2)
            # CHECK:   quantum.dealloc_qb
            # CHECK:   scf.yield
            qp.T(0)

            # CHECK: quantum.custom "Hadamard"
            # CHECK: quantum.custom "S"
            # # Avoid Y-measurement, so Z-measurement should be used
            # CHECK: pbc.ppm ["Z", "X", "Z"]
            # CHECK: pbc.ppm ["X"]
            # CHECK: [[pred:%.+]] = arith.xori
            # CHECK: pbc.ppr ["Z", "X"](2) {{.+}},{{.+}} cond([[pred]])
            qp.CNOT([0, 1])

        return (
            cir_default(),
            cir_inject_magic_state(),
            cir_pauli_corrected(),
            cir_pauli_corrected_avoid_y(),
        )

    print(circuit_ppr_to_ppm.mlir_opt)


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

        # CHECK-LABEL: public @cir_clifford_to_ppm
        @ppm_compilation(decompose_method="auto-corrected")
        @qp.qnode(qp.device("null.qubit", wires=2))
        def cir_clifford_to_ppm():
            # decompose Clifford to PPM
            # CHECK: pbc.select.ppm ({{%.+}} ? ["X"] : ["Z"])
            # CHECK: pbc.ppm ["X", "Z", "Z"]
            # CHECK: pbc.ppm ["Z", "Y"](-)
            # CHECK: pbc.ppm ["X"]
            # CHECK: pbc.select.ppm ({{%.+}} ? ["X"] : ["Z"])
            qp.H(0)
            qp.CNOT(wires=[0, 1])
            qp.T(0)
            qp.T(1)
            return [measure(0), measure(1)]

        # CHECK-LABEL: public @cir_clifford_to_ppm_with_params
        @ppm_compilation(
            decompose_method="clifford-corrected", avoid_y_measure=True, max_pauli_size=2
        )
        @qp.qnode(qp.device("null.qubit", wires=5))
        def cir_clifford_to_ppm_with_params():
            # decompose Clifford to PPM with params
            # CHECK-NOT: pbc.ppm [{{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}}, {{.+}}]
            # CHECK-NOT: pbc.ppm [{{.+}}, {{.+}}, {{.+}}, {{.+}}]
            # It can be decomposed to three pauli strings in decomposing ppr to ppm
            # CHECK: pbc.ppm [{{.+}}, {{.+}}, {{.+}}]
            for idx in range(5):
                qp.H(idx)
                qp.CNOT(wires=[idx, idx + 1])
                qp.CNOT(wires=[idx + 1, (idx + 2) % 5])
                qp.T(idx)
                qp.T(idx + 1)
            return [measure(idx) for idx in range(5)]

        return cir_clifford_to_ppm(), cir_clifford_to_ppm_with_params()

    print(test_clifford_to_ppm_workflow.mlir_opt)


test_clifford_to_ppm()


def test_reduce_t_depth():
    """
    Test the `reduce_t_depth` pass.
    """

    pipe = [("pipe", ["quantum-compilation-stage"])]

    # CHECK-LABEL: public @test_reduce_t_depth_workflow
    @qjit(pipelines=pipe, target="mlir")
    @reduce_t_depth
    @merge_ppr_ppm
    @commute_ppr
    @to_ppr
    @qp.qnode(qp.device("null.qubit", wires=3))
    def test_reduce_t_depth_workflow():
        # CHECK: pbc.ppr ["X"](8)
        # CHECK: pbc.ppr ["X"](8)
        # CHECK: pbc.ppr ["Y", "X"](8)
        # CHECK: pbc.ppr ["X", "Y", "X"](8)
        # CHECK: pbc.ppr ["X"](8)
        # CHECK: pbc.ppr ["X", "X", "Y"](8)
        n = 3
        for i in range(n):
            qp.H(wires=i)
            qp.S(wires=i)
            qp.CNOT(wires=[i, (i + 1) % n])
            qp.T(wires=i)
            qp.H(wires=i)
            qp.T(wires=i)

        return [measure(wires=i) for i in range(n)]

    print(test_reduce_t_depth_workflow.mlir_opt)


test_reduce_t_depth()


def test_ppr_to_mbqc():
    """
    Test the `ppr_to_mbqc` pass.
    """

    pipe = [("pipe", ["quantum-compilation-stage"])]

    # CHECK-LABEL: public @test_ppr_to_mbqc_workflow
    @qjit(pipelines=pipe, target="mlir")
    @ppr_to_mbqc
    @to_ppr
    @qp.qnode(qp.device("null.qubit", wires=2))
    def test_ppr_to_mbqc_workflow():
        # CHECK: quantum.custom "Hadamard"
        # CHECK: quantum.custom "RZ"
        # CHECK: quantum.custom "Hadamard"
        # CHECK: quantum.custom "RZ"

        # CHECK: quantum.custom "Hadamard"
        # CHECK: quantum.custom "CNOT"
        # CHECK: quantum.custom "RZ"
        # CHECK: quantum.custom "CNOT"
        # CHECK: quantum.custom "Hadamard"
        # CHECK-NOT: pbc.ppr
        # CHECK-NOT: pbc.ppm
        # CHECK: quantum.measure
        qp.H(0)
        qp.CNOT([0, 1])
        return measure(1)

    print(test_ppr_to_mbqc_workflow.mlir_opt)


test_ppr_to_mbqc()


def test_decompose_arbitrary_ppr():
    """
    Test the `decompose_arbitrary_ppr` pass.
    """

    qp.capture.enable()

    pipe = [("pipe", ["quantum-compilation-stage"])]

    # CHECK-LABEL: public @test_decompose_arbitrary_ppr_workflow
    @qjit(pipelines=pipe, target="mlir")
    @qp.transform(pass_name="decompose-arbitrary-ppr")
    @qp.transform(pass_name="to-ppr")
    @qp.qnode(qp.device("null.qubit", wires=3))
    def test_decompose_arbitrary_ppr_workflow():
        # CHECK: pbc.ppr ["Y"](8)
        # CHECK-NOT: pbc.ppr ["Z"](2)
        # CHECK: quantum.custom "Hadamard"
        # CHECK: pbc.ppm ["X", "Y", "Z", "Z"]
        # CHECK: pbc.ppr ["X"](2)
        # CHECK: pbc.ppr.arbitrary ["Z"]
        # CHECK: pbc.ppm ["X"]
        # CHECK: pbc.ppr ["X", "Y", "Z"](2)
        # CHECK: quantum.dealloc_qb
        qp.PauliRot(np.pi / 4, pauli_word="Y", wires=0)
        qp.PauliRot(0.123, pauli_word="XYZ", wires=[0, 1, 2])

    print(test_decompose_arbitrary_ppr_workflow.mlir_opt)
    qp.capture.disable()


test_decompose_arbitrary_ppr()
