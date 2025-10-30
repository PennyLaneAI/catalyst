# Copyright 2024 Xanadu Quantum Technologies Inc.

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

"""Test Pauli rotations and Pauli measurements with FTQC device."""

import numpy as np
import pennylane as qml
from pennylane.ftqc.catalyst_pass_aliases import (
    commute_ppr,
    merge_ppr_ppm,
    ppm_compilation,
    ppr_to_ppm,
    to_ppr,
)

import catalyst.passes as catalyst_passes
from catalyst import qjit


def test_single_qubit_pauli_rotations():
    """Test single qubit PauliRot"""
    qml.capture.enable()
    dev = qml.device("catalyst.ftqc", wires=1)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 4, "X", wires=0)
        qml.PauliRot(np.pi / 2, "Y", wires=0)
        qml.PauliRot(np.pi, "Z", wires=0)

    # CHECK: [[q0:%.+]] = qec.ppr ["X"](8)
    # CHECK: [[q1:%.+]] = qec.ppr ["Y"](4) [[q0]]
    # CHECK: [[q2:%.+]] = qec.ppr ["Z"](2) [[q1]]
    print(circuit.mlir_opt)
    qml.capture.disable()


test_single_qubit_pauli_rotations()


def test_multi_qubit_pauli_rotations():
    """Test multi-qubit PauliRot"""
    qml.capture.enable()
    dev = qml.device("catalyst.ftqc", wires=3)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 4, "XYZ", wires=[0, 1, 2])
        qml.PauliRot(np.pi / 2, "YZ", wires=[1, 2])
        qml.PauliRot(np.pi, "ZX", wires=[0, 2])
        qml.PauliRot(np.pi / 4, "XYZ", wires=[0, 1, 2])

    # CHECK: [[q0:%.+]]:3 = qec.ppr ["X", "Y", "Z"](8)
    # CHECK: [[q1:%.+]]:2 = qec.ppr ["Y", "Z"](4) [[q0]]#1, [[q0]]#2
    # CHECK: [[q2:%.+]]:2 = qec.ppr ["Z", "X"](2) [[q0]]#0, [[q1]]#1
    # CHECK: [[q3:%.+]]:3 = qec.ppr ["X", "Y", "Z"](8) [[q2]]#0, [[q1]]#0, [[q2]]#1
    print(circuit.mlir_opt)
    qml.capture.disable()


test_multi_qubit_pauli_rotations()


def test_single_qubit_pauli_measurements():
    """Test single qubit PauliMeasure"""
    qml.capture.enable()
    dev = qml.device("catalyst.ftqc", wires=1)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @qml.qnode(device=dev)
    def circuit():
        qml.pauli_measure("X", wires=0)
        qml.pauli_measure("Y", wires=0)
        qml.pauli_measure("Z", wires=0)

    # CHECK: [[m0:%.+]], [[q0:%.+]] = qec.ppm ["X"]
    # CHECK: [[m1:%.+]], [[q1:%.+]] = qec.ppm ["Y"] [[q0]]
    # CHECK: [[m2:%.+]], [[q2:%.+]] = qec.ppm ["Z"] [[q1]]
    print(circuit.mlir_opt)
    qml.capture.disable()


test_single_qubit_pauli_measurements()


def test_multi_qubit_pauli_measurements():
    """Test multi-qubit PauliMeasure"""
    qml.capture.enable()
    dev = qml.device("catalyst.ftqc", wires=3)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @qml.qnode(device=dev)
    def circuit():
        qml.pauli_measure("XYZ", wires=[0, 1, 2])
        qml.pauli_measure("ZX", wires=[1, 2])
        qml.pauli_measure("XYZ", wires=[0, 1, 2])

    # CHECK: [[m0:%.+]], [[q0:%.+]]:3 = qec.ppm ["X", "Y", "Z"]
    # CHECK: [[m1:%.+]], [[q1:%.+]]:2 = qec.ppm ["Z", "X"] [[q0]]#1, [[q0]]#2
    # CHECK: [[m2:%.+]], [[q2:%.+]]:3 = qec.ppm ["X", "Y", "Z"] [[q0]]#0, [[q1]]#0, [[q1]]#1
    print(circuit.mlir_opt)
    qml.capture.disable()


test_multi_qubit_pauli_measurements()


def test_pauli_rot_and_measure_combined():
    """Test PauliRot and PauliMeasrue"""
    qml.capture.enable()
    dev = qml.device("catalyst.ftqc", wires=2)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 4, "X", wires=0)
        qml.PauliRot(np.pi / 2, "Y", wires=1)
        qml.PauliRot(np.pi / 4, "XY", wires=[0, 1])
        qml.pauli_measure("X", wires=0)
        qml.pauli_measure("Y", wires=1)
        qml.pauli_measure("XY", wires=[0, 1])

    # CHECK: qec.ppr ["X"](8)
    # CHECK: qec.ppr ["Y"](4)
    # CHECK: qec.ppr ["X", "Y"](8)
    # CHECK: qec.ppm ["X"]
    # CHECK: qec.ppm ["Y"]
    # CHECK: qec.ppm ["X", "Y"]
    print(circuit.mlir_opt)
    qml.capture.disable()


test_pauli_rot_and_measure_combined()


def test_clifford_t_ppr_ppm_combined():
    """Test to-ppr pass with Clifford+T gates, PPR gates, and PPM gates"""
    qml.capture.enable()
    dev = qml.device("catalyst.ftqc", wires=3)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.S(wires=1)
        qml.Hadamard(wires=1)
        qml.T(wires=1)
        qml.PauliRot(np.pi / 2, "YZ", wires=[1, 2])
        qml.pauli_measure("YZ", wires=[1, 2])

    # CHECK: qec.ppr ["Z"](4)
    # CHECK: qec.ppr ["Z"](4)
    # CHECK: qec.ppr ["X"](4)
    # CHECK: qec.ppr ["Z"](4)
    # CHECK: qec.ppr ["Z"](8)
    # CHECK: qec.ppr ["Y", "Z"](4)
    # CHECK: qec.ppm ["Y", "Z"]
    print(circuit.mlir_opt)
    qml.capture.disable()


test_clifford_t_ppr_ppm_combined()


def test_commute_ppr():
    """Test commute-ppr pass"""
    qml.capture.enable()
    dev = qml.device("catalyst.ftqc", wires=1)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @commute_ppr
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        # The PauliRot and S gate should be commuted to the back after T gate.
        qml.PauliRot(np.pi / 2, "Z", wires=0)  # Clifford gate
        qml.S(wires=0)  # Clifford gate
        qml.T(wires=0)  # Non-Clifford gate

    # CHECK: qec.ppr ["Z"](8)
    # CHECK: qec.ppr ["Z"](4)
    # CHECK: qec.ppr ["Z"](4)
    print(circuit.mlir_opt)
    qml.capture.disable()


test_commute_ppr()


def test_merge_ppr_ppm():
    """Test merge-ppr-ppm pass"""
    qml.capture.enable()
    dev = qml.device("catalyst.ftqc", wires=1)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @merge_ppr_ppm
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 2, "Z", wires=0)
        qml.pauli_measure("X", wires=0)

    # CHECK: qec.ppm ["Y"](-1)
    print(circuit.mlir_opt)
    qml.capture.disable()


test_merge_ppr_ppm()


def test_ppr_to_ppm():
    """Test ppr_to_ppm pass"""
    qml.capture.enable()
    dev = qml.device("catalyst.ftqc", wires=1)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @ppr_to_ppm
    @merge_ppr_ppm
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 2, "X", wires=0)
        qml.PauliRot(np.pi / 4, "Y", wires=0)
        qml.pauli_measure("X", wires=0)

    # CHECK: qec.select.ppm
    # CHECK-NOT: qec.ppr ["Y"](8)
    # CHECK: qec.ppm ["X"]
    print(circuit.mlir_opt)
    qml.capture.disable()


test_ppr_to_ppm()


def test_ppm_compilation():
    """Test ppm_compilation pass"""
    qml.capture.enable()
    dev = qml.device("catalyst.ftqc", wires=1)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @ppm_compilation
    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.PauliRot(np.pi / 2, "X", wires=0)
        qml.PauliRot(np.pi / 4, "Y", wires=0)
        qml.T(wires=0)
        qml.pauli_measure("X", wires=0)

    # CHECK: qec.ppm ["X", "Z"]
    # CHECK: qec.select.ppm
    # CHECK: qec.ppr ["X"](2)
    # CHECK: qec.ppm ["Y", "Z"]
    # CHECK-NOT: qec.ppr ["Z"](8)
    print(circuit.mlir_opt)
    qml.capture.disable()


test_ppm_compilation()


def test_pauli_rot_and_measure_with_cond():
    """Test PauliRot and PauliMeasure works with qml.cond"""
    qml.capture.enable()
    dev = qml.device("catalyst.ftqc", wires=1)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 2, "Z", wires=0)
        m = qml.pauli_measure("Z", wires=0)
        qml.cond(m, qml.PauliRot)(theta=np.pi / 2, pauli_word="Z", wires=0)

    # CHECK: qec.ppr ["Z"](4)
    # CHECK: qec.ppm ["Z"]
    # CHECK: scf.if
    # CHECK: qec.ppr ["Z"](4)
    # CHECK: scf.yield
    # CHECK: else
    # CHECK: scf.yield
    print(circuit.mlir_opt)
    qml.capture.disable()


test_pauli_rot_and_measure_with_cond()


def test_pauli_rot_with_adjoint_and_capture_disabled():
    """Test PauliRot with adjoint and capture disabled"""
    dev = qml.device("catalyst.ftqc", wires=1)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @qml.qnode(device=dev)
    def circuit():
        qml.adjoint(qml.PauliRot(np.pi / 2, "Z", wires=0))

    # CHECK: qec.ppr ["Z"](-4)
    print(circuit.mlir_opt)


test_pauli_rot_with_adjoint_and_capture_disabled()


def test_plain_lowering_with_capture_disabled():
    """Test lowering without passes with capture disabled"""
    dev = qml.device("catalyst.ftqc", wires=1)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 2, "X", wires=0)
        qml.PauliRot(np.pi / 4, "Y", wires=0)
        qml.pauli_measure("X", wires=0)

    # CHECK: qec.ppr ["X"](4)
    # CHECK: qec.ppr ["Y"](8)
    # CHECK: qec.ppm ["X"]
    print(circuit.mlir_opt)


test_plain_lowering_with_capture_disabled()


def test_with_capture_disabled():
    """Test with capture disabled"""
    dev = qml.device("catalyst.ftqc", wires=1)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @catalyst_passes.commute_ppr
    @catalyst_passes.to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.PauliRot(np.pi / 2, "X", wires=0)
        qml.PauliRot(np.pi / 4, "Y", wires=0)
        qml.T(wires=0)
        qml.pauli_measure("X", wires=0)

    # CHECK: qec.ppr ["X"](-8)
    # CHECK: qec.ppr ["Y"](-8)
    # CHECK: qec.ppr ["Z"](4)
    # CHECK: qec.ppr ["X"](4)
    # CHECK: qec.ppr ["Z"](4)
    # CHECK: qec.ppr ["X"](4)
    # CHECK: qec.ppm ["X"]
    print(circuit.mlir_opt)


test_with_capture_disabled()
