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

from functools import partial

import numpy as np
import pennylane as qml
from pennylane.ftqc.catalyst_pass_aliases import (
    commute_ppr,
    merge_ppr_ppm,
    ppm_compilation,
    ppr_to_ppm,
    to_ppr,
)

from catalyst import qjit


def test_pauli_rot_lowering():
    """Test that qml.PauliRot is lowered to quantum.paulirot."""
    qml.capture.enable()
    dev = qml.device("null.qubit", wires=1)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipeline, target="mlir")
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 4, "X", wires=0)

    # CHECK: [[cst:%.+]]  = arith.constant 0.78539816339744828
    # CHECK: quantum.paulirot ["X"]([[cst]])
    print(circuit.mlir_opt)
    qml.capture.disable()


test_pauli_rot_lowering()


def test_single_qubit_pauli_rotations():
    """Test single qubit PauliRot"""
    qml.capture.enable()
    dev = qml.device("null.qubit", wires=1)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
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


def test_arbitrary_angle_pauli_rotations():
    """Test arbitrary angle PauliRot"""
    qml.capture.enable()
    dev = qml.device("null.qubit", wires=1)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(0.42, "X", wires=0)

    # CHECK: [[cst:%.+]] = arith.constant 2.100000e-01 : f64
    # CHECK: [[q0:%.+]] = qec.ppr.arbitrary ["X"]([[cst]])
    print(circuit.mlir_opt)
    qml.capture.disable()


test_arbitrary_angle_pauli_rotations()


def test_arbitrary_negative_angle_pauli_rotations():
    """Test arbitrary angle PauliRot"""
    qml.capture.enable()
    dev = qml.device("null.qubit", wires=1)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(-0.42, "X", wires=0)

    # CHECK: [[cst:%.+]] = arith.constant -2.100000e-01 : f64
    # CHECK: [[q0:%.+]] = qec.ppr.arbitrary ["X"]([[cst]])
    print(circuit.mlir_opt)
    qml.capture.disable()


test_arbitrary_negative_angle_pauli_rotations()


def test_dynamic_angle_pauli_rotations():
    """Test dynamic angle PauliRot"""
    qml.capture.enable()
    dev = qml.device("null.qubit", wires=1)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
    @qml.qnode(device=dev)
    def circuit(x: float):
        qml.PauliRot(x, "X", wires=0)

    # CHECK: [[cst:%.+]] = arith.constant 2.000000e+00 : f64
    # CHECK: [[extracted:%.+]] = tensor.extract
    # CHECK: [[div:%.+]] = arith.divf [[extracted]], [[cst]] : f64
    # CHECK: [[q0:%.+]] = qec.ppr.arbitrary ["X"]([[div]])
    print(circuit.mlir_opt)
    qml.capture.disable()


test_dynamic_angle_pauli_rotations()


def test_multi_qubit_pauli_rotations():
    """Test multi-qubit PauliRot"""
    qml.capture.enable()
    dev = qml.device("null.qubit", wires=3)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
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


def test_arbitrary_angle_multi_qubit_pauli_rotations():
    """Test arbitrary angle multi-qubit PauliRot"""
    qml.capture.enable()
    dev = qml.device("null.qubit", wires=3)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(0.42, "XZ", wires=[0, 1])
        qml.PauliRot(0.84, "YX", wires=[0, 1])

    # CHECK: [[cst:%.+]] = arith.constant 4.200000e-01 : f64
    # CHECK: [[cst_1:%.+]] = arith.constant 2.100000e-01 : f64
    # CHECK: [[q0:%.+]]:2 = qec.ppr.arbitrary ["X", "Z"]([[cst_1]])
    # CHECK: [[q1:%.+]]:2 = qec.ppr.arbitrary ["Y", "X"]([[cst]]) [[q0]]#0, [[q0]]#1
    print(circuit.mlir_opt)
    qml.capture.disable()


test_arbitrary_angle_multi_qubit_pauli_rotations()


def test_dynamic_angle_multi_qubit_pauli_rotations():
    """Test dynamic angle multi-qubit PauliRot"""
    qml.capture.enable()
    dev = qml.device("null.qubit", wires=3)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
    @qml.qnode(device=dev)
    def circuit(x: float):
        qml.PauliRot(x, "XZ", wires=[0, 1])
        qml.PauliRot(x, "YX", wires=[0, 1])

    # CHECK: [[cst:%.+]] = arith.constant 2.000000e+00 : f64
    # CHECK: [[extracted:%.+]] = tensor.extract
    # CHECK: [[div:%.+]] = arith.divf [[extracted]], [[cst]] : f64
    # CHECK: [[q0:%.+]]:2 = qec.ppr.arbitrary ["X", "Z"]([[div]])
    # CHECK: [[extracted_1:%.+]] = tensor.extract
    # CHECK: [[div_1:%.+]] = arith.divf [[extracted_1]], [[cst]] : f64
    # CHECK: [[q1:%.+]]:2 = qec.ppr.arbitrary ["Y", "X"]([[div_1]]) [[q0]]#0, [[q0]]#1
    print(circuit.mlir_opt)
    qml.capture.disable()


test_dynamic_angle_multi_qubit_pauli_rotations()


def test_single_qubit_pauli_measurements():
    """Test single qubit PauliMeasure"""
    qml.capture.enable()
    dev = qml.device("null.qubit", wires=1)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
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
    dev = qml.device("null.qubit", wires=3)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
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
    dev = qml.device("null.qubit", wires=2)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
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
    dev = qml.device("null.qubit", wires=3)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

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
    dev = qml.device("null.qubit", wires=1)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

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
    dev = qml.device("null.qubit", wires=1)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipeline, target="mlir")
    @merge_ppr_ppm
    @to_ppr
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
    dev = qml.device("null.qubit", wires=1)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipeline, target="mlir")
    @ppr_to_ppm
    @merge_ppr_ppm
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 2, "X", wires=0)
        qml.PauliRot(np.pi / 4, "Y", wires=0)
        qml.pauli_measure("X", wires=0)

    # CHECK: qec.ppm ["X", "Y"]
    # CHECK: qec.ppm ["Y", "Z"]
    # CHECK: scf.if
    # CHECK: qec.ppm ["Y"]
    # CHECK: }
    # CHECK: else
    # CHECK: qec.ppm ["X"]
    # CHECK: }
    # CHECK-NOT: qec.ppr ["Y"](8)
    # CHECK: qec.ppm ["X"]
    print(circuit.mlir_opt)
    qml.capture.disable()


test_ppr_to_ppm()


def test_ppm_compilation():
    """Test ppm_compilation pass"""
    qml.capture.enable()
    dev = qml.device("null.qubit", wires=1)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

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
    # CHECK: scf.if
    # CHECK: qec.ppm ["Y"]
    # CHECK: }
    # CHECK: else
    # CHECK: qec.ppm ["X"]
    # CHECK: }
    # CHECK: qec.ppr ["X"](2)
    # CHECK: qec.ppm ["Y", "Z"]
    # CHECK-NOT: qec.ppr ["Z"](8)
    print(circuit.mlir_opt)
    qml.capture.disable()


test_ppm_compilation()


def test_pauli_rot_and_measure_with_cond():
    """Test PauliRot and PauliMeasure works with qml.cond"""
    qml.capture.enable()
    dev = qml.device("null.qubit", wires=1)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 2, wires=0, pauli_word="Z")
        m = qml.pauli_measure("Z", wires=0)
        qml.cond(m, partial(qml.PauliRot, pauli_word="Z"))(theta=np.pi / 2, wires=0)

    # CHECK: [[q0:%.+]] = qec.ppr ["Z"](4)
    # CHECK: qec.ppm ["Z"]
    # CHECK: scf.if
    # CHECK: qec.ppr ["Z"](4)
    # CHECK: scf.yield
    # CHECK: else
    # CHECK: scf.yield
    print(circuit.mlir_opt)
    qml.capture.disable()


test_pauli_rot_and_measure_with_cond()


def test_pauli_rot_with_adjoint_region():
    """Test PauliRot with qml.adjoint region"""
    qml.capture.enable()
    dev = qml.device("null.qubit", wires=2)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    def f():
        qml.PauliRot(np.pi / 4, "XZ", wires=[0, 1])

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 2, "YX", wires=[0, 1])
        qml.adjoint(f)()

    # CHECK: qec.ppr ["Y", "X"](4)
    # CHECK: qec.ppr ["X", "Z"](-8)
    print(circuit.mlir_opt)
    qml.capture.disable()


test_pauli_rot_with_adjoint_region()


def test_pauli_rot_with_adjoint_single_gate():
    """Test PauliRot with qml.adjoint as a direct gate"""
    qml.capture.enable()
    dev = qml.device("null.qubit", wires=2)

    pipeline = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.adjoint(qml.PauliRot(np.pi / 2, "XZ", wires=[0, 1]))

    # CHECK: qec.ppr ["X", "Z"](-4)
    print(circuit.mlir_opt)
    qml.capture.disable()


test_pauli_rot_with_adjoint_single_gate()
