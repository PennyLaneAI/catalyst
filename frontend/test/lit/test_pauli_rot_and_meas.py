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

from catalyst import qjit
from catalyst.passes import to_ppr, commute_ppr, merge_ppr_ppm, ppr_to_ppm, ppm_compilation


def test_single_qubit_pauli_rotations():
    """Test single qubit Pauli rotations with different angles and Pauli strings."""
    dev = qml.device("catalyst.ftqc", wires=1)
    
    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 4, "X", wires=0)
        qml.PauliRot(np.pi / 2, "Y", wires=0)
        qml.PauliRot(np.pi, "Z", wires=0)
        return

    # CHECK: qec.ppr ["X"](8)
    # CHECK: qec.ppr ["Y"](4)  
    # CHECK: qec.ppr ["Z"](2)
    print(circuit.mlir_opt)

test_single_qubit_pauli_rotations()

def test_multi_qubit_pauli_rotations():
    """Test multi-qubit Pauli rotations with different Pauli strings."""
    dev = qml.device("catalyst.ftqc", wires=3)
    
    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 4, "XY", wires=[0, 1])
        qml.PauliRot(np.pi / 2, "YZ", wires=[1, 2])
        qml.PauliRot(np.pi, "ZX", wires=[2, 0])
        qml.PauliRot(np.pi / 4, "XYZ", wires=[0, 1, 2])
        return

    # CHECK: qec.ppr ["X", "Y"](8)
    # CHECK: qec.ppr ["Y", "Z"](4)
    # CHECK: qec.ppr ["Z", "X"](2)
    # CHECK: qec.ppr ["X", "Y", "Z"](8)
    print(circuit.mlir_opt)

test_multi_qubit_pauli_rotations()

def test_single_qubit_pauli_measurements():
    """Test single qubit Pauli measurements with different Pauli strings."""
    dev = qml.device("catalyst.ftqc", wires=1)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.pauli_measure("X", wires=0)
        qml.pauli_measure("Y", wires=0)
        qml.pauli_measure("Z", wires=0)
        return

    # CHECK: qec.ppm ["X"]
    # CHECK: qec.ppm ["Y"]
    # CHECK: qec.ppm ["Z"]
    print(circuit.mlir_opt)

test_single_qubit_pauli_measurements()

def test_multi_qubit_pauli_measurements():
    """Test multi-qubit Pauli measurements with different Pauli strings."""
    dev = qml.device("catalyst.ftqc", wires=3)

    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)
        qml.pauli_measure("XY", wires=[0, 1])
        qml.pauli_measure("ZX", wires=[1, 2])
        qml.pauli_measure("XYZ", wires=[0, 1, 2])
        return

    # CHECK: qec.ppm ["X", "Y"]
    # CHECK: qec.ppm ["Z", "X"]
    # CHECK: qec.ppm ["X", "Y", "Z"]
    print(circuit.mlir_opt)

test_multi_qubit_pauli_measurements()


def test_pauli_rot_and_measure_combined():
    """Test combining Pauli rotations and measurements in one circuit."""
    dev = qml.device("catalyst.ftqc", wires=2)
    
    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

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
        return

    # CHECK: qec.ppr ["X"](8)
    # CHECK: qec.ppr ["Y"](4)
    # CHECK: qec.ppr ["X", "Y"](8)
    # CHECK: qec.ppm ["X"]
    # CHECK: qec.ppm ["Y"]
    # CHECK: qec.ppm ["X", "Y"]
    print(circuit.mlir_opt)

test_pauli_rot_and_measure_combined()


def test_clifford_t_ppr_ppm_combined():
    """Test to-ppr pass with Clifford+T gates, PPR gates, and PPM gates."""
    dev = qml.device("catalyst.ftqc", wires=3)
    
    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.T(wires=0)
        qml.PauliRot(np.pi / 4, "X", wires=0)
        qml.pauli_measure("X", wires=0)

        qml.S(wires=1) 
        qml.Hadamard(wires=1)
        qml.T(wires=1)        
        qml.PauliRot(np.pi / 2, "YZ", wires=[1, 2])
        qml.pauli_measure("YZ", wires=[1, 2])
        return

    # Wire 0
    # CHECK: qec.ppr ["Z"](4)
    # CHECK: qec.ppr ["X"](4)
    # CHECK: qec.ppr ["Z"](4)
    # CHECK: qec.ppr ["Z"](8)
    # CHECK: qec.ppr ["X"](8)
    # CHECK: qec.ppm ["X"]

    # Wire 1 and 2
    # CHECK: qec.ppr ["Z"](4)
    # CHECK: qec.ppr ["Z"](4)
    # CHECK: qec.ppr ["X"](4)
    # CHECK: qec.ppr ["Z"](4)
    # CHECK: qec.ppr ["Z"](8)
    # CHECK: qec.ppr ["Y", "Z"](4)
    # CHECK: qec.ppm ["Y", "Z"]
    print(circuit.mlir_opt)
    
test_clifford_t_ppr_ppm_combined()


def test_commute_ppr():
    """Test commute-ppr pass: PauliRot and S gates should be commuted to the back after T gate."""
    dev = qml.device("catalyst.ftqc", wires=1)
    
    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @commute_ppr
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 2, "Z", wires=0)  # Clifford gate
        qml.S(wires=0)                         # Clifford gate
        qml.T(wires=0)                         # Non-Clifford gate
        return

    # CHECK: qec.ppr ["Z"](8)
    # CHECK: qec.ppr ["Z"](4)
    # CHECK: qec.ppr ["Z"](4)
    print(circuit.mlir_opt)

test_commute_ppr()


def test_merge_ppr_ppm():
    """Test merge-ppr-ppm pass: Clifford PauliRot should be merged into PauliMeasure."""
    dev = qml.device("catalyst.ftqc", wires=1)
    
    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @merge_ppr_ppm
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 2, "Z", wires=0)
        qml.pauli_measure("X", wires=0)
        return

    # CHECK: qec.ppm ["Y"]
    print(circuit.mlir_opt)

test_merge_ppr_ppm()


def test_ppr_to_ppm():
    """Test ppr_to_ppm pass: PauliRot should be decomposed into PPM."""
    dev = qml.device("catalyst.ftqc", wires=1)
    
    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @merge_ppr_ppm
    @ppr_to_ppm
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.PauliRot(np.pi / 2, "X", wires=0)
        qml.PauliRot(np.pi / 4, "Y", wires=0)
        return

    # CHECK: qec.ppm ["X", "Y"](-1)
    # CHECK: qec.ppm ["X"]
    # CHECK: qec.ppm ["Y", "Z"](-1)
    print(circuit.mlir_opt)

test_ppr_to_ppm()


def test_ppm_compilation():
    """Test ppm_compilation pass: PauliRot should be decomposed into PPM."""
    dev = qml.device("catalyst.ftqc", wires=1)
    
    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, keep_intermediate=2, target="mlir")
    @ppm_compilation
    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.PauliRot(np.pi / 2, "X", wires=0)
        qml.PauliRot(np.pi / 4, "Y", wires=0)
        qml.T(wires=0)
        qml.pauli_measure("X", wires=0)
        return

    # CHECK: qec.ppm ["X", "Z"]
    # CHECK: qec.select.ppm
    # CHECK: qec.ppr ["X"](2)
    # CHECK: qec.fabricate  magic
    # CHECK: qec.ppm ["Y", "Z"]
    # CHECK: qec.select.ppm
    # CHECK: qec.ppr ["Y"](2)
    # CHECK: qec.ppm ["Z"]
    print(circuit.mlir_opt)

test_ppm_compilation()



def test_with_capture_enabled():
    """Test ppm_compilation pass: PauliRot should be decomposed into PPM."""
    from pennylane.ftqc.catalyst_pass_aliases import commute_ppr, to_ppr
    
    qml.capture.enable()

    dev = qml.device("catalyst.ftqc", wires=1)
    
    pipeline = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipeline, target="mlir")
    @commute_ppr
    @to_ppr
    @qml.qnode(device=dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.PauliRot(np.pi / 2, "X", wires=0)
        qml.PauliRot(np.pi / 4, "Y", wires=0)
        qml.T(wires=0)
        qml.pauli_measure("X", wires=0)
        return

    # CHECK: qec.ppr ["X"](-8)
    # CHECK: qec.ppr ["Y"](-8)
    # CHECK: qec.ppr ["Z"](4)
    # CHECK: qec.ppr ["X"](4)
    # CHECK: qec.ppr ["Z"](4)
    # CHECK: qec.ppr ["X"](4)
    print(circuit.mlir_opt)

test_with_capture_enabled()
