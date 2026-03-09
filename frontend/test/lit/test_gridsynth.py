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

"""
Unit tests for the gridsynth decomposition pass.
"""

# RUN: %PYTHON %s | FileCheck %s
# pylint: disable=line-too-long

from functools import partial

import pennylane as qml

from catalyst import qjit
from catalyst.passes import gridsynth

# Pipeline to stop after quantum compilation (where gridsynth runs)
# This prevents lowerings that might fail for pbc.ppr.
pipe = [("pipe", ["quantum-compilation-stage"])]

# ==============================================================================
# Test 1: RZ Registration (Clifford+T basis)
# ==============================================================================


def test_rz_registration():
    """Test that the gridsynth pass is correctly registered for RZ."""

    @qjit(target="mlir")
    @gridsynth(epsilon=0.01)
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit(x: float):
        qml.RZ(x, wires=0)
        return qml.probs()

    # CHECK: transform.named_sequence @__transform_main
    # CHECK: transform.apply_registered_pass "gridsynth" with options = {"epsilon" = 1.000000e-02 : f64, "ppr-basis" = false}
    # CHECK-LABEL: func.func public @circuit
    # CHECK: quantum.custom "RZ"
    print(circuit.mlir)


test_rz_registration()

# ==============================================================================
# Test 2: RZ Lowering (Clifford+T basis)
# ==============================================================================


def test_rz_lowering():
    """Test that RZ is correctly lowered to the decomposition function."""

    @qjit(target="mlir", pipelines=pipe)
    @gridsynth(epsilon=0.01)
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit(x: float):
        qml.RZ(x, wires=0)
        return qml.probs()

    # CHECK-DAG:   func.func private @rs_decomposition_get_gates(memref<?xindex>, f64, f64, i1)
    # CHECK-DAG:   func.func private @rs_decomposition_get_phase(f64, f64, i1) -> f64
    # CHECK-DAG:   func.func private @rs_decomposition_get_size(f64, f64, i1) -> index

    # CHECK-LABEL: func.func private @__catalyst_decompose_RZ{{.*}}
    # CHECK:       scf.index_switch
    # CHECK:       case 0 {
    # CHECK:         quantum.custom "T"
    # CHECK:       }

    # CHECK-LABEL: func.func public @circuit{{.*}}
    # CHECK-NOT:   quantum.custom "RZ"
    # CHECK:       call @__catalyst_decompose_RZ{{.*}}
    print(circuit.mlir_opt)


test_rz_lowering()

# ==============================================================================
# Test 3: PhaseShift Registration
# ==============================================================================


def test_phaseshift_registration():
    """Test that the gridsynth pass is correctly registered for PhaseShift."""

    @qjit(target="mlir")
    @gridsynth(epsilon=0.01)
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit(x: float):
        qml.PhaseShift(x, wires=0)
        return qml.probs()

    # CHECK:       transform.apply_registered_pass "gridsynth"
    # CHECK-LABEL: func.func public @circuit
    # CHECK:       quantum.custom "PhaseShift"
    print(circuit.mlir)


test_phaseshift_registration()

# ==============================================================================
# Test 4: PhaseShift Lowering
# ==============================================================================


def test_phaseshift_lowering():
    """Test that PhaseShift is decomposed into RZ + GlobalPhase."""

    @qjit(target="mlir", pipelines=pipe)
    @gridsynth(epsilon=0.01)
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit(x: float):
        qml.PhaseShift(x, wires=0)
        return qml.probs()

    # CHECK-DAG:   func.func private @rs_decomposition_get_gates(memref<?xindex>, f64, f64, i1)
    # CHECK-DAG:   func.func private @rs_decomposition_get_phase(f64, f64, i1) -> f64
    # CHECK-DAG:   func.func private @rs_decomposition_get_size(f64, f64, i1) -> index

    # CHECK-LABEL: func.func private @__catalyst_decompose_RZ{{.*}}

    # CHECK-LABEL: func.func public @circuit{{.*}}
    # CHECK:       call @__catalyst_decompose_RZ{{.*}}
    # CHECK:       quantum.gphase
    print(circuit.mlir_opt)


test_phaseshift_lowering()

# ==============================================================================
# Test 5: PPR Registration
# ==============================================================================


def test_ppr_registration():
    """Test that ppr_basis=True is passed to the transform."""

    @qjit(target="mlir")
    @gridsynth(epsilon=0.01, ppr_basis=True)
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit(x: float):
        qml.RZ(x, wires=0)
        return qml.probs()

    # CHECK: transform.apply_registered_pass "gridsynth" with options = {"epsilon" = 1.000000e-02 : f64, "ppr-basis" = true}
    print(circuit.mlir)


test_ppr_registration()

# ==============================================================================
# Test 6: PPR Lowering
# ==============================================================================


def test_ppr_lowering():
    """Test that PPR basis generates pbc.ppr operations."""

    @qjit(target="mlir", pipelines=pipe)
    @gridsynth(epsilon=0.01, ppr_basis=True)
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit(x: float):
        qml.RZ(x, wires=0)
        return qml.probs()

    # CHECK-DAG:   func.func private @rs_decomposition_get_gates(memref<?xindex>, f64, f64, i1)
    # CHECK-DAG:   func.func private @rs_decomposition_get_phase(f64, f64, i1) -> f64
    # CHECK-DAG:   func.func private @rs_decomposition_get_size(f64, f64, i1) -> index

    # CHECK-LABEL: func.func private @__catalyst_decompose_RZ_ppr_basis{{.*}}
    # CHECK:       scf.index_switch
    # CHECK:       case 1 {
    # CHECK:         pbc.ppr ["X"](2)
    # CHECK:       }

    # CHECK-LABEL: func.func public @circuit{{.*}}
    # CHECK:       call @__catalyst_decompose_RZ_ppr_basis{{.*}}
    print(circuit.mlir_opt)


test_ppr_lowering()


# ==============================================================================
# Test 7: Capture Workflow Lowering (Clifford+T)
# ==============================================================================


def test_capture_workflow_clifford():
    """Test the capture workflow with qml.transforms.gridsynth (Clifford+T)."""
    qml.capture.enable()

    @qjit(target="mlir", pipelines=pipe)
    @partial(qml.transforms.gridsynth, epsilon=0.01, ppr_basis=False)
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit(x: float):
        qml.RZ(x, wires=0)
        return qml.probs()

    # CHECK-DAG:   func.func private @rs_decomposition_get_gates(memref<?xindex>, f64, f64, i1)
    # CHECK-DAG:   func.func private @rs_decomposition_get_phase(f64, f64, i1) -> f64
    # CHECK-DAG:   func.func private @rs_decomposition_get_size(f64, f64, i1) -> index

    # CHECK-LABEL: func.func private @__catalyst_decompose_RZ{{.*}}
    # CHECK:       scf.index_switch
    # CHECK:       case 0 {
    # CHECK:         quantum.custom "T"
    # CHECK:       }

    # CHECK-LABEL: func.func public @circuit{{.*}}
    # CHECK-NOT:   quantum.custom "RZ"
    # CHECK:       call @__catalyst_decompose_RZ{{.*}}
    print(circuit.mlir_opt)

    qml.capture.disable()


test_capture_workflow_clifford()

# ==============================================================================
# Test 8: Capture Workflow Lowering (PPR)
# ==============================================================================


def test_capture_workflow_ppr():
    """Test the capture workflow with qml.transforms.gridsynth (PPR)."""
    qml.capture.enable()

    @qjit(target="mlir", pipelines=pipe)
    @partial(qml.transforms.gridsynth, epsilon=0.01, ppr_basis=True)
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def circuit(x: float):
        qml.RZ(x, wires=0)
        return qml.probs()

    # CHECK-DAG:   func.func private @rs_decomposition_get_gates(memref<?xindex>, f64, f64, i1)
    # CHECK-DAG:   func.func private @rs_decomposition_get_phase(f64, f64, i1) -> f64
    # CHECK-DAG:   func.func private @rs_decomposition_get_size(f64, f64, i1) -> index

    # CHECK-LABEL: func.func private @__catalyst_decompose_RZ_ppr_basis{{.*}}
    # CHECK:       scf.index_switch
    # CHECK:       case 1 {
    # CHECK:         pbc.ppr ["X"](2)
    # CHECK:       }

    # CHECK-LABEL: func.func public @circuit{{.*}}
    # CHECK:       call @__catalyst_decompose_RZ_ppr_basis{{.*}}
    print(circuit.mlir_opt)

    qml.capture.disable()


test_capture_workflow_ppr()
