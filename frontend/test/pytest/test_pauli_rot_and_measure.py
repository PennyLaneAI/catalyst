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

"""Test Pauli rotation and measurement lowering"""

import numpy as np
import pennylane as qml
import pytest
from pennylane.ftqc.catalyst_pass_aliases import to_ppr

from catalyst import qjit


def test_pauli_rot_lowering(capture_mode):
    """Test that Pauli rotation is lowered to quantum.paulirot."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(capture=capture_mode, pipelines=pipe, target="mlir")
    def test_pauli_rot_lowering_workflow():

        @qml.qnode(qml.device("null.qubit", wires=1))
        def f():
            qml.PauliRot(np.pi / 4, "X", wires=0)

        return f()

    optimized_ir = test_pauli_rot_lowering_workflow.mlir_opt
    assert "quantum.paulirot" in optimized_ir


def test_pauli_rot_lowering_with_ctrl_qubits(capture_mode):
    """Test that Pauli rotation with control qubits is converted to pbc.ppr.
    Note that control PauliRot is currently not supported by the to_ppr pass.
    """
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(capture=capture_mode, pipelines=pipe, target="mlir")
    def test_pauli_rot_lowering_with_ctrl_qubits_workflow():

        @qml.qnode(qml.device("null.qubit", wires=2))
        def f():
            qml.ctrl(qml.PauliRot(np.pi / 4, "X", wires=0), control=1)

        return f()

    optimized_ir = test_pauli_rot_lowering_with_ctrl_qubits_workflow.mlir_opt
    assert "quantum.paulirot" in optimized_ir
    assert "ctrls" in optimized_ir


def test_pauli_rot_to_ppr(capture_mode):
    """Test that Pauli rotation is converted to pbc.ppr."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(capture=capture_mode, pipelines=pipe, target="mlir")
    @to_ppr
    def test_pauli_rot_to_ppr_workflow():

        @qml.qnode(qml.device("null.qubit", wires=1))
        def f():
            qml.PauliRot(np.pi / 4, "X", wires=0)

        return f()

    optimized_ir = test_pauli_rot_to_ppr_workflow.mlir_opt
    assert "pbc.ppr" in optimized_ir


def test_pauli_rot_with_arbitrary_angle_to_ppr(capture_mode):
    """Test that Pauli rotation for arbitrary angle."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(capture=capture_mode, pipelines=pipe, target="mlir")
    @to_ppr
    def test_pauli_rot_with_arbitrary_angle_to_ppr_workflow():

        @qml.qnode(qml.device("null.qubit", wires=1))
        def f():
            qml.PauliRot(0.42, "X", wires=0)

        return f()

    optimized_ir = test_pauli_rot_with_arbitrary_angle_to_ppr_workflow.mlir_opt
    assert "pbc.ppr.arbitrary" in optimized_ir


def test_pauli_rot_with_dynamic_angle_to_ppr(capture_mode):
    """Test that Pauli rotation for dynamic angle."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(capture=capture_mode, pipelines=pipe, target="mlir")
    @to_ppr
    def test_pauli_rot_with_dynamic_angle_to_ppr_workflow():

        @qml.qnode(qml.device("null.qubit", wires=1))
        def f(x: float):
            qml.PauliRot(x, "X", wires=0)

        return f(0.42)

    optimized_ir = test_pauli_rot_with_dynamic_angle_to_ppr_workflow.mlir_opt
    assert "pbc.ppr.arbitrary" in optimized_ir


def test_pauli_measure_to_ppm(capture_mode):
    """Test that Pauli measurement is converted to pbc.ppm."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(capture=capture_mode, pipelines=pipe, target="mlir")
    @to_ppr
    def test_pauli_measure_to_ppr_workflow():

        @qml.qnode(qml.device("null.qubit", wires=1))
        def f():
            qml.pauli_measure("X", wires=0)

        return f()

    optimized_ir = test_pauli_measure_to_ppr_workflow.mlir_opt
    assert "pbc.ppm" in optimized_ir


def test_pauli_rot_to_ppr_pauli_word_error(capture_mode):
    """Test that unsupported pauli words raises `ValueError`."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    with pytest.raises(
        ValueError,
        match=r"The given Pauli word \"A\" contains characters that are not allowed. "
        r"Allowed characters are I, X, Y and Z",
    ):

        @qjit(capture=capture_mode, pipelines=pipe, target="mlir")
        def test_pauli_rot_to_ppr_pauli_word_error_workflow():

            @qml.qnode(qml.device("null.qubit", wires=1))
            def f():
                qml.PauliRot(np.pi / 4, "A", wires=0)

            return f()


def test_pauli_measure_to_ppr_pauli_word_error(capture_mode):
    """Test that unsupported pauli words raises `ValueError`."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    with pytest.raises(
        ValueError,
        match=r"Only Pauli words consisting of 'I', 'X', 'Y', and 'Z' are allowed.",
    ):

        @qjit(capture=capture_mode, pipelines=pipe, target="mlir")
        def test_pauli_measure_to_ppr_pauli_word_error_workflow():

            @qml.qnode(qml.device("null.qubit", wires=1))
            def f():
                qml.pauli_measure("A", wires=0)

            return f()
