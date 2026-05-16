# Copyright 2025-2026 Xanadu Quantum Technologies Inc.

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
import pennylane as qp
import pytest
from pennylane.transforms import to_ppr

from catalyst import qjit


def test_pauli_rot_lowering():
    """Test that Pauli rotation is lowered to quantum.paulirot."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipe, target="mlir", capture=True)
    def test_pauli_rot_lowering_workflow():

        @qp.qnode(qp.device("null.qubit", wires=1))
        def f():
            qp.PauliRot(np.pi / 4, "X", wires=0)

        return f()

    optimized_ir = test_pauli_rot_lowering_workflow.mlir_opt
    assert "quantum.paulirot" in optimized_ir


def test_pauli_rot_lowering_with_ctrl_qubits():
    """Test that Pauli rotation with control qubits is converted to quantum.paulirot.
    Note that control PauliRot is currently not supported by the to_ppr pass.
    """
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipe, target="mlir", capture=True)
    def test_pauli_rot_lowering_with_ctrl_qubits_workflow():

        @qp.qnode(qp.device("null.qubit", wires=2))
        def f():
            qp.ctrl(qp.PauliRot(np.pi / 4, "X", wires=0), control=1)

        return f()

    optimized_ir = test_pauli_rot_lowering_with_ctrl_qubits_workflow.mlir_opt
    assert "quantum.paulirot" in optimized_ir
    assert "ctrls" in optimized_ir


def test_pauli_rot_to_ppr():
    """Test that Pauli rotation is converted to pbc.ppr."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipe, target="mlir", capture=True)
    @to_ppr
    def test_pauli_rot_to_ppr_workflow():

        @qp.qnode(qp.device("null.qubit", wires=1))
        def f():
            qp.PauliRot(np.pi / 4, "X", wires=0)

        return f()

    optimized_ir = test_pauli_rot_to_ppr_workflow.mlir_opt
    assert "pbc.ppr" in optimized_ir


def test_pauli_rot_with_arbitrary_angle_to_ppr():
    """Test that Pauli rotation for arbitrary angle."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipe, target="mlir", capture=True)
    @to_ppr
    def test_pauli_rot_with_arbitrary_angle_to_ppr_workflow():

        @qp.qnode(qp.device("null.qubit", wires=1))
        def f():
            qp.PauliRot(0.42, "X", wires=0)

        return f()

    optimized_ir = test_pauli_rot_with_arbitrary_angle_to_ppr_workflow.mlir_opt
    assert "pbc.ppr.arbitrary" in optimized_ir


def test_pauli_rot_with_dynamic_angle_to_ppr():
    """Test that Pauli rotation for dynamic angle."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipe, target="mlir", capture=True)
    @to_ppr
    def test_pauli_rot_with_dynamic_angle_to_ppr_workflow():

        @qp.qnode(qp.device("null.qubit", wires=1))
        def f(x: float):
            qp.PauliRot(x, "X", wires=0)

        return f(0.42)

    optimized_ir = test_pauli_rot_with_dynamic_angle_to_ppr_workflow.mlir_opt
    assert "pbc.ppr.arbitrary" in optimized_ir


def test_pauli_measure_to_ppm():
    """Test that Pauli measurement is converted to pbc.ppm."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipe, target="mlir", capture=True)
    @to_ppr
    def test_pauli_measure_to_ppr_workflow():

        @qp.qnode(qp.device("null.qubit", wires=1))
        def f():
            qp.pauli_measure("X", wires=0)

        return f()

    optimized_ir = test_pauli_measure_to_ppr_workflow.mlir_opt
    assert "pbc.ppm" in optimized_ir


def test_pauli_rot_to_ppr_pauli_word_error():
    """Test that unsupported pauli words raises `ValueError`."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    with pytest.raises(
        ValueError,
        match=r"The given Pauli word \"A\" contains characters that are not allowed. "
        r"Allowed characters are I, X, Y and Z",
    ):

        @qjit(pipelines=pipe, target="mlir", capture=True)
        def test_pauli_rot_to_ppr_pauli_word_error_workflow():

            @qp.qnode(qp.device("null.qubit", wires=1))
            def f():
                qp.PauliRot(np.pi / 4, "A", wires=0)

            return f()


def test_pauli_measure_to_ppr_pauli_word_error():
    """Test that unsupported pauli words raises `ValueError`."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    with pytest.raises(
        ValueError,
        match=r"Only Pauli words consisting of 'I', 'X', 'Y', and 'Z' are allowed.",
    ):

        @qjit(pipelines=pipe, target="mlir", capture=True)
        def test_pauli_measure_to_ppr_pauli_word_error_workflow():

            @qp.qnode(qp.device("null.qubit", wires=1))
            def f():
                qp.pauli_measure("A", wires=0)

            return f()


def test_controlled_pauli_rot_failure():
    """
    Test that controlled PauliRot fails at runtime.
    """

    @qjit(capture=True)
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def workflow():
        qp.ctrl(qp.PauliRot(np.pi / 4, "X", wires=0), control=1)
        return qp.probs()

    with pytest.raises(RuntimeError, match="Controlled PauliRot is not supported"):
        workflow()


def test_legacy_pauli_rot_lowering():
    """Test that Pauli rotation is lowered to quantum.paulirot."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipe, target="mlir", capture=False)
    def test_legacy_pauli_rot_lowering_workflow():

        @qp.qnode(qp.device("null.qubit", wires=1))
        def f():
            qp.PauliRot(pauli_word="Z", wires=[0], theta=np.pi)
            return qp.probs()

        return f()

    optimized_ir = test_legacy_pauli_rot_lowering_workflow.mlir_opt
    assert "quantum.paulirot" in optimized_ir


def test_legacy_controlled_pauli_rot_lowering():
    """Test that controlled Pauli rotation is lowered to quantum.paulirot."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipe, target="mlir", capture=False)
    def test_legacy_controlled_pauli_rot_lowering_workflow():

        @qp.qnode(qp.device("null.qubit", wires=1))
        def f():
            qp.ctrl(qp.PauliRot(pauli_word="Z", wires=[0], theta=np.pi), control=1)
            return qp.state()

        return f()

    optimized_ir = test_legacy_controlled_pauli_rot_lowering_workflow.mlir_opt
    assert "quantum.paulirot" in optimized_ir
    assert "ctrls" in optimized_ir


def test_legacy_pauli_measure_lowering():
    """Test that Pauli measurement is lowered to pbc.ppm."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipe, target="mlir", capture=False)
    def test_legacy_pauli_measure_lowering_workflow():
        import catalyst

        @qp.qnode(qp.device("null.qubit", wires=2))
        def f():
            catalyst.pauli_measure(wires=[0], pauli_word="X")
            return qp.state()

        return f()

    optimized_ir = test_legacy_pauli_measure_lowering_workflow.mlir_opt
    assert "pbc.ppm" in optimized_ir


def test_legacy_cond_pauli_measure_result():
    """Test conditional on Pauli measurement result is lowered properly."""
    pipe = [("pipe", ["quantum-compilation-stage"])]

    @qjit(pipelines=pipe, target="mlir", autograph=True, capture=False)
    def test_legacy_cond_pauli_measure_result_workflow():
        import catalyst

        @qp.qnode(qp.device("null.qubit", wires=3))
        def f():
            m = catalyst.pauli_measure(wires=[0], pauli_word="X")
            if m:
                qp.PauliRot(np.pi / 4, "XX", wires=[0, 1])
            else:
                qp.PauliRot(np.pi / 4, "YY", wires=[0, 1])
            return qp.probs()

        return f()

    optimized_ir = test_legacy_cond_pauli_measure_result_workflow.mlir_opt
    assert "pbc.ppm" in optimized_ir
    assert "scf.if" in optimized_ir
    assert 'quantum.paulirot ["X", "X"]' in optimized_ir
    assert "else" in optimized_ir
    assert 'quantum.paulirot ["Y", "Y"]' in optimized_ir
