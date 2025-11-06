import numpy as np
import pennylane as qml
import pytest

from catalyst import qjit


def test_pauli_rot_to_ppr():
    qml.capture.enable()
    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipe, target="mlir")
    def test_pauli_rot_to_ppr_workflow():

        @qml.qnode(qml.device("catalyst.ftqc", wires=1))
        def f():
            qml.PauliRot(np.pi / 4, "X", wires=0)

        return f()

    optimized_ir = test_pauli_rot_to_ppr_workflow.mlir_opt
    assert "qec.ppr" in optimized_ir
    qml.capture.disable()


def test_pauli_measure_to_ppr():
    qml.capture.enable()
    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipe, target="mlir")
    def test_pauli_measure_to_ppr_workflow():

        @qml.qnode(qml.device("catalyst.ftqc", wires=1))
        def f():
            qml.pauli_measure("X", wires=0)

        return f()

    optimized_ir = test_pauli_measure_to_ppr_workflow.mlir_opt
    assert "qec.ppm" in optimized_ir
    qml.capture.disable()


def test_pauli_rot_to_ppr_error():
    qml.capture.enable()
    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    with pytest.raises(
        ValueError,
    ):

        @qjit(pipelines=pipe, target="mlir")
        def test_pauli_rot_to_ppr_error_workflow():

            @qml.qnode(qml.device("catalyst.ftqc", wires=1))
            def f():
                qml.PauliRot(np.pi / 12, "X", wires=0)

            return f()

    qml.capture.disable()
