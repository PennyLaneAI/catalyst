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

"""Test the quantum peephole passes"""

import numpy as np
import pennylane as qml
import pytest

from catalyst import measure, pipeline, qjit
from catalyst.passes import (
    cancel_inverses,
    commute_ppr,
    merge_ppr_ppm,
    merge_rotations,
    ppm_compilation,
    ppr_to_ppm,
    to_ppr,
)

# pylint: disable=missing-function-docstring


### Test peephole pass decorators preserve functionality of circuits ###
@pytest.mark.parametrize("theta", [42.42])
def test_cancel_inverses_functionality(theta, backend):

    def circuit(x):
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=0)
        return qml.probs()

    reference_workflow = qml.QNode(circuit, qml.device("default.qubit", wires=1))

    customized_device = qml.device(backend, wires=1)
    qjitted_workflow = qjit(qml.QNode(circuit, customized_device))
    optimized_workflow = qjit(cancel_inverses(qml.QNode(circuit, customized_device)))

    assert np.allclose(reference_workflow(theta), qjitted_workflow(theta))
    assert np.allclose(reference_workflow(theta), optimized_workflow(theta))


@pytest.mark.parametrize("theta", [42.42])
def test_merge_rotation_functionality(theta, backend):

    def circuit(x):
        qml.RX(x, wires=0)
        qml.RX(x, wires=0)
        qml.RZ(x, wires=0)
        qml.adjoint(qml.RZ)(x, wires=0)
        qml.Rot(x, x, x, wires=0)
        qml.Rot(x, x, x, wires=0)
        qml.PhaseShift(x, wires=0)
        qml.PhaseShift(x, wires=0)
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=0)
        return qml.probs()

    reference_workflow = qml.QNode(circuit, qml.device("default.qubit", wires=1))

    customized_device = qml.device(backend, wires=1)
    qjitted_workflow = qjit(qml.QNode(circuit, customized_device))
    optimized_workflow = qjit(merge_rotations(qml.QNode(circuit, customized_device)))

    assert np.allclose(reference_workflow(theta), qjitted_workflow(theta))
    assert np.allclose(reference_workflow(theta), optimized_workflow(theta))


@pytest.mark.parametrize("theta", [42.42])
def test_cancel_inverses_functionality_outside_qjit(theta, backend):

    @cancel_inverses
    @qml.qnode(qml.device(backend, wires=1))
    def f(x):
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=0)
        return qml.probs()

    @qjit
    def workflow():
        @cancel_inverses
        @qml.qnode(qml.device(backend, wires=1))
        def g(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.probs()

        _f = f(theta)
        _g = g(theta)
        return _f, _g

    assert np.allclose(workflow()[0], workflow()[1])


@pytest.mark.parametrize("theta", [42.42])
def test_pipeline_functionality(theta, backend):
    """
    Test that the @pipeline decorator does not change functionality
    when all the passes in the pipeline does not change functionality.
    """
    my_pipeline = {
        "cancel_inverses": {},
        "merge_rotations": {},
    }

    @qjit
    def workflow():
        @qml.qnode(qml.device(backend, wires=2))
        def f(x):
            qml.RX(0.1, wires=[0])
            qml.RX(x, wires=[0])
            qml.Hadamard(wires=[1])
            qml.Hadamard(wires=[1])
            return qml.expval(qml.PauliY(wires=0))

        no_pipeline_result = f(theta)
        pipeline_result = pipeline(my_pipeline)(f)(theta)

        return no_pipeline_result, pipeline_result

    res = workflow()
    assert np.allclose(res[0], res[1])


### Test bad usages of pass decorators ###
def test_cancel_inverses_bad_usages():
    """
    Tests that an error is raised when cancel_inverses is not used properly
    """

    def test_cancel_inverses_not_on_qnode():
        def classical_func():
            return 42.42

        with pytest.raises(
            TypeError,
            match="A QNode is expected, got the classical function",
        ):
            pipeline({})(classical_func)

        with pytest.raises(
            TypeError,
            match="A QNode is expected, got the classical function",
        ):
            cancel_inverses(classical_func)

        with pytest.raises(
            TypeError,
            match="A QNode is expected, got the classical function",
        ):
            merge_rotations(classical_func)

    test_cancel_inverses_not_on_qnode()


def test_chained_passes():
    """
    Test that chained passes are present in the transform passes.
    """

    @qjit()
    @merge_rotations
    @cancel_inverses
    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def test_chained_apply_passes_workflow(x: float):
        qml.Hadamard(wires=[1])
        qml.RX(x, wires=[0])
        qml.RX(-x, wires=[0])
        qml.Hadamard(wires=[1])
        return qml.expval(qml.PauliY(wires=0))

    assert "remove-chained-self-inverse" in test_chained_apply_passes_workflow.mlir
    assert "merge-rotations" in test_chained_apply_passes_workflow.mlir


def test_convert_clifford_to_ppr():

    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipe, target="mlir")
    def test_convert_clifford_to_ppr_workflow():

        @to_ppr
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f():
            qml.H(0)
            qml.S(1)
            qml.T(0)
            qml.CNOT([0, 1])

        return f()

    assert 'transform.apply_registered_pass "to_ppr"' in test_convert_clifford_to_ppr_workflow.mlir
    optimized_ir = test_convert_clifford_to_ppr_workflow.mlir_opt
    assert 'transform.apply_registered_pass "to_ppr"' not in optimized_ir
    assert "qec.ppr" in optimized_ir


def test_commute_ppr():

    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipe, target="mlir")
    def test_commute_ppr_workflow():

        @commute_ppr
        @to_ppr
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f():
            qml.H(0)
            qml.S(1)
            qml.T(0)
            qml.CNOT([0, 1])
            return measure(0), measure(1)

        return f()

    assert 'transform.apply_registered_pass "commute_ppr"' in test_commute_ppr_workflow.mlir
    optimized_ir = test_commute_ppr_workflow.mlir_opt
    assert 'transform.apply_registered_pass "commute_ppr"' not in optimized_ir
    assert "qec.ppr" in optimized_ir
    assert "qec.ppm" in optimized_ir


def test_merge_ppr_ppm():

    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipe, target="mlir")
    def test_merge_ppr_ppm_workflow():

        @merge_ppr_ppm
        @to_ppr
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f():
            qml.H(0)
            qml.S(1)
            qml.CNOT([0, 1])
            return measure(0), measure(1)

        return f()

    assert 'transform.apply_registered_pass "merge_ppr_ppm"' in test_merge_ppr_ppm_workflow.mlir
    optimized_ir = test_merge_ppr_ppm_workflow.mlir_opt
    assert 'transform.apply_registered_pass "merge_ppr_ppm"' not in optimized_ir
    assert 'qec.ppm ["Z", "X"]' in optimized_ir
    assert 'qec.ppm ["X"]' in optimized_ir


def test_ppr_to_ppm():

    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipe, target="mlir")
    def test_ppr_to_ppm_workflow():

        @ppr_to_ppm
        @to_ppr
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f():
            qml.H(0)
            qml.S(1)
            qml.T(0)
            qml.CNOT([0, 1])
            return measure(0), measure(1)

        return f()

    assert (
        'transform.apply_registered_pass "decompose_non_clifford_ppr"'
        in test_ppr_to_ppm_workflow.mlir
    )
    assert (
        'transform.apply_registered_pass "decompose_clifford_ppr"' in test_ppr_to_ppm_workflow.mlir
    )
    optimized_ir = test_ppr_to_ppm_workflow.mlir_opt
    assert 'transform.apply_registered_pass "decompose_non_clifford_ppr"' not in optimized_ir
    assert 'transform.apply_registered_pass "decompose_clifford_ppr"' not in optimized_ir
    assert "quantum.alloc_qb" in optimized_ir
    assert "qec.fabricate  magic" in optimized_ir
    assert "qec.select.ppm" in optimized_ir
    assert 'qec.ppr ["X"]' in optimized_ir


def test_ppr_to_ppm_inject_magic_state():

    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipe, target="mlir")
    def test_ppr_to_ppm_workflow():

        @ppr_to_ppm(decompose_method="clifford-corrected", avoid_y_measure=True)
        @to_ppr
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f():
            qml.H(0)
            qml.S(1)
            qml.T(0)
            qml.CNOT([0, 1])
            return measure(0), measure(1)

        return f()

    assert (
        'transform.apply_registered_pass "decompose_non_clifford_ppr"'
        in test_ppr_to_ppm_workflow.mlir
    )
    assert (
        'transform.apply_registered_pass "decompose_clifford_ppr"' in test_ppr_to_ppm_workflow.mlir
    )
    optimized_ir = test_ppr_to_ppm_workflow.mlir_opt
    assert 'transform.apply_registered_pass "decompose_non_clifford_ppr"' not in optimized_ir
    assert 'transform.apply_registered_pass "decompose_clifford_ppr"' not in optimized_ir


def test_commute_ppr_and_merge_ppr_ppm_with_max_pauli_size():

    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipe, target="mlir")
    def test_convert_clifford_to_ppr_workflow():

        device = qml.device("lightning.qubit", wires=2)

        @merge_ppr_ppm
        @commute_ppr(max_pauli_size=2)
        @to_ppr
        @qml.qnode(device)
        def f():
            qml.CNOT([0, 2])
            qml.T(0)
            return measure(0), measure(1)

        @merge_ppr_ppm(max_pauli_size=1)
        @commute_ppr
        @to_ppr
        @qml.qnode(device)
        def g():
            qml.CNOT([0, 2])
            qml.T(0)
            qml.T(1)
            qml.CNOT([0, 1])
            return measure(0), measure(1)

        return f(), g()

    assert (
        'transform.apply_registered_pass "commute_ppr"'
        in test_convert_clifford_to_ppr_workflow.mlir
    )
    assert (
        'transform.apply_registered_pass "merge_ppr_ppm"'
        in test_convert_clifford_to_ppr_workflow.mlir
    )

    optimized_ir = test_convert_clifford_to_ppr_workflow.mlir_opt
    assert 'transform.apply_registered_pass "commute_ppr"' not in optimized_ir
    assert 'transform.apply_registered_pass "merge_ppr_ppm"' not in optimized_ir


def test_clifford_to_ppm():

    pipe = [("pipe", ["enforce-runtime-invariants-pipeline"])]

    @qjit(pipelines=pipe, target="mlir")
    def test_clifford_to_ppm_workflow():

        @ppm_compilation
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def f():
            for idx in range(5):
                qml.H(idx)
                qml.CNOT(wires=[idx, idx + 1])
                qml.T(idx)
                qml.T(idx + 1)
            return measure(0)

        @ppm_compilation(
            decompose_method="clifford-corrected", avoid_y_measure=True, max_pauli_size=2
        )
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def g():
            for idx in range(5):
                qml.H(idx)
                qml.CNOT(wires=[idx, idx + 1])
                qml.T(idx)
                qml.T(idx + 1)

        return f(), g()

    assert 'transform.apply_registered_pass "ppm_compilation"' in test_clifford_to_ppm_workflow.mlir
    optimized_ir = test_clifford_to_ppm_workflow.mlir_opt
    assert 'transform.apply_registered_pass "ppm_compilation"' not in optimized_ir
    assert 'qec.ppm ["X", "Z", "Z"]' in optimized_ir
    assert 'qec.ppm ["Z", "Y"]' in optimized_ir
    assert 'qec.ppr ["X", "Z"](2)' in optimized_ir


if __name__ == "__main__":
    pytest.main(["-x", __file__])
