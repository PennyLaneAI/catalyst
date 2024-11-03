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

from catalyst import pipeline, qjit
from catalyst.passes import cancel_inverses, merge_rotations

# pylint: disable=missing-function-docstring

#
# Complex_merging_rotations
#

# Parameterize test with different angle sets for qml.Rot and qml.CRot to ensure coverage of complex cases.
@pytest.mark.parametrize("params1, params2", [
    ((0.5, 1.0, 1.5), (0.6, 0.8, 0.7)),  # Arbitrary angles for general coverage
    ((np.pi / 2, np.pi / 4, np.pi / 6), (np.pi, 3 * np.pi / 4, np.pi / 3))  # Important angles with multiples of π
])
def test_complex_merge_rotation(params1, params2, backend):
    """Comprehensive test for complex merge rotations with qml.Rot and qml.CRot using full-angle formulas."""

    # Test for qml.Rot
    @qjit
    def rot_workflow():
        @qml.qnode(qml.device(backend, wires=1))
        def f():
            qml.Rot(params1[0], params1[1], params1[2], wires=0)
            qml.Rot(params2[0], params2[1], params2[2], wires=0)
            return qml.probs()

        @merge_rotations
        @qml.qnode(qml.device(backend, wires=1))
        def g():
            qml.Rot(params1[0], params1[1], params1[2], wires=0)
            qml.Rot(params2[0], params2[1], params2[2], wires=0)
            return qml.probs()

        return f(), g()

    # Reference function for qml.Rot without merging
    @qml.qnode(qml.device("default.qubit", wires=1))
    def rot_reference():
        qml.Rot(params1[0], params1[1], params1[2], wires=0)
        qml.Rot(params2[0], params2[1], params2[2], wires=0)
        return qml.probs()

    # Verify results for qml.Rot
    rot_results = rot_workflow()
    assert np.allclose(rot_results[0], rot_results[1]), "Merged result for qml.Rot differs from unmerged."
    assert np.allclose(rot_results[1], rot_reference()), "Merged result for qml.Rot differs from reference."

    # Test for qml.CRot
    @qjit
    def crot_workflow():
        @qml.qnode(qml.device(backend, wires=2))
        def f():
            qml.CRot(params1[0], params1[1], params1[2], wires=[0, 1])
            qml.CRot(params2[0], params2[1], params2[2], wires=[0, 1])
            return qml.probs()

        @merge_rotations
        @qml.qnode(qml.device(backend, wires=2))
        def g():
            qml.CRot(params1[0], params1[1], params1[2], wires=[0, 1])
            qml.CRot(params2[0], params2[1], params2[2], wires=[0, 1])
            return qml.probs()

        return f(), g()

    # Reference function for qml.CRot without merging
    @qml.qnode(qml.device("default.qubit", wires=2))
    def crot_reference():
        qml.CRot(params1[0], params1[1], params1[2], wires=[0, 1])
        qml.CRot(params2[0], params2[1], params2[2], wires=[0, 1])
        return qml.probs()

    # Verify results for qml.CRot
    crot_results = crot_workflow()
    assert np.allclose(crot_results[0], crot_results[1]), "Merged result for qml.CRot differs from unmerged."
    assert np.allclose(crot_results[1], crot_reference()), "Merged result for qml.CRot differs from reference."

#
# cancel_inverses
#


### Test peephole pass decorators preserve functionality of circuits ###
@pytest.mark.parametrize("theta", [42.42])
def test_cancel_inverses_functionality(theta, backend):

    @qjit
    def workflow():
        @qml.qnode(qml.device(backend, wires=1))
        def f(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.probs()

        @cancel_inverses
        @qml.qnode(qml.device(backend, wires=1))
        def g(x):
            qml.RX(x, wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.probs()

        return f(theta), g(theta)

    @qml.qnode(qml.device("default.qubit", wires=1))
    def reference(x):
        qml.RX(x, wires=0)
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=0)
        return qml.probs()

    assert np.allclose(workflow()[0], workflow()[1])
    assert np.allclose(workflow()[1], reference(theta))


@pytest.mark.parametrize("theta", [42.42])
def test_merge_rotation_functionality(theta, backend):

    @qjit
    def workflow():
        @qml.qnode(qml.device(backend, wires=1))
        def f(x):
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

        @merge_rotations
        @qml.qnode(qml.device(backend, wires=1))
        def g(x):
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

        return f(theta), g(theta)

    @qml.qnode(qml.device("default.qubit", wires=1))
    def reference(x):
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

    assert np.allclose(workflow()[0], workflow()[1])
    assert np.allclose(workflow()[1], reference(theta))


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
            pipeline()(classical_func)

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


if __name__ == "__main__":
    pytest.main(["-x", __file__])
