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
# cancel_inverses
#


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


#
# merge_rotations
#


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


if __name__ == "__main__":
    pytest.main(["-x", __file__])
