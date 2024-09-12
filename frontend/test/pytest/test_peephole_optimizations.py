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

from catalyst import qjit
from catalyst.passes import cancel_inverses

# pylint: disable=missing-function-docstring


#
# cancel_inverses
#


class TestHermitianGatesSelfInverse:
    """Tests two consecutive Hermitian (self inverse) Gates get canceled out."""

    ### Test two consecutive Hadamard gates get canceled out ###
    @pytest.mark.parametrize("theta", [42.42])
    def test_Hadamard(self, theta, backend):

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def f(theta):
            qml.RX(theta, wires=0)
            return qml.probs()

        @qjit
        @cancel_inverses
        @qml.qnode(qml.device(backend, wires=1))
        def g(theta):
            qml.RX(theta, wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            return qml.probs()

        expected = f(theta)
        result = g(theta)
        assert np.allclose(result, expected)

    ### Test two consecutive PauliX gates get canceled out ###
    @pytest.mark.parametrize("theta", [42.42])
    def test_PauliX(self, theta, backend):

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def f(theta):
            qml.RX(theta, wires=0)
            return qml.probs()

        @qjit
        @cancel_inverses
        @qml.qnode(qml.device(backend, wires=1))
        def g(theta):
            qml.RX(theta, wires=0)
            qml.PauliX(wires=0)
            qml.PauliX(wires=0)
            return qml.probs()

        expected = f(theta)
        result = g(theta)
        assert np.allclose(result, expected)

    ### Test two consecutive PauliY gates get canceled out ###
    @pytest.mark.parametrize("theta", [42.42])
    def test_PauliY(self, theta, backend):

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def f(theta):
            qml.RX(theta, wires=0)
            return qml.probs()

        @qjit
        @cancel_inverses
        @qml.qnode(qml.device(backend, wires=1))
        def g(theta):
            qml.RX(theta, wires=0)
            qml.PauliY(wires=0)
            qml.PauliY(wires=0)
            return qml.probs()

        expected = f(theta)
        result = g(theta)
        assert np.allclose(result, expected)

    ### Test two consecutive PauliZ gates get canceled out ###
    @pytest.mark.parametrize("theta", [42.42])
    def test_PauliZ(self, theta, backend):

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def f(theta):
            qml.RX(theta, wires=0)
            return qml.probs()

        @qjit
        @cancel_inverses
        @qml.qnode(qml.device(backend, wires=1))
        def g(theta):
            qml.RX(theta, wires=0)
            qml.PauliZ(wires=0)
            qml.PauliZ(wires=0)
            return qml.probs()

        expected = f(theta)
        result = g(theta)
        assert np.allclose(result, expected)

    ### Test two consecutive CNOT gates get canceled out ###
    @pytest.mark.parametrize("theta", [42.42])
    def test_CNOT(self, theta, backend):

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def f(theta):
            qml.RX(theta, wires=0)
            qml.RX(theta, wires=1)
            return qml.probs()

        @qjit
        @cancel_inverses
        @qml.qnode(qml.device(backend, wires=2))
        def g(theta):
            qml.RX(theta, wires=0)
            qml.RX(theta, wires=1)
            qml.CNOT(wires=(0, 1))
            qml.CNOT(wires=(0, 1))
            return qml.probs()

        expected = f(theta)
        result = g(theta)
        assert np.allclose(result, expected)

    ### Test two consecutive CY gates get canceled out ###
    @pytest.mark.parametrize("theta", [42.42])
    def test_CY(self, theta, backend):

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def f(theta):
            qml.RX(theta, wires=0)
            qml.RX(theta, wires=1)
            return qml.probs()

        @qjit
        @cancel_inverses
        @qml.qnode(qml.device(backend, wires=2))
        def g(theta):
            qml.RX(theta, wires=0)
            qml.RX(theta, wires=1)
            qml.CY(wires=(0, 1))
            qml.CY(wires=(0, 1))
            return qml.probs()

        expected = f(theta)
        result = g(theta)
        assert np.allclose(result, expected)

    ### Test two consecutive CZ gates get canceled out ###
    @pytest.mark.parametrize("theta", [42.42])
    def test_CZ(self, theta, backend):

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def f(theta):
            qml.RX(theta, wires=0)
            qml.RX(theta, wires=1)
            return qml.probs()

        @qjit
        @cancel_inverses
        @qml.qnode(qml.device(backend, wires=2))
        def g(theta):
            qml.RX(theta, wires=0)
            qml.RX(theta, wires=1)
            qml.CZ(wires=(0, 1))
            qml.CZ(wires=(0, 1))
            return qml.probs()

        expected = f(theta)
        result = g(theta)
        assert np.allclose(result, expected)

    ### Test two consecutive SWAP gates get canceled out ###
    @pytest.mark.parametrize("theta", [42.42])
    def test_SWAP(self, theta, backend):

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def f(theta):
            qml.RX(theta, wires=0)
            qml.RX(theta, wires=1)
            return qml.probs()

        @qjit
        @cancel_inverses
        @qml.qnode(qml.device(backend, wires=2))
        def g(theta):
            qml.RX(theta, wires=0)
            qml.RX(theta, wires=1)
            qml.SWAP(wires=(0, 1))
            qml.SWAP(wires=(0, 1))
            return qml.probs()

        expected = f(theta)
        result = g(theta)
        assert np.allclose(result, expected)

    ### Test two consecutive Toffoli gates get canceled out ###
    @pytest.mark.parametrize("theta", [42.42])
    def test_Toffoli(self, theta, backend):

        @qjit
        @qml.qnode(qml.device(backend, wires=3))
        def f(theta):
            qml.RX(theta, wires=0)
            qml.RX(theta, wires=1)
            qml.RX(theta, wires=2)
            return qml.probs()

        @qjit
        @cancel_inverses
        @qml.qnode(qml.device(backend, wires=3))
        def g(theta):
            qml.RX(theta, wires=0)
            qml.RX(theta, wires=1)
            qml.RX(theta, wires=2)
            qml.Toffoli(wires=(0, 1, 2))
            qml.Toffoli(wires=(0, 1, 2))
            return qml.probs()

        expected = f(theta)
        result = g(theta)
        assert np.allclose(result, expected)


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
            cancel_inverses(classical_func)

    test_cancel_inverses_not_on_qnode()


if __name__ == "__main__":
    pytest.main(["-x", __file__])
