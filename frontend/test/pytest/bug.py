import pennylane as qml
import pytest
from catalyst import qjit


def test_observable_as_parameter(backend):
    """Test to see if we can pass an observable parameter to qfunc."""

    coeffs0 = [0.3, -5.1]
    H0 = qml.Hamiltonian(qml.math.array(coeffs0), [qml.PauliZ(0), qml.PauliY(1)])

    @qjit
    def circuit(obs):
        return qml.expval(obs)

    circuit(H0)
