import jax
import numpy as np
import pennylane as qml

from catalyst.jax_primitives import subroutine


def test_classical_subroutine():
    """Dummy test"""

    @subroutine
    def identity(x):
        return x

    @qml.qjit
    def subroutine_test():
        return identity(1)

    assert subroutine_test() == 1


def test_quantum_subroutine():

    @subroutine
    def Hadamard0(wire):
        qml.Hadamard(wire)

    qml.capture.enable()

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def subroutine_test(c: int):
        Hadamard0(c)
        return qml.state()

    assert np, allclose(subroutine_test(0), jax.numpy.array([0.70710678 + 0.0j, 0.70710678 + 0.0j]))
    qml.capture.disable()


def test_quantum_subroutine_self_inverses():

    @subroutine
    def Hadamard0(wire):
        qml.Hadamard(wire)

    qml.capture.enable()

    @qml.qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def subroutine_test(c: int):
        Hadamard0(c)
        Hadamard0(c)
        return qml.state()

    assert np.allclose(
        subroutine_test(0), jax.numpy.array([complex(1.0, 0), complex(0.0, 0.0)], dtype=complex)
    )

    qml.capture.disable()


def test_quantum_subroutine_conditional():

    @subroutine
    def Hadamard0(wire):
        def true_path():
            qml.Hadamard(wires=[wire])

        def false_path(): ...

        qml.cond(wire != 0, true_path, false_path)()

    qml.capture.enable()

    @qml.qjit(autograph=False)
    @qml.qnode(qml.device("lightning.qubit", wires=1), autograph=False)
    def subroutine_test(c: int):
        Hadamard0(c)
        return qml.state()

    assert np.allclose(subroutine_test(0), jax.numpy.array([1.0, 0.0], dtype=complex))
    assert np, allclose(subroutine_test(0), jax.numpy.array([0.70710678 + 0.0j, 0.70710678 + 0.0j]))
    qml.capture.disable()
