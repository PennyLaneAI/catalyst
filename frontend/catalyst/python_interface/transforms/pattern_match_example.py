import pennylane as qml

import catalyst

from .pattern_match import pattern_match

qml.capture.enable()


def pattern1(w1, w2):
    qml.CZ([w1, w2])
    qml.PauliZ(w2)


def rewrite1(w1, w2):
    qml.PauliX(w1)
    qml.S(w1)
    qml.T(w2)


def pattern2(w1):
    qml.T(w1)


def rewrite2(w1):
    qml.PauliX(w1)
    qml.S(w1)
    qml.H(w1)


@pattern_match(patterns={pattern1: rewrite1, pattern2: rewrite2})
@catalyst.qjit
@qml.qnode(qml.device("lightning.qubit", wires=4))
def workflow():
    qml.CZ([0, 1])
    qml.Z(1)

    qml.T(0)

    qml.CNOT([1, 2])
    qml.PauliY(2)
    qml.PauliX(3)
    return qml.state()


xmod = workflow()
print(xmod)
# To view output, check out pattern_match_example.mlir
