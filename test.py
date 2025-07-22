import jax
import pennylane as qml

from catalyst import *
from catalyst.jax_primitives import AbstractQbit, qdef

qml.capture.enable()


@qdef(num_params=2)
def decomp(x, y, w1, w2):
    qml.GlobalPhase(x + y)
    qml.H(w1)
    qml.CNOT([w1, w2])


@qjit
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit(x: float):
    qml.RX(x, wires=0)
    decomp(float, float, int, int)
    return qml.expval(qml.PauliZ(0))


print(circuit.mlir)
