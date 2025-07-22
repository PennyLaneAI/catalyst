import jax
import pennylane as qml

from catalyst import *
from catalyst.jax_primitives import AbstractQbit, qdef

qml.capture.enable()


@qdef
def decomp(x, y, w):
    jax.debug.print("hello add {} and {}", x, y)
    qml.GlobalPhase(0.1)


@qjit
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit(x: float):
    qml.RX(x, wires=0)
    decomp(float, float, AbstractQbit())
    return qml.expval(qml.PauliZ(0))


print(circuit.mlir)
