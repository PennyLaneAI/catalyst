import jax
import pennylane as qml

from catalyst import *
from catalyst.jax_primitives import AbstractQbit, qdef

qml.capture.enable()

@qdef
def decomp(x, y, w):
    jax.debug.print("hello add {} and {}", x, y)
    qml.GlobalPhase(1.14)
    qml.RX(x, wires=w)
    qml.RY(y, wires=w)
    qml.CNOT(wires=[w, w+1])
    qml.RZ(x+y, wires=w) # param wire

@qjit
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit(x: float, w: int):
    qml.RX(x, wires=0)
    # decomp(float, float, AbstractQbit()) # not working with AbstractQbit for param wires 
    decomp(float, float, int)
    return qml.expval(qml.PauliZ(0))


print(circuit.mlir)
