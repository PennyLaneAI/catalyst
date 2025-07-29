import jax
import pennylane as qml

from catalyst import *
from catalyst.jax_primitives import AbstractQbit, decomposition_rule

qml.capture.enable()


@decomposition_rule
def decomp(x, y, wires):
    jax.debug.print("hello add {} and {}", x, y)
    qml.GlobalPhase(1.14)
    # qml.RX(x, wires=wires[0])
    # qml.RY(y, wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RZ(x + y, wires=2) # param wire

@qjit
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit(x: float, wires: jax.core.ShapedArray(shape=[2], dtype=int)):
    qml.RX(x, wires=0)
    # decomp(float, float, AbstractQbit()) # not working with AbstractQbit for param wires
    decomp(float, float, wires)
    return qml.expval(qml.PauliZ(0))


# print(circuit.mlir)
print(circuit.jaxpr)
