import pennylane as qml
import pytest

import catalyst
from catalyst import qjit
from catalyst.debug import get_compilation_stage, replace_ir
from catalyst.third_party.oqd import OQDDevice

def f():
    dev = qml.device("default.qubit")
    @qml.qnode(dev)
    def circ():
        qml.RX(1.23, wires=2)
        qml.RX(4.56, wires=3)
        #return qml.expval(qml.Z(wires=2)) # scalar return fine
        return qml.probs(wires=[2,3])  # static ret shape, fine
        #return qml.probs()  # dynamic ret shape, need to update?

    return circ()

print("reference: ", f())

@qjit
def f_cat():
    dev = qml.device("lightning.qubit", wires=0)
    # explcit wires=0 for now since I don't want to do tracing in runtime prototype
    # But the idea is do quantum.alloc(0) if wires argument not present

    @qml.qnode(dev)
    def circ():
        qml.RX(1.23, wires=2)
        qml.RX(4.56, wires=3)
        #return qml.expval(qml.Z(wires=2)) # scalar return fine
        return qml.probs(wires=[2,3])  # static ret shape, fine
        #return qml.probs()  # dynamic ret shape, need to update?

    return circ()

print("cat: ", f_cat())

