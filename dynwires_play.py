import pennylane as qml
import pytest

import catalyst
from catalyst import qjit
from catalyst.debug import get_compilation_stage, replace_ir
from catalyst.third_party.oqd import OQDDevice

def ref():
    dev = qml.device("default.qubit", wires=10)

    @qml.qnode(dev)
    def circ():
        qml.RX(1.23, wires=9)
        return qml.expval(qml.Z(wires=9))
        #return qml.probs()

    return circ()

print("reference: ", ref())


#@qjit(keep_intermediate=True)
def f(num_qubits):
    print("compiling...")
    dev = qml.device("lightning.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circ():
        qml.RX(1.23, wires=2)
        #return qml.expval(qml.Z(wires=num_qubits-1))
        #return qml.probs(wires=[0,num_qubits-1])  # if not running manual IR
        return qml.probs()  # if running manual IR

    return circ()
print("ref with 3: ", f(3))
print("ref with 5: ", f(5))

f = qjit(keep_intermediate=True)(f)

manual = "0_probs.mlir"
with open(manual, "r") as file:
    ir = file.read()
replace_ir(f, "mlir", ir)

print("dynamic alloc with 3: ", f(3))
print("dynamic alloc again with 5: ", f(5))
