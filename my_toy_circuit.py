import jax
import numpy as np
import pennylane as qml

import catalyst
from catalyst import qjit

# dev = qml.device("lightning.qubit", wires=2, shots=5)
dev = qml.device("lightning.qubit", wires=3)
U = 1j / np.sqrt(2) * np.array([[1, 1], [1, -1]])

pipeline = {"cancel_inverses":{}, "merge_rotations":{}}

# @qjit(seed=37, keep_intermediate=True)
#@qjit(keep_intermediate=True)
@qjit(circuit_transform_pipeline = pipeline)
@qml.qnode(dev)
def circuit(theta):
    qml.ctrl(qml.GlobalPhase(0.123), 0)
    qml.ctrl(qml.adjoint(qml.GlobalPhase(0.123)), 0)
    qml.GlobalPhase(1.23)
    qml.adjoint(qml.GlobalPhase(1.23))
    qml.Hadamard(wires=[0])
    qml.RX(theta, wires=[1])
    qml.adjoint(qml.RX(theta, wires=[1]))
    qml.QubitUnitary(U, wires=[2])
    qml.adjoint(qml.QubitUnitary(U, wires=[2]))
    qml.adjoint(qml.RY(12.34, wires=[1]))
    qml.RY(12.34, wires=[1])
    # return qml.expval(qml.PauliZ(wires=[0]))
    # return qml.sample()
    return qml.probs()


res = circuit(12.3)
print(res)
