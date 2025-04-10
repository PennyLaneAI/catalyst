import jax
import pennylane as qml
import catalyst
from catalyst import qjit

# @qjit
# def identity(x):
# 	return x

# print(identity(42.0))


dev = qml.device("lightning.qubit", wires=1)

@qjit
@qml.qnode(dev)
def foo():
	return qml.probs()

print(foo())
