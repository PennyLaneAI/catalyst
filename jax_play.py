import jax
import pennylane as qml
import catalyst
from catalyst import qjit

# @qjit
# def identity(x):
# 	return x

# print(identity(42.0))




@qjit
def foo(x, N):
	dev = qml.device("lightning.qubit", wires=N)
	@qml.qnode(dev)
	def circuit():
		return qml.probs()
	return circuit()

print(foo(42.0, 3))
