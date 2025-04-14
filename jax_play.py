import jax
import pennylane as qml
import catalyst
from catalyst import qjit, cond

##########################

# @qjit
# def identity(x):
# 	return x

# print(identity(42.0))
# print(identity.jaxpr)


# def subcircuit(angle):
# 	qml.RX(angle, wires=0)

###########################

# #@qjit
# def foo(x, N):
# 	dev = qml.device("lightning.qubit", wires=N)
# 	@qml.qnode(dev)
# 	def circuit():
# 		subcircuit(x)
# 		return qml.probs()
# 	return circuit()

# cat = qjit(foo)

# print(foo(42.0, 3))
# print(cat(42.0, 3))

############################

@qjit
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit(n):
    @cond(n > 4)
    def cond_fn():
        return n**2

    @cond_fn.otherwise
    def else_fn():
        return n+1

    return cond_fn()

print(circuit(10), circuit(1))
