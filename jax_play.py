import jax
import pennylane as qml
import catalyst
from catalyst import qjit, cond, measure

##########################

# @qjit
# def identity(x):
#     nest = lambda : x
#     return nest()+1

# print(identity(42.0))
# print(identity.jaxpr)


# def subcircuit(angle):
#   qml.RX(angle, wires=0)

###########################

# @qjit
# def foo(x, N):
#   dev = qml.device("lightning.qubit", wires=N)
#   @qml.qnode(dev)
#   def circuit():
#       #subcircuit(x)
#       qml.RX(x, wires=0)
#       return qml.probs()
#   return circuit()

# cat = qjit(foo)

# print(foo(42.0, 3))
# print(cat(42.0, 3))

############################

# @qjit
# @qml.qnode(qml.device("lightning.qubit", wires=1))
# def circuit(n):
#     @cond(n > 4)
#     def cond_fn():
#         return 1
#         #qml.Hadamard(0)
#         #return n**2

#     @cond_fn.otherwise
#     def else_fn():
#         return 2
#         #qml.RX(3.14, 0)
#         #return n

#     return cond_fn()

# print(circuit(10), circuit(1))


#############################

@qjit
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit(pred: bool):
    @cond(pred)
    def conditional_flip():
        qml.PauliX(0)

    @conditional_flip.otherwise
    def conditional_flip():
        qml.Identity(0)

    conditional_flip()

    return measure(wires=0)

print(circuit(False))
