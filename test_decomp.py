import pennylane as qml

from catalyst.jax_primitives import qdef

qml.capture.enable()


@qdef
def foo(param=None):
    qml.RX(param, wires=0)


@qdef
def XX():
    print("It's XX")
    qml.PauliX(wires=0)
    qml.PauliX(wires=0)


@qml.qjit(keep_intermediate=True)
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit_func(param: float):

    foo(param + 1)
    # XX()
    return qml.expval(qml.PauliZ(0))


# print(circuit_func.jaxpr)
# print(circuit_func.mlir)
print(circuit_func(0.1))
