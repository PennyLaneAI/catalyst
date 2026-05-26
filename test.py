import pennylane as qp

dev = qp.device("default.qubit", wires=10, shots=500)  # or ANY PL device

qp.capture.enable()

H = qp.Z(0) @ qp.Y(1)

@qp.qjit(autograph=True)
@qp.qnode(dev)
def circuit(x):
    if x > 2:
        qp.RX(x, wires=0)
    else:
        qp.RY(x ** 2, wires=1)
    return qp.expval(H)

print(circuit(0.543))


from catalyst.device import pycat_device

dev = qp.device("default.mixed", wires=2)  # or ANY PL device

dev = pycat_device(
    dev,
    custom_toml_path="/Users/josh/xanadu/catalyst/runtime/lib/backend/pennylane_python/pennylane_python.toml",
    seed=500
)

@qp.qjit(autograph=True)
@qp.qnode(dev)
def circuit(x):
    if x > 2:
        qp.RX(x, wires=0)
    else:
        qp.RY(x ** 2, wires=1)
    return qp.probs(wires=[0, 1])

print(circuit(0.543))
