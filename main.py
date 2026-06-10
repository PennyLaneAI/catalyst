import pennylane as qp

@qp.qjit(keep_intermediate=2, capture=False)
@qp.qnode(device=qp.device("lightning.qubit"))
def circuit():
    qp.H(0)
    return qp.probs()


circuit()