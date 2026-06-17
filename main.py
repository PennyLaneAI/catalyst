import pennylane as qp


@qp.qjit(keep_intermediate=2, capture=False)
@qp.transform(pass_name="phase-folding")
@qp.qnode(device=qp.device("lightning.qubit", wires=2))
def circuit_base():
    qp.CNOT([0, 1])
    qp.T(1)
    qp.CNOT([0, 1])
    qp.CNOT([1, 0])
    qp.T(0)
    qp.CNOT([1, 0])
    return qp.probs()


@qp.qjit(keep_intermediate=2, capture=False)
@qp.transform(pass_name="phase-folding")
@qp.qnode(device=qp.device("lightning.qubit", wires=2))
def circuit_ex424():
    qp.T(0)
    qp.T(1)
    qp.CNOT([0, 1])
    qp.adjoint(qp.T(1))
    qp.CNOT([0, 1])

    qp.H(1)
    qp.T(0)
    qp.T(1)
    qp.CNOT([1, 0])
    qp.adjoint(qp.T(0))
    qp.CNOT([1, 0])


# @qp.qjit(keep_intermediate=2, capture=False)
# # @qp.transform(pass_name="phase-folding")
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circuit_ex425():
#     # qp.StatePrep([0, 1], wires=1)
#     qp.BasisState(0, wires=1)   # becomes dynamic, so not yet
#     qp.T(1)
#     qp.CNOT([0, 1])
#     qp.T(0)
#     qp.T(1)


@qp.qjit(keep_intermediate=2, capture=False)
@qp.transform(pass_name="phase-folding")
@qp.qnode(device=qp.device("lightning.qubit", wires=1))
def circuit_ex426():
    qp.T(0)
    qp.PauliX(0)
    qp.adjoint(qp.T(0))
    qp.PauliX(0)
    return qp.probs()


circuit_base()
circuit_ex424()
# circuit_ex425()
circuit_ex426()
