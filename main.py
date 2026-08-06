import pennylane as qp
# from catalyst import qjit

from catalyst import cond, for_loop


# dev = qp.device("lightning.qubit", wires=2)

# @qjit(autograph=True, keep_intermediate=2)
# @qp.transform(pass_name="convert-to-reference-semantics")
# @qp.qnode(dev)
# def circuit(x: float):

#     qp.CNOT([0, 1])
#     qp.T(1)
#     qp.CNOT([0, 1])
#     qp.CNOT([1, 0])
#     qp.T(0)
#     qp.CNOT([1, 0])

#     # if x > 0.5:
#     #     qp.T(0)
#     #     qp.Hadamard(wires=0)
#     # else:
#     #     qp.RZ(x, wires=0)
        
#     return qp.probs()

# # 4. Trigger the JIT compilation by calling the function
# result = circuit(0.8)
# # print(f"Execution Result: {result}\n")

# # 5. Print the primary MLIR representation
# print("=== Generated MLIR ===")
# print(circuit.mlir)



# @qp.qjit(autograph=True, keep_intermediate=2, capture=False)
# @qp.transform(pass_name="convert-to-value-semantics")
# @qp.transform(pass_name="phase-folding-qref")
# @qp.transform(pass_name="convert-to-reference-semantics")
# # Clifford + Rz
# # @qp.decompose(gates=[qp.X, qp.Y, qp.Z, qp.RZ, qp.CNOT])
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circuit_base(x: float):
#     # qp.CNOT([0, 1])
#     # qp.T(1)
#     # qp.CNOT([0, 1])
#     # qp.CNOT([1, 0])
#     # qp.T(0)
#     # qp.CNOT([1, 0])

#     qp.T(0)

#     if x > 1.4:
#         qp.X(0)
#         qp.Hadamard(wires=1)
#     else:
#         qp.Y(0)
#         qp.RZ(x, wires=1)

#     qp.T(0)

#     return qp.probs()


# circuit_base(1.5)
# @qp.qjit(keep_intermediate=2, capture=False)
# @qp.transform(pass_name="phase-folding")
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circuit_ex424():
#     qp.T(0)
#     qp.T(1)
#     qp.CNOT([0, 1])
#     qp.adjoint(qp.T(1))
#     qp.CNOT([0, 1])

#     qp.H(1)
#     qp.T(0)
#     qp.T(1)
#     qp.CNOT([1, 0])
#     qp.adjoint(qp.T(0))
#     qp.CNOT([1, 0])


@qp.qjit(keep_intermediate=2, capture=False)
# @qp.transform(pass_name="phase-folding")
@qp.qnode(device=qp.device("lightning.qubit", wires=2))
def circuit_ex425():
    qp.StatePrep([0, 1], wires=1)
    qp.BasisState(0, wires=1)   # becomes dynamic, so not yet
    qp.T(1)
    qp.CNOT([0, 1])
    qp.T(0)
    qp.T(1)


# @qp.qjit(keep_intermediate=2, capture=False)
# @qp.transform(pass_name="phase-folding")
# @qp.qnode(device=qp.device("lightning.qubit", wires=1))
# def circuit_ex426():
#     qp.T(0)
#     qp.PauliX(0)
#     qp.adjoint(qp.T(0))
#     qp.PauliX(0)
#     return qp.probs()


# circuit_base()
# circuit_ex424()
circuit_ex425()
# circuit_ex426()

# # gates you support
# # gates you don't support
# # control flow