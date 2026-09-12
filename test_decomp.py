import jax
import pennylane as qp

from catalyst import qjit
from catalyst.jax_primitives import decomposition_rule
from catalyst.passes import graph_decomposition

# @decomposition_rule(op_type=qp.Toffoli)
# def decompose_toffoli(wires):
#     qp.Z(0)

# @qp.qjit(autograph=True, keep_intermediate=2, capture=False)
# # @qp.transform(pass_name="convert-to-value-semantics")
# # @qp.transform(pass_name="phase-folding-qref")
# # @qp.transform(pass_name="convert-to-reference-semantics")
# # @graph_decomposition(gate_set=[qp.H, qp.X, qp.Z, qp.Y, qp.S, qp.CNOT, qp.SWAP, qp.T, qp.RZ])
# @qp.qnode(device=qp.device("lightning.qubit", wires=3))
# def circuit_decomp(x: float):
#     qp.CNOT([0, 1])
#     # qp.Toffoli([0, 1, 2])
#     qp.T(1)

#     decompose_toffoli(jax.core.ShapedArray((3,), int))

#     # if x > 1.4:
#     #     qp.Toffoli([1, 2, 0])
#     # else:
#     #     qp.Y(0)
#     #     qp.RZ(x, wires=1)

#     return qp.probs()

# circuit_decomp(1.5)


# @qp.qjit(autograph=True, keep_intermediate=2, capture=False)
# # @qp.transform(pass_name="convert-to-value-semantics")
# # @qp.transform(pass_name="phase-folding-qref")
# @qp.transform(pass_name="convert-to-reference-semantics")
# # Clifford + Rz
# # @qp.decompose(gates=[qp.X, qp.Y, qp.Z, qp.RZ, qp.CNOT])
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circuit_base():
#     qp.CNOT([0, 1])
#     qp.T(1)
#     qp.CNOT([0, 1])
#     qp.CNOT([1, 0])
#     qp.T(0)
#     qp.CNOT([1, 0])
#     return qp.probs()

# circuit_base()

# pauli_gates = {qp.X, qp.Y, qp.Z, qp.I}
# clifford_gates = pauli_gates | {qp.S, qp.H, qp.CNOT, qp.SWAP}
# clifford_rz_gates = clifford_gates | {qp.T, qp.RZ}

pauli_gates = {"PauliX", "PauliY", "PauliZ", "Identity"}
clifford_gates = pauli_gates | {"S", "Hadamard", "CNOT", "SWAP"}
clifford_rz_gates = clifford_gates | {"T", "RZ", "GlobalPhase"}


# @qp.templates.Subroutine
def f1(x: float, wires):
    # def true_fn():
    #     qp.X(wires[0])

    # def false_fn():
    #     qp.H(wires[1])

    # qp.cond(x > 1.4, true_fn, false_fn)()
    if x > 1.4:
        qp.X(wires[0])
    else:
        qp.H(wires[1])
    return


@qjit(autograph=True, keep_intermediate=2, capture=True)
@qp.transform(pass_name="convert-to-value-semantics")
# @qp.transform(pass_name="phase-folding-qref")
@qp.transform(pass_name="cse")
@qp.transform(pass_name="convert-to-reference-semantics")
@qp.transform(pass_name="symbol-dce")
@qp.transforms.decompose(gate_set=clifford_rz_gates)
@qp.qnode(device=qp.device("lightning.qubit", wires=3))
def circ1(x: float):
    qp.Toffoli(wires=[0, 1, 2])
    # if x > 1.4:
    #     qp.X(0)
    # else:
    #     qp.H(1)

    f1(x, wires=[0, 1])
    # qp.X(0)
    return qp.probs()


# @qjit(autograph=True, keep_intermediate=2, capture=True)
# # @qp.transform(pass_name="convert-to-value-semantics")
# # @qp.transform(pass_name="phase-folding-qref")
# # @qp.transform(pass_name="cse")
# # @qp.transform(pass_name="convert-to-reference-semantics")
# # @qp.transform(pass_name="symbol-dce")
# @qp.transforms.decompose(gate_set=clifford_rz_gates)
# @qp.qnode(device=qp.device("lightning.qubit", wires=3))
# def circ2():
#     # qp.Toffoli(wires=[0,1,2])
#     qp.X(0)
#     return qp.probs()

# @qp.qnode(device=qp.device("lightning.qubit", wires=3))
# def tof():
#     qp.Toffoli(wires=[0,1,2])
#     return qp.expval(qp.Z(0))

# print(qp.draw(tof_decomp)())
# print(tof_decomp())
# print(tof_decomp.mlir)
# print(qp.draw(tof)())
# print(tof())
print(circ1(1.5))
# print(circ2())
