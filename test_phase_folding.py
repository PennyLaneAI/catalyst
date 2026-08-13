import pennylane as qp
from catalyst import qjit
from catalyst.passes import graph_decomposition

pauli_gates = {"PauliX", "PauliY", "PauliZ", "Identity"}
clifford_gates = pauli_gates | {"S", "Hadamard", "CNOT", "SWAP"}
clifford_rz_gates = clifford_gates | {"T", "RZ", "GlobalPhase"}

@qjit(autograph=True, keep_intermediate=2, capture=True, verbose=True)
@qp.transform(pass_name="convert-to-value-semantics")
@qp.transform(pass_name="phase-folding-qref")
@qp.transform(pass_name="cse")
@qp.transform(pass_name="convert-to-reference-semantics")
@qp.transform(pass_name="symbol-dce")
@qp.transforms.decompose(gate_set=clifford_rz_gates)
@qp.qnode(device=qp.device("lightning.qubit", wires=3))
def circ1(x: float):
    # qp.Toffoli(wires=[0,1,2])

    qp.RZ(1.2, 0)
    if x > 1.4:
        qp.X(0)
        qp.H(1)
    else:
        qp.Y(0)
        qp.RZ(0.3, 1)
    qp.RZ(2.3, 0)
    # qp.Toffoli(wires=[0,1,2])
    return qp.probs()

print(circ1(1.5))

# try with another simple program with Toffoli gates.
# this decomposition, doesn't work over control flow.
# try defining if with those ugly cond ops.
# try using graph decomposition stuff.
