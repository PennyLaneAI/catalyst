import pennylane as qp

from catalyst import qjit
from catalyst.passes import phase_folding

pauli_gates = {"PauliX", "PauliY", "PauliZ", "Identity"}
clifford_gates = pauli_gates | {"S", "Hadamard", "CNOT", "SWAP"}
clifford_rz_gates = clifford_gates | {"T", "RZ", "GlobalPhase"}


def CCZ(a: int, b: int, c: int):
    qp.T(a)
    qp.T(b)
    qp.T(c)
    qp.CNOT([a, b])
    qp.CNOT([b, c])
    qp.CNOT([c, a])
    qp.adjoint(qp.T(a))
    qp.adjoint(qp.T(b))
    qp.T(c)
    qp.CNOT([b, a])
    qp.adjoint(qp.T(a))
    qp.CNOT([b, c])
    qp.CNOT([c, a])
    qp.CNOT([a, b])
    return


def CCX(a: int, b: int, c: int):
    qp.H(c)
    CCZ(a, b, c)
    qp.H(c)
    return


def Toffoli(a: int, b: int, c: int):
    qp.T(a)
    qp.T(b)
    qp.H(c)

    qp.CNOT([c, a])

    qp.adjoint(qp.T(a))
    qp.CNOT([b, c])

    qp.CNOT([b, a])
    qp.adjoint(qp.T(c))

    qp.T(a)
    qp.CNOT([b, c])

    qp.CNOT([c, a])

    qp.adjoint(qp.T(a))
    qp.T(c)

    qp.CNOT([b, a])
    qp.H(c)

    return


# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.transforms.decompose(gate_set=clifford_rz_gates)
# @qp.qnode(device=qp.device("lightning.qubit", wires=3))
# def circ1(x: float):
#     # Toffoli(0, 1, 2)

#     qp.RZ(1.2, 0)
#     if x > 1.4:
#         qp.X(0)
#         qp.H(1)
#     else:
#         qp.Y(0)
#         qp.RZ(0.3, 1)
#     qp.RZ(2.3, 0)

#     # Toffoli(2, 1, 0)
#     return qp.probs()


# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.transforms.decompose(gate_set=clifford_rz_gates)
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def if_simple(x: float):
#     qp.T(0)
#     if x > 1.4:
#         qp.CNOT([0, 1])
#     # qp.T(0)
#     qp.adjoint(qp.T(0))

#     return qp.probs()

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.transforms.decompose(gate_set=clifford_rz_gates)
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def loop_block(x: int):
#     qp.T(0)
#     qp.T(1)
#     for i in range(x):
#         qp.T(0)
#     qp.H(0)
#     # qp.T(0)
#     qp.adjoint(qp.T(0))

#     return qp.probs()

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.transforms.decompose(gate_set=clifford_rz_gates)
# @qp.qnode(device=qp.device("lightning.qubit", wires=3))
# def loop_cycle(x: int):
#     qp.T(1)
#     for i in range(x):
#         qp.SWAP([0, 1])
#         qp.SWAP([1, 2])
#     # qp.T(1)
#     qp.adjoint(qp.T(1))

#     return qp.probs()

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.transforms.decompose(gate_set=clifford_rz_gates)
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def loop_h(x: int):
#     qp.T(1)
#     for i in range(x):
#         qp.H(0)
#     # qp.T(1)
#     qp.adjoint(qp.T(1))

#     return qp.probs()

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.transforms.decompose(gate_set=clifford_rz_gates)
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def loop_nested(x: int):
#     # qp.StatePrep([1, 0], wires=0)

#     qp.T(1)
#     for i in range(x):
#         qp.T(0)
#         for j in range(i):
#             qp.X(1)
#     # qp.T(1)
#     qp.adjoint(qp.T(1))

#     return qp.probs()

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.transforms.decompose(gate_set=clifford_rz_gates)
# @qp.qnode(device=qp.device("lightning.qubit", wires=3))
# def loop_nonlinear(x: int):
#     # qp.StatePrep([1, 0], wires=2)
#     qp.X(0)
#     CCX(0, 1, 2)
#     qp.T(2)
#     CCX(0, 1, 2)
#     qp.X(0)
#     for i in range(x):
#         qp.CNOT([0, 1])
#     qp.X(0)
#     CCX(0, 1, 2)
#     qp.adjoint(qp.T(2))
#     CCX(0, 1, 2)
#     qp.X(0)
#     return qp.state()

# # @qjit(autograph=True, keep_intermediate=2, capture=True)
# # @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.transforms.decompose(gate_set=clifford_rz_gates)
# # @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# # def loop_null(x: int):
# #     qp.T(1)

# #     qp.StatePrep([1, 0], wires=0)

# #     for i in range(x):
# #         qp.T(1)
# #         qp.T(0)
# #     qp.T(1)
# #     # qp.adjoint(qp.T(1))

# #     return qp.probs()

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.transforms.decompose(gate_set=clifford_rz_gates)
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def loop_simple(x: int):
#     qp.T(0)
#     for i in range(x):
#         qp.CNOT([0, 1])
#     # qp.T(0)
#     qp.adjoint(qp.T(0))

#     return qp.probs()

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.transforms.decompose(gate_set=clifford_rz_gates)
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def loop_swap(x: int):
#     qp.CNOT([0, 1])
#     qp.T(1)
#     qp.CNOT([0, 1])
#     for i in range(x):
#         qp.SWAP([0, 1])
#     qp.CNOT([0, 1])
#     # qp.T(1)
#     qp.adjoint(qp.T(1))
#     qp.CNOT([0, 1])

#     return qp.probs()

# # @qjit(autograph=True, keep_intermediate=2, capture=True)
# # @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.transforms.decompose(gate_set=clifford_rz_gates)
# # @qp.qnode(device=qp.device("lightning.qubit", wires=1))
# # def reset_simple():
# #     qp.T(0)
# #     qp.StatePrep([1, 0], wires=0)
# #     qp.T(0)

# #     return qp.probs()

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# @qp.qnode(device=qp.device("lightning.qubit", wires=3))
# def rus(x: int):
#     qp.H(0)
#     qp.T(0)

#     for i in range(x):
#         # reset 1, 2 to |0>
#         qp.H(1)
#         qp.H(2)
#         CCX(1, 2, 0)
#         qp.S(0)
#         CCX(1, 2, 0)
#         qp.Z(0)
#         qp.H(1)
#         qp.H(2)
#         qp.measure(1)
#         qp.measure(2)

#     qp.adjoint(qp.T(0))
#     qp.H(0)

#     return qp.probs()


# print(circ1(1.5))

# # try with another simple program with Toffoli gates.


@qjit(autograph=True, keep_intermediate=2, capture=False)
# @phase_folding(report_stats=True, trace_abstraction=True)
@qp.transform(pass_name="convert-to-value-semantics")
@qp.transform(pass_name="convert-to-reference-semantics")
@qp.qnode(device=qp.device("null.qubit", wires=257))
def grover_pf(x: int):
    n = 7
    a = 2**n
    trgt = 2 * a

    print(n)
    print(a)
    print(trgt)
    qp.X(trgt)

    for i in range(x):
        # Superposition
        for j in range(a):
            qp.H(j)

        # Oracle
        CCX(0, 1, a)
        for j in range(a - 4):
            CCX(j + 2, a + j, a + j + 1)
        qp.H(trgt)
        CCX(a - 1, a + a - 3, trgt)
        qp.H(trgt)
        for j in range(a - 4):
            CCX(j + 2, a + j, a + j + 1)
        CCX(0, 1, a)

        # Diffusion
        for j in range(a):
            qp.H(j)
            qp.X(j)
        CCX(0, 1, a)
        for j in range(a - 4):
            CCX(j + 2, a + j, a + j + 1)
        qp.H(a - 1)
        CCX(a - 2, a + a - 4, a - 1)
        qp.H(a - 1)
        for j in range(a - 4):
            CCX(j + 2, a + j, a + j + 1)
        CCX(0, 1, a)
        for j in range(a):
            qp.H(j)
            qp.X(j)

    return


# a = rus_pf(10)
# b = rus_qjit(10)
# print(a)
# print(b)
# print(a == b)

print(grover_pf(10))
