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
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circ_if_simple(x: float):
#     qp.T(0)
#     if x > 1.4:
#         qp.CNOT([0, 1])
#     # qp.T(0)
#     qp.adjoint(qp.T(0))

#     return qp.probs()

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circ_loop_block(x: int):
#     qp.T(0)
#     qp.T(1)
#     for i in range(x):
#         qp.T(0)
#     qp.H(0)
#     # qp.T(0)
#     qp.adjoint(qp.T(0))

#     return qp.probs()

# # @qjit(autograph=True, keep_intermediate=2, capture=True)
# # @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.qnode(device=qp.device("lightning.qubit", wires=3))
# # def circ_loop_cycle(x: int):
# #     qp.T(1)
# #     for i in range(x):
# #         qp.SWAP([0, 1])
# #         qp.SWAP([1, 2])
# #     # qp.T(1)
# #     qp.adjoint(qp.T(1))

# #     return qp.probs()

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circ_loop_h(x: int):
#     qp.T(1)
#     for i in range(x):
#         qp.H(0)
#     # qp.T(1)
#     qp.adjoint(qp.T(1))

#     return qp.probs()

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circ_loop_nested(x: int):
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
# @qp.qnode(device=qp.device("lightning.qubit", wires=3))
# def circ_loop_nonlinear(x: int):
#     qp.StatePrep([1, 0], wires=2)
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

# # # @qjit(autograph=True, keep_intermediate=0, capture=True)
# # # @qp.qnode(device=qp.device("lightning.qubit", wires=3))
# # # def circ_loop_nonlinear_no_pf(x: int):
# #     # qp.StatePrep([1, 0], wires=2)
# #     qp.X(0)
# #     CCX(0, 1, 2)
# #     qp.T(2)
# #     CCX(0, 1, 2)
# #     qp.X(0)
# #     for i in range(x):
# #         qp.CNOT([0, 1])
# #     qp.X(0)
# #     CCX(0, 1, 2)
# #     qp.adjoint(qp.T(2))
# #     CCX(0, 1, 2)
# #     qp.X(0)
# #     return qp.state()

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circ_loop_null(x: int):
#     qp.T(1)

#     # qp.StatePrep([1, 0], wires=0)

#     for i in range(x):
#         qp.T(1)
#         qp.T(0)
#     qp.T(1)
#     # qp.adjoint(qp.T(1))

#     return qp.probs()

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circ_loop_simple(x: int):
#     qp.T(0)
#     for i in range(x):
#         qp.CNOT([0, 1])
#     # qp.T(0)
#     qp.adjoint(qp.T(0))

#     return qp.probs()

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# def circ_loop_swap(x: int):
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

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.transforms.decompose(gate_set=clifford_rz_gates)
# @qp.qnode(device=qp.device("lightning.qubit", wires=1))
# def circ_reset_simple():
#     qp.T(0)
#     qp.StatePrep([1, 0], wires=0)
#     qp.T(0)

#     return qp.probs()

# @qjit(autograph=True, keep_intermediate=2, capture=True)
# @phase_folding(report_stats=True, trace_abstraction=True)
# @qp.qnode(device=qp.device("lightning.qubit", wires=3))
# def circ_rus(x: int):
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

# # @qjit(autograph=True, keep_intermediate=2, capture=True)
# # @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# # def circ_nested_loop_if(x: int, y: float):
#     qp.T(0)
#     qp.X(0)
#     for i in range(x):
#         qp.T(1)
#         qp.SWAP([0, 1])
#         if y > 1.4:
#             qp.T(0)
#             qp.T(1)
#             qp.CNOT([0, 1])
#             qp.T(1)
#             qp.CNOT([0, 1])
#         else:
#             qp.T(0)
#             qp.CNOT([0, 1])
#             qp.CNOT([1, 0])
#             qp.CNOT([0, 1])
            
#             qp.H(0)

#             qp.CNOT([0, 1])
#             qp.CNOT([1, 0])
#             qp.CNOT([0, 1])
#             qp.T(0)
#         qp.T(0)
#         qp.SWAP([0, 1])
#     qp.T(0)
#     return qp.probs()

# # @qjit(autograph=True, keep_intermediate=2, capture=True)
# # @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# # def circ_pure_ex():
# #     qp.CNOT([0, 1])
# #     qp.T(1)
# #     qp.CNOT([0, 1])
    
# #     qp.CNOT([1, 0])
# #     qp.adjoint(qp.T(0))
# #     qp.CNOT([1, 0])

# #     return qp.state()

# # @qjit(autograph=True, keep_intermediate=2, capture=True)
# # @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# # def circ_hybrid_ex(x: int):
# #     qp.CNOT([0, 1])
# #     qp.T(1)
# #     qp.CNOT([0, 1])
    
# #     for i in range(x):
# #         qp.SWAP([0, 1])

# #     qp.CNOT([1, 0])
# #     qp.adjoint(qp.T(0))
# #     qp.CNOT([1, 0])

# #     return qp.state()


# # @qjit(autograph=True, keep_intermediate=2, capture=True)
# # @phase_folding(report_stats=True, trace_abstraction=True)
# # @qp.qnode(device=qp.device("lightning.qubit", wires=2))
# # def circ_test(x: int):
# #     qp.Identity(0)
# #     qp.Identity(1)

# #     return qp.state()




# # import numpy

# # print("--------------------------------")
# # print(numpy.allclose(loop_nonlinear(10), loop_nonlinear_no_pf(10)))


@qjit(autograph=False, keep_intermediate=2, capture=False)
@phase_folding(report_stats=True, trace_abstraction=True)
@qp.qnode(device=qp.device("lightning.qubit", wires=10))
def circ_test(x: int):
    n = 5
    for i in range(x):
        qp.CNOT([0, 1])




        qp.CNOT([i, i + 1])

    return qp.state()