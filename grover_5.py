"""PennyLane translation of Feynman benchmarks/qasm/grover_5.qasm.

9 wires: search qubits 0–4, phase qubit 5, ancillas 6–8.
The QASM wraps every CCX in H H · CCX · H H (a no-op sandwich).
"""

import pennylane as qp

from catalyst import qjit
from catalyst.passes import phase_folding


def CCZ(p: int, q: int, r: int):
    qp.T(p)
    qp.T(q)
    qp.T(r)
    qp.CNOT([p, q])
    qp.CNOT([q, r])
    qp.CNOT([r, p])
    qp.adjoint(qp.T(p))
    qp.adjoint(qp.T(q))
    qp.T(r)
    qp.CNOT([q, p])
    qp.adjoint(qp.T(p))
    qp.CNOT([q, r])
    qp.CNOT([r, p])
    qp.CNOT([p, q])
    return


def CCX(p: int, q: int, r: int):
    qp.H(r)
    CCZ(p, q, r)
    qp.H(r)
    return


n = 6
a = 2**n
trgt = 2 * a

def reset():
    for i in range(trgt + 1):
        print(i)
        qp.BasisState(0, wires=i)
    return

def superposition():
    for i in range(a):
        qp.H(i)
    # qp.H(0)
    # qp.H(1)
    # qp.H(2)
    # qp.H(3)
    # qp.H(4)


# def loop_ccx():
#     # for j in range(a - 4):
#     #     CCX(j + 2, a + j, a + j + 1)
#     j = 0
#     CCX(j + 2, a + j, a + j + 1)
#     j = 1
#     CCX(j + 2, a + j, a + j + 1)
#     j = 2
#     CCX(j + 2, a + j, a + j + 1)
#     j = 3
#     CCX(j + 2, a + j, a + j + 1)
#     j = 4
#     CCX(j + 2, a + j, a + j + 1)
#     j = 5
#     CCX(j + 2, a + j, a + j + 1)
#     j = 6
#     CCX(j + 2, a + j, a + j + 1)
#     j = 7
#     CCX(j + 2, a + j, a + j + 1)
#     j = 8
#     CCX(j + 2, a + j, a + j + 1)
#     j = 9
#     CCX(j + 2, a + j, a + j + 1)
#     j = 10
#     CCX(j + 2, a + j, a + j + 1)
#     j = 11
#     CCX(j + 2, a + j, a + j + 1)
#     j = 12
#     CCX(j + 2, a + j, a + j + 1)
#     j = 13
#     CCX(j + 2, a + j, a + j + 1)
#     j = 14
#     CCX(j + 2, a + j, a + j + 1)
#     j = 15
#     CCX(j + 2, a + j, a + j + 1)
#     j = 16
#     CCX(j + 2, a + j, a + j + 1)
#     j = 17
#     CCX(j + 2, a + j, a + j + 1)
#     j = 18
#     CCX(j + 2, a + j, a + j + 1)
#     j = 19
#     CCX(j + 2, a + j, a + j + 1)
#     j = 20
#     CCX(j + 2, a + j, a + j + 1)
#     j = 21
#     CCX(j + 2, a + j, a + j + 1)
#     j = 22
#     CCX(j + 2, a + j, a + j + 1)
#     j = 23
#     CCX(j + 2, a + j, a + j + 1)
#     j = 24
#     CCX(j + 2, a + j, a + j + 1)
#     j = 25
#     CCX(j + 2, a + j, a + j + 1)
#     j = 26
#     CCX(j + 2, a + j, a + j + 1)
#     j = 27
#     CCX(j + 2, a + j, a + j + 1)


# def loop_h_x():
#     # for j in range(a):
#     #     qp.H(j)
#     #     qp.X(j)
#     j = 0
#     qp.H(j)
#     qp.X(j)
#     j = 1
#     qp.H(j)
#     qp.X(j)
#     j = 2
#     qp.H(j)
#     qp.X(j)
#     j = 3
#     qp.H(j)
#     qp.X(j)
#     j = 4
#     qp.H(j)
#     qp.X(j)
#     j = 5
#     qp.H(j)
#     qp.X(j)
#     j = 6
#     qp.H(j)
#     qp.X(j)
#     j = 7
#     qp.H(j)
#     qp.X(j)
#     j = 8
#     qp.H(j)
#     qp.X(j)
#     j = 9
#     qp.H(j)
#     qp.X(j)
#     j = 10
#     qp.H(j)
#     qp.X(j)
#     j = 11
#     qp.H(j)
#     qp.X(j)
#     j = 12
#     qp.H(j)
#     qp.X(j)
#     j = 13
#     qp.H(j)
#     qp.X(j)
#     j = 14
#     qp.H(j)
#     qp.X(j)
#     j = 15
#     qp.H(j)
#     qp.X(j)
#     j = 16
#     qp.H(j)
#     qp.X(j)
#     j = 17
#     qp.H(j)
#     qp.X(j)
#     j = 18
#     qp.H(j)
#     qp.X(j)
#     j = 19
#     qp.H(j)
#     qp.X(j)
#     j = 20
#     qp.H(j)
#     qp.X(j)
#     j = 21
#     qp.H(j)
#     qp.X(j)
#     j = 22
#     qp.H(j)
#     qp.X(j)
#     j = 23
#     qp.H(j)
#     qp.X(j)
#     j = 24
#     qp.H(j)
#     qp.X(j)
#     j = 25
#     qp.H(j)
#     qp.X(j)
#     j = 26
#     qp.H(j)
#     qp.X(j)
#     j = 27
#     qp.H(j)
#     qp.X(j)
#     j = 28
#     qp.H(j)
#     qp.X(j)
#     j = 29
#     qp.H(j)
#     qp.X(j)
#     j = 30
#     qp.H(j)
#     qp.X(j)
#     j = 31
#     qp.H(j)
#     qp.X(j)
#     j = 32
#     return


def oracle():
    CCX(0, 1, a)
    for j in range(a - 4):
        CCX(j + 2, a + j, a + j + 1)
    # loop_ccx()

    qp.H(trgt)
    CCX(a - 1, a + a - 3, trgt)
    qp.H(trgt)

    for j in range(a - 4):
        CCX(j + 2, a + j, a + j + 1)
    # loop_ccx()

    CCX(0, 1, a)
    return


def diffusion():
    for j in range(a):
        qp.H(j)
        qp.X(j)
    # loop_h_x()
    CCX(0, 1, a)

    for j in range(a - 4):
        CCX(j + 2, a + j, a + j + 1)
    # loop_ccx()

    qp.H(a - 1)
    CCX(a - 2, a + a - 4, a - 1)
    qp.H(a - 1)

    for j in range(a - 4):
        CCX(j + 2, a + j, a + j + 1)
    # loop_ccx()

    CCX(0, 1, a)
    for j in range(a):
        qp.H(j)
        qp.X(j)
    # loop_h_x()
    return


@qjit(autograph=False, keep_intermediate=2, capture=False)
# @phase_folding(report_stats=True, trace_abstraction=True)
@qp.transform(pass_name="convert-to-value-semantics")
@qp.transform(pass_name="convert-to-reference-semantics")
@qp.qnode(device=qp.device("null.qubit", wires=129))
def grover_5(x: int):

    reset()
    
    
    qp.X(trgt)

    # for _ in range(x):
    superposition()
    oracle()
    diffusion()

    return

# @qjit(autograph=False, keep_intermediate=0, capture=False)
# @phase_folding(report_stats=True, trace_abstraction=False)
# # @qp.transform(pass_name="convert-to-value-semantics")
# # @qp.transform(pass_name="convert-to-reference-semantics")
# @qp.qnode(device=qp.device("null.qubit", wires=65))
# def grover_pf_5_loopless():
    # n = 5
    # a = 2**n
    # trgt = 2 * a
    # x = 1

    # print(n)
    # print(a)
    # print(trgt)
    # qp.X(trgt)


    # @loop
    # for i in range(x):
    #     # Superposition
    #     for j in range(a):
    #         qp.H(j)

    #     # Oracle
    #     CCX(0, 1, a)
    #     for j in range(a - 4):
    #         CCX(j + 2, a + j, a + j + 1)
    #     qp.H(trgt)
    #     CCX(a - 1, a + a - 3, trgt)
    #     qp.H(trgt)
    #     for j in range(a - 4):
    #         CCX(j + 2, a + j, a + j + 1)
    #     CCX(0, 1, a)

    #     # Diffusion
    #     for j in range(a):
    #         qp.H(j)
    #         qp.X(j)
    #     CCX(0, 1, a)
    #     for j in range(a - 4):
    #         CCX(j + 2, a + j, a + j + 1)
    #     qp.H(a - 1)
    #     CCX(a - 2, a + a - 4, a - 1)
    #     qp.H(a - 1)
    #     for j in range(a - 4):
    #         CCX(j + 2, a + j, a + j + 1)
    #     CCX(0, 1, a)
    #     for j in range(a):
    #         qp.H(j)
    #         qp.X(j)

    # return

# print(qp.spec(grover_pf)())
