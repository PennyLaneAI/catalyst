import pennylane as qml
import pennylane.numpy as pnp

from typing import List, Tuple, Optional, Any
from dataclasses import dataclass
from time import time

from .types import Problem


def sudoku_nqubits(clause_list) -> int:
    return max(map(max, clause_list)) + 1  # type:ignore # Number of input qubits


def grover_loops(N) -> int:
    """Ref https://learn.microsoft.com/en-us/azure/quantum/concepts-grovers"""
    return int(pnp.floor(pnp.pi / 4 * pnp.sqrt((2**N) / 1) - 0.5))


class ProblemPL(Problem):
    def __init__(self, dev, nlayers=None, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        nqubits = self.nqubits
        assert (nqubits - 3) % 2 == 0
        l = list(range((nqubits - 3) // 2))
        CLAUSE_LIST = list(zip(l, l[1:] + [0]))
        # CLAUSE_LIST = [(0,1),(0,2),(1,3),(2,3)]

        N = sudoku_nqubits(CLAUSE_LIST)

        iqr = list(range(N))  # Input quantum register
        cqr = [x + max(iqr) + 1 for x in range(len(CLAUSE_LIST))]  # Clause quantum register
        oqr = [x + max(cqr) + 1 for x in [0]]  # Solution qubit
        anc1 = [x + max(oqr) + 1 for x in [0]]
        anc2 = [x + max(anc1) + 1 for x in [0]]
        total = iqr + cqr + oqr + anc1 + anc2

        self.CLAUSE_LIST = CLAUSE_LIST
        self.N = N
        self.iqr = iqr
        self.cqr = cqr
        self.oqr = oqr
        self.anc1 = anc1
        self.anc2 = anc2
        self.total = total
        self.cqr_oqr = cqr + oqr

        self.nlayers = grover_loops(N) if nlayers is None else nlayers
        assert len(total) == nqubits, f"{N}, {nqubits} != len({total})"

    def trial_params(self, i: int) -> Any:
        return pnp.array([[pnp.pi / (i + 1) for _ in self.iqr] for _ in range(3)],
                         dtype=pnp.float64)


def sudoku_oracle(t):
    """An oracle solving a simple binary-sudoku game: in 2x2 gird there should
    be (a) no column that conain the same binary value twice and (b) no row
    which contain the same binary value twice"""
    clause_list = t.CLAUSE_LIST

    # Compute clauses
    for i, clause in enumerate(clause_list):
        qml.CNOT(wires=[t.iqr[clause[0]], t.cqr[i]])
        qml.CNOT(wires=[t.iqr[clause[1]], t.cqr[i]])

    # Flip 'output' bit if all clauses are satisfied
    qml.MultiControlledX(wires=t.cqr_oqr, work_wires=t.anc1)

    # Uncompute clauses to reset clause-checking bits to 0
    for i, clause in reversed(list(enumerate(clause_list))):
        qml.CNOT(wires=[t.iqr[clause[1]], t.cqr[i]])
        qml.CNOT(wires=[t.iqr[clause[0]], t.cqr[i]])


def diffuser(t):
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in t.iqr:
        qml.Hadamard(wires=[qubit])
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in t.iqr:
        qml.PauliX(wires=[qubit])
    # Do multi-controlled-Z gate
    for qubit in t.iqr[:-1]:
        qml.Hadamard(wires=[qubit])
    qml.MultiControlledX(wires=t.iqr, work_wires=t.anc2)
    for qubit in t.iqr[:-1]:
        qml.Hadamard(wires=[qubit])
    # Apply transformation |11..1> -> |00..0>
    for qubit in t.iqr:
        qml.PauliX(wires=[qubit])
    # Apply transformation |00..0> -> |s>
    for qubit in t.iqr:
        qml.Hadamard(wires=[qubit])


def grover_mainloop(t: ProblemPL, weights):
    # Initialize the state
    for qubit in t.iqr:
        qml.Hadamard(wires=[qubit])

    for _ in range(int(t.nlayers)):
        # Apply the oracle
        sudoku_oracle(t)
        # Apply the diffuser
        diffuser(t)

        qml.BasicEntanglerLayers(weights=weights, wires=t.iqr)
    return qml.state()


def workflow(t: ProblemPL, weights):
    @qml.qnode(t.dev, **t.qnode_kwargs)
    def _main(weights):
        return grover_mainloop(t, weights)

    return _main(weights)


def size(p: ProblemPL) -> int:
    @qml.qnode(p.dev, expansion_strategy='device', **p.qnode_kwargs)
    def _main(weights):
        return grover_mainloop(p, weights)
    _main.construct([p.trial_params(0)], {})
    return len(_main.tape.operations)


def run_jax_lightning_qubit(N=7):
    import jax

    jax.config.update("jax_enable_x64", True)

    p = ProblemPL(qml.device("lightning.qubit", wires=N, shots=None), 4, interface="jax")
    print(f"Size: {size(p)}")

    @jax.jit
    def _main(params):
        return workflow(p, params)

    return _main(p.trial_params(0))


def run_lightning_qubit(N=7):
    p = ProblemPL(qml.device("lightning.qubit", wires=N, shots=None), 4, interface=None)
    print(f"Size: {size(p)}")

    def _main(params):
        return workflow(p, params)

    return _main(p.trial_params(0))


def run_default_qubit(N=7):
    p = ProblemPL(qml.device("default.qubit", wires=N, shots=None), 4, interface=None)
    print(f"Size: {size(p)}")

    def _main(params):
        return workflow(p, params)

    return _main(p.trial_params(0))

