import pennylane as qml
import jax.numpy as jnp
from typing import Any

from catalyst import qjit, for_loop, while_loop

from .types import Problem


def sudoku_nqubits(clause_list) -> int:
    return max(map(max, clause_list)) + 1  # type:ignore # Number of input qubits


def grover_loops(N) -> int:
    """Ref https://learn.microsoft.com/en-us/azure/quantum/concepts-grovers"""
    return int(jnp.floor(jnp.pi / 4 * jnp.sqrt((2**N) / 1) - 0.5))


class ProblemC(Problem):
    def __init__(self, dev, nlayers=None, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        nqubits = self.nqubits
        assert (nqubits - 3) % 2 == 0
        l = list(range((nqubits - 3) // 2))
        CLAUSE_LIST = list(zip(l, l[1:] + [0]))
        N = sudoku_nqubits(CLAUSE_LIST)
        iqr = jnp.array(list(range(N)))
        cqr = jnp.array([x + max(iqr) + 1 for x in range(len(CLAUSE_LIST))])
        oqr = jnp.array([x + max(cqr) + 1 for x in [0]])
        anc1 = jnp.array([x + max(oqr) + 1 for x in [0]])
        anc2 = jnp.array([x + max(anc1) + 1 for x in [0]])
        cqr_oqr = jnp.concatenate((cqr, oqr))
        total = jnp.concatenate((iqr, cqr, oqr, anc1, anc2))

        self.CLAUSE_LIST = jnp.array(CLAUSE_LIST)
        self.N = N
        self.iqr = iqr
        self.cqr = cqr
        self.oqr = oqr
        self.anc1 = anc1
        self.anc2 = anc2
        self.total = total
        self.cqr_oqr = cqr_oqr

        self.nlayers = grover_loops(N) if nlayers is None else nlayers
        assert len(total) == nqubits
        self.qcircuit = None

    def trial_params(self, i: int) -> Any:
        return jnp.array(
            [[jnp.pi / (i + 1) for _ in self.iqr] for _ in range(3)], dtype=jnp.float64
        )


def sudoku_oracle(t):
    """An oracle solving a simple binary-sudoku game: in 2x2 gird there should
    be (a) no column that conain the same binary value twice and (b) no row
    which contain the same binary value twice"""

    @for_loop(0, len(t.CLAUSE_LIST), 1)
    def loop1(i):
        clause = t.CLAUSE_LIST[i]
        qml.CNOT(wires=[t.iqr[clause[0]], t.cqr[i]])
        qml.CNOT(wires=[t.iqr[clause[1]], t.cqr[i]])

    loop1()

    # Flip 'output' bit if all clauses are satisfied
    qml.MultiControlledX(wires=t.cqr_oqr, work_wires=t.anc1)

    # Uncompute clauses to reset clause-checking bits to 0
    @while_loop(lambda i: i >= 0)
    def loop2(i):
        clause = t.CLAUSE_LIST[i]
        qml.CNOT(wires=[t.iqr[clause[1]], t.cqr[i]])
        qml.CNOT(wires=[t.iqr[clause[0]], t.cqr[i]])
        return i - 1

    loop2(len(t.CLAUSE_LIST) - 1)


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


def qcompile(p: ProblemC, weights):
    def _main(weights):
        # Initialize the state
        @for_loop(0, len(p.iqr), 1)
        def loop_init(qubit):
            qml.Hadamard(wires=[qubit])

        loop_init()

        @for_loop(0, jnp.int64(p.nlayers), 1)
        def loop(_):
            # Apply the oracle
            sudoku_oracle(p)
            # Apply the diffuser
            diffuser(p)

            qml.BasicEntanglerLayers(weights=weights, wires=p.iqr)
        loop()
        return qml.state()

    qcircuit = qml.QNode(_main, p.dev, **p.qnode_kwargs)
    # qcircuit.construct([weights], {})
    p.qcircuit = qcircuit


def workflow(p: ProblemC, weights):
    return p.qcircuit(weights)


def run_catalyst(N=7):
    p = ProblemC(qml.device("lightning.qubit", wires=N), None)

    @qjit
    def _main(params):
        qcompile(p, params)
        return workflow(p, params)

    return _main(p.trial_params(0))

