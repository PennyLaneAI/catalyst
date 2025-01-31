# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Grover-like problem, PennyLane/PennyLane+JAX implementation"""

# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-instance-attributes

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pennylane as qml
import pennylane.numpy as pnp
from catalyst_benchmark.types import Problem


def clause_nqubits(clause_list) -> int:
    """Problem-specific number of qubits"""
    return max(map(max, clause_list)) + 1  # type:ignore # Number of input qubits


def grover_loops(N) -> int:
    """Ref https://learn.microsoft.com/en-us/azure/quantum/concepts-grovers"""
    return int(pnp.floor(pnp.pi / 4 * pnp.sqrt((2**N) / 1) - 0.5))


@dataclass
class ProblemPL(Problem):
    """PennyLane implementation details of the Grover problem"""

    def __init__(self, dev, nlayers=None, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        nqubits = self.nqubits
        assert (nqubits - 3) % 2 == 0
        l = list(range((nqubits - 3) // 2))
        CLAUSE_LIST = list(zip(l, l[1:] + [0]))
        N = clause_nqubits(CLAUSE_LIST)
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
        self.qcircuit = None

    def trial_params(self, i: int) -> Any:
        return pnp.array(
            [[pnp.pi / (i + 1) for _ in self.iqr] for _ in range(3)], dtype=pnp.float64
        )


def oracle(t):
    """A Grover oracle solving a mock combinatorial problem."""
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
    """Diffuser part of the Grover algorithm"""
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


def qcompile(p: ProblemPL, weights):
    """Compile the quantum parts of the problem"""

    def _main(weights):
        # Initialize the state
        for qubit in p.iqr:
            qml.Hadamard(wires=[qubit])

        for _ in range(int(p.nlayers)):
            # Apply the oracle
            oracle(p)
            # Apply the diffuser
            diffuser(p)

            qml.BasicEntanglerLayers(weights=weights, wires=p.iqr)
        return qml.state()

    qcircuit = qml.QNode(_main, p.dev, **p.qnode_kwargs)
    qcircuit.construct([weights], {})
    p.qcircuit = qcircuit


def workflow(p: ProblemPL, weights):
    """Problem workflow"""
    return p.qcircuit(weights)


def size(p: ProblemPL) -> int:
    """Compute the size of the problem circuit"""
    qnode_kwargs = deepcopy(p.qnode_kwargs)
    qnode_kwargs.update({"expansion_strategy": "device"})
    qcompile(p, p.trial_params(0))
    return len(p.qcircuit.tape.operations)


def run_jax_default_qubit(N=7, L=10):
    """Test entry point"""
    import jax

    jax.config.update("jax_enable_x64", True)

    p = ProblemPL(qml.device("default.qubit", wires=N, shots=None), L, interface="jax")
    print(f"Size: {size(p)}")

    @jax.jit
    def _main(params):
        qcompile(p, params)
        return workflow(p, params)

    return _main(p.trial_params(0))


def profile_jax_default_qubit(N=7, L=10):
    """Test entry point"""
    import jax

    jax.config.update("jax_enable_x64", True)

    p = ProblemPL(qml.device("default.qubit", wires=N, shots=None), L, interface="jax")
    print(f"Size: {size(p)}")

    @jax.jit
    def _main(params):
        qcompile(p, params)
        return workflow(p, params)

    with jax.profiler.trace("_jax-trace", create_perfetto_link=True):
        jax.make_jaxpr(_main)(p.trial_params(0))


def run_jax_lightning_qubit(N=7):
    """Test entry point"""
    import jax

    jax.config.update("jax_enable_x64", True)

    p = ProblemPL(qml.device("lightning.qubit", wires=N, shots=None), 4, interface="jax")
    print(f"Size: {size(p)}")

    @jax.jit
    def _main(params):
        qcompile(p, params)
        return workflow(p, params)

    return _main(p.trial_params(0))


def run_lightning_qubit(N=7):
    """Test entry point"""
    p = ProblemPL(qml.device("lightning.qubit", wires=N, shots=None), 4, interface=None)
    print(f"Size: {size(p)}")

    def _main(params):
        qcompile(p, params)
        return workflow(p, params)

    return _main(p.trial_params(0))


def run_default_qubit(N=7):
    """Test entry point"""
    p = ProblemPL(qml.device("default.qubit", wires=N, shots=None), 4, interface=None)
    print(f"Size: {size(p)}")

    def _main(params):
        qcompile(p, params)
        return workflow(p, params)

    return _main(p.trial_params(0))
