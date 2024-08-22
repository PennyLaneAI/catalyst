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
""" ChemVQE problem, PennyLane+Catalyst implementation """

# pylint: disable=too-many-instance-attributes
# pylint: disable=no-value-for-parameter; It happens when we use Catalyst control-flow

from dataclasses import dataclass
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
import pennylane as qml
from catalyst_benchmark.types import Problem
from jax.core import ShapedArray
from pennylane import AllSinglesDoubles

import catalyst
from catalyst import qjit

DMALIASES = {"finite-diff": "fd", "parameter-shift": "ps", "adjoint": "adj"}


@dataclass
class ProblemInfo:
    """ChemVQE problem specification"""

    name: str
    bond: float


# fmt:off
NQubits = int
PROBLEMS: Dict[NQubits, ProblemInfo] = {
    4:  ProblemInfo('HeH+', 0.775),  # 4 qubit
    6:  ProblemInfo('H3+',  0.874),  # 6 qubit
    8:  ProblemInfo('H4',   1.00),   # 8 qubit
    12: ProblemInfo('HF',   0.917), # 12 qubit
    14: ProblemInfo('BeH2', 1.300), # 14 qubit
    16: ProblemInfo('H8',   1.00),  # 16 qubit
}
# fmt:on


@dataclass
class ProblemCVQE(Problem):
    """Catalyst implementation details of the VQE problem"""

    def __init__(self, dev, diff_method, nsteps=10, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        self.nsteps = nsteps
        self.diff_method = DMALIASES[diff_method]
        pi = PROBLEMS[self.nqubits]
        data = qml.data.load("qchem", molname=pi.name, basis="STO-3G", bondlength=pi.bond)[0]
        electrons = data.molecule.n_electrons
        qubits = data.molecule.n_orbitals * 2
        assert qubits == self.nqubits
        ham = data.hamiltonian
        self.hf_state = qml.qchem.hf_state(electrons, qubits)
        self.singles, self.doubles = qml.qchem.excitations(electrons, qubits)
        self.excitations = self.singles + self.doubles
        self.ham = ham
        self.qcircuit = None
        self.qgrad = None

    def trial_params(self, _: int) -> Any:
        return jnp.zeros(len(self.excitations), dtype=jnp.float64)


def qcompile_hybrid(p: ProblemCVQE, weights):
    """Compile the quantum parts of the problem"""

    def _circuit(params):
        AllSinglesDoubles(params, range(p.nqubits), p.hf_state, p.singles, p.doubles)
        return qml.expval(qml.Hamiltonian(np.array(p.ham.coeffs), p.ham.ops))

    qcircuit = qml.QNode(_circuit, p.dev, **p.qnode_kwargs)
    qcircuit.construct([weights], {})
    p.qcircuit = qcircuit


def qcompile(p: ProblemCVQE, weights):
    """Compile the quantum parts of the problem"""
    qcompile_hybrid(p, weights)
    qgrad = catalyst.grad(p.qcircuit, argnums=0, method=p.diff_method)
    p.qgrad = qgrad
    return p


def workflow(p: ProblemCVQE, params):
    """Problem workflow"""
    stepsize = 0.5
    theta = params

    @catalyst.for_loop(0, p.nsteps, 1)
    def loop(_, theta):
        dtheta = p.qgrad(theta)
        return theta - dtheta[0] * stepsize

    return loop(theta)


def workflow_hybrid(p: ProblemCVQE, params):
    """Problem workflow"""
    stepsize = 0.5
    theta = params

    @catalyst.for_loop(0, p.nsteps, 1)
    def loop(_, theta):
        state = p.qcircuit(theta)[0]
        return theta - jnp.mean(state) * stepsize

    return loop(theta)


SHOTS = None
DIFFMETHOD = "finite-diff"
NSTEPS = 1


def run_catalyst(N=6):
    """Test problem entry point"""
    p = ProblemCVQE(
        dev=qml.device("lightning.qubit", wires=N, shots=SHOTS),
        nsteps=NSTEPS,
        diff_method=DIFFMETHOD,
    )
    params = p.trial_params(0)

    @qjit
    def _main(params: ShapedArray(params.shape, params.dtype)):
        qcompile(p, params)
        return workflow(p, params)

    theta = _main(p.trial_params(0))
    print(f"Resulting parameters: {theta}")
