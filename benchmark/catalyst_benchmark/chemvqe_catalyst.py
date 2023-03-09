import pennylane as qml
import jax.numpy as jnp
import catalyst
import numpy as np

from typing import Any, Dict
from dataclasses import dataclass
from pennylane import AllSinglesDoubles
from catalyst import qjit
from jax.core import ShapedArray

from .types import Problem

DMALIASES = {"finite-diff": "fd", "parameter-shift": "ps", "adjoint": "adj"}

@dataclass
class ProblemInfo:
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


class ProblemCVQE(Problem):
    def __init__(self, dev, diff_method, nsteps=10, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        self.nsteps = nsteps
        self.diff_method = DMALIASES[diff_method]
        pi = PROBLEMS[self.nqubits]
        data = qml.data.load("qchem", molname=pi.name, basis="STO-3G", bondlength=pi.bond)[0]
        electrons = data.molecule.n_electrons
        qubits = data.molecule.n_orbitals * 2
        assert qubits == self.nqubits
        ham =  data.hamiltonian
        self.hf_state = qml.qchem.hf_state(electrons, qubits)
        self.singles, self.doubles = qml.qchem.excitations(electrons, qubits)
        self.excitations = self.singles + self.doubles
        self.ham = ham

    def trial_params(self, _: int) -> Any:
        return jnp.zeros(len(self.excitations), dtype=jnp.float64)


def workflow(p: ProblemCVQE, params):
    @qml.qnode(p.dev, **p.qnode_kwargs)
    def circuit(params):
        AllSinglesDoubles(params, range(p.nqubits), p.hf_state, p.singles, p.doubles)
        return qml.expval(qml.Hamiltonian(np.array(p.ham.coeffs), p.ham.ops))

    stepsize = 0.5
    diff = catalyst.grad(circuit, argnum=0, method=p.diff_method)
    theta = params

    @catalyst.for_loop(0, p.nsteps, 1)
    def loop(i, theta):
        dtheta = diff(theta)
        return theta - dtheta[0] * stepsize

    return loop(theta)


SHOTS = None
DIFFMETHOD = "finite-diff"
NSTEPS = 1


def run_catalyst(N=6):
    p = ProblemCVQE(
        dev=qml.device("lightning.qubit", wires=N, shots=SHOTS),
        nsteps=NSTEPS,
        diff_method=DIFFMETHOD,
    )
    params = p.trial_params(0)

    @qjit
    def _main(params: ShapedArray(params.shape, params.dtype)):
        return workflow(p, params)

    theta = _main(p.trial_params(0))
    print(f"Resulting parameters: {theta}")
