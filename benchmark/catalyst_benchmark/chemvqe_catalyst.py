import pennylane as qml
import jax.numpy as jnp
import catalyst
import numpy as np

from typing import Any, Dict
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from pennylane import AllSinglesDoubles
from catalyst import qjit
from jax.core import ShapedArray

from .types import Problem


@dataclass_json
@dataclass
class ProblemInfo:
    molname: str
    basis: str
    bond: float
    qubits: int
    gate_count: int


NQubits = int
PROBLEMS: Dict[NQubits, ProblemInfo] = {
    4: {"molname": "HeH+", "basis": "6-31G", "bond": "0.5", "qubits": 4, "gate_count": 4},
    6: {"molname": "H3+", "basis": "STO-3G", "bond": "0.5", "qubits": 6, "gate_count": 9},
    8: {"molname": "H4", "basis": "STO-3G", "bond": "0.5", "qubits": 8, "gate_count": 27},
    12: {"molname": "HF", "basis": "STO-3G", "bond": "0.5", "qubits": 12, "gate_count": 36},
    14: {"molname": "BeH2", "basis": "STO-3G", "bond": "0.5", "qubits": 14, "gate_count": 205},
    16: {"molname": "H8", "basis": "STO-3G", "bond": "0.5", "qubits": 16, "gate_count": 361},
}

DMDICT = {"finite-diff": "fd", "parameter-shift": "ps", "adjoint": "adj"}


class ProblemCVQE(Problem):
    def __init__(self, dev, nsteps=10, diff_method: str = "finite-diff", **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        self.diff_method = DMDICT[diff_method]
        self.nsteps = nsteps
        pi = ProblemInfo.from_dict(PROBLEMS[self.nqubits])
        molname, basis, bond = pi.molname, pi.basis, pi.bond
        data = qml.data.load("qchem", molname=molname, basis=basis, bondlength=bond)[0]
        symbols = data.molecule.symbols
        geometry = data.molecule.coordinates
        electrons = data.molecule.n_electrons
        self.ham, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry)
        assert qubits == self.nqubits
        self.hf_state = qml.qchem.hf_state(electrons, qubits)
        self.singles, self.doubles = qml.qchem.excitations(electrons, qubits)
        self.excitations = self.singles + self.doubles
        self.param_dtype = jnp.float64

    def trial_params(self, _: int) -> Any:
        return jnp.ones(len(self.excitations), dtype=self.param_dtype)


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
