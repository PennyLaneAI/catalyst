import pennylane.numpy as pnp
import pennylane as qml

from typing import Any, Dict
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from pennylane import AllSinglesDoubles

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


class ProblemCVQE(Problem):
    def __init__(self, dev, nsteps=10, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
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

    def trial_params(self, _: int) -> Any:
        return pnp.ones(len(self.excitations), dtype=pnp.float64)


def workflow(p: ProblemCVQE, params):
    @qml.qnode(p.dev, **p.qnode_kwargs)
    def circuit(params):
        AllSinglesDoubles(params, range(p.nqubits), p.hf_state, p.singles, p.doubles)
        return qml.expval(p.ham)

    stepsize = 0.5
    diff = qml.grad(circuit, argnum=0)
    theta = params

    for i in range(p.nsteps):
        dtheta = diff(theta)
        theta = theta - dtheta * stepsize

    return theta

SHOTS = None
DIFFMETHOD = 'parameter-shift'
NSTEPS = 1

def run_default_qubit(N=6):
    p = ProblemCVQE(qml.device("default.qubit", wires=N, shots=SHOTS),
                    nsteps=NSTEPS,
                    diff_method=DIFFMETHOD)

    def _main(params):
        return workflow(p, params)

    theta = _main(p.trial_params(0))
    print(f"Resulting parameters: {theta}")


def run_lightning_qubit(N=6):
    p = ProblemCVQE(qml.device("lightning.qubit", wires=N, shots=SHOTS),
                    nsteps=NSTEPS, diff_method=DIFFMETHOD)

    def _main(params):
        return workflow(p, params)

    theta = _main(p.trial_params(0))
    print(f"Resulting parameters: {theta}")


def run_jax_(devname, N=6):
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_array", True)

    p = ProblemCVQE(dev=qml.device(devname, wires=N, shots=SHOTS),
                    nsteps=NSTEPS, interface="jax", diff_method=DIFFMETHOD)

    @jax.jit
    def _main(params):
        return workflow(p, params)

    theta = _main(p.trial_params(0))
    print(f"Resulting parameters: {theta}")


def run_jax_default_qubit(N=6):
    return run_jax_("default.qubit.jax", N)


def run_jax_lightning_qubit(N=6):
    return run_jax_("lightning.qubit", N)

