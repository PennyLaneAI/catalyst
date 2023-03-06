import catalyst
import pennylane as qml
import numpy as np
import jax.numpy as jnp

from typing import Any
from catalyst import qjit
from .types import Problem

symbols = ["H", "H", "H"]
coordinates = np.array([0.028, 0.054, 0.0, 0.986, 1.610, 0.0, 1.855, 0.002, 0.0])


class ProblemVQE(Problem):
    def __init__(self, dev, ntrials=10, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        self.params = jnp.array([0.0, 0.0])
        self.ntrials = ntrials
        self.hf = qml.qchem.hf_state(electrons=2, orbitals=self.nqubits)

        # Building the molecular hamiltonian for the trihydrogen cation
        self.hamiltonian, self.molqubits = qml.qchem.molecular_hamiltonian(
            symbols, coordinates, charge=1
        )
        assert (
            self.nqubits >= self.molqubits
        ), f"self.nqubits {self.nqubits} >= self.molqubits {self.molqubits}"

    def trial_params(self, i: int) -> Any:
        return jnp.array([])


def grad_descent(p: ProblemVQE, _):
    @qml.qnode(p.dev, **p.qnode_kwargs)
    def catalyst_cost_func(params):
        qml.BasisState(p.hf, wires=list(range(p.nqubits)))
        qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
        qml.DoubleExcitation(params[1], wires=[0, 1, 4, 5])
        return qml.expval(qml.Hamiltonian(np.array(p.hamiltonian.coeffs), p.hamiltonian.ops))

    stepsize = 0.4
    diff = catalyst.grad(catalyst_cost_func, argnum=0)
    theta = p.params

    # for_loop can only be used in JIT mode
    @catalyst.for_loop(0, p.ntrials, 1)
    def loop(i, theta):
        h = diff(theta)
        return theta - h[0] * stepsize

    return loop(theta)


def run(N=6):
    """Test run function"""
    p = ProblemVQE(dev=qml.device("lightning.qubit", wires=N), ntrials=2)

    @qjit
    def _main(pp):
        return grad_descent(p, pp)

    theta = _main(p.trial_params(0))
    print(f"Final angle parameters: {theta}")
