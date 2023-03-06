from typing import Any

import pennylane.numpy as pnp
import pennylane as qml
import numpy as np

from .types import Problem

symbols = ["H", "H", "H"]
coordinates = np.array([0.028, 0.054, 0.0, 0.986, 1.610, 0.0, 1.855, 0.002, 0.0])


class ProblemVQE(Problem):
    def __init__(self, dev, ntrials=10, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        self.params = pnp.array([0.0, 0.0], requires_grad=True)
        self.ntrials = ntrials
        self.hf = qml.qchem.hf_state(electrons=2, orbitals=self.nqubits)

        # Building the molecular hamiltonian for the trihydrogen cation
        self.hamiltonian, self.molqubits = qml.qchem.molecular_hamiltonian(
            symbols, coordinates, charge=1
        )
        assert (
            self.nqubits >= self.molqubits
        ), f"self.nqubits ({self.nqubits}) >= self.molqubits ({self.molqubits})"

    def trial_params(self, i: int) -> Any:
        return pnp.array([])


def grad_descent(p: ProblemVQE, _):
    @qml.qnode(p.dev, **p.qnode_kwargs)
    def cost_func(params):
        qml.BasisState(p.hf, wires=list(range(p.nqubits)))
        qml.DoubleExcitation(params[0], wires=[0, 1, 2, 3])
        qml.DoubleExcitation(params[1], wires=[0, 1, 4, 5])
        return qml.expval(qml.Hamiltonian(np.array(p.hamiltonian.coeffs), p.hamiltonian.ops))

    stepsize = 0.4
    theta = p.params
    diff = qml.grad(cost_func, argnum=0)

    for i in range(p.ntrials):
        h = diff(theta)[0]
        theta = theta - h * stepsize

    return theta


def run_default(N=6):
    """Test run function"""
    p = ProblemVQE(qml.device("default.qubit", wires=N), ntrials=2)

    def _main(params):
        return grad_descent(p, params)

    theta = _main(p.trial_params(0))

    print(f"Final angle parameters: {theta}")


def run_jax(N=6):
    """Test run function"""
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_array", True)

    p = ProblemVQE(dev=qml.device("lightning.qubit", wires=N), ntrials=2, interface="jax")

    @jax.jit
    def _main(params):
        return grad_descent(p, params)

    theta = _main(p.trial_params(0))

    print(f"Final angle parameters: {theta}")

