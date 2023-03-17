""" ChemVQE problem, PennyLane+Catalyst implementation """
from copy import deepcopy

import pennylane as qml
import pennylane.numpy as pnp
from .types import Problem


class ProblemPL(Problem):
    """Catalyst implementation details of the VQE problem"""

    def __init__(self, dev, nlayers=None, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        self.qcircuit = None
        self.nlayers = nlayers if nlayers else 1

    def trial_params(self, n: int):
        return pnp.array([1.0 / (2.0 * (1 + n) * pnp.pi)], dtype=pnp.float64)


def qcompile(p: ProblemPL, params):
    """Compile the quantum parts of the problem"""

    def _circuit(params):
        N = p.nqubits
        L = p.nlayers
        for wG in range(N):
            qml.Hadamard(wires=[wG])
        for _ in range(L):
            for wG in range(N):
                for wC in range(wG + 1, N):
                    phi = (2 ** (wC - wG)) / params[0]
                    qml.ctrl(qml.PhaseShift, control=wC)(phi=phi, wires=[wG])
        return qml.state()

    qcircuit = qml.QNode(_circuit, p.dev, **p.qnode_kwargs)
    qcircuit.construct([params], {})
    p.qcircuit = qcircuit
    return p


def workflow(p: ProblemPL, params):
    """Problem workflow"""
    return p.qcircuit(params)


def size(p: ProblemPL) -> int:
    """Compute the size of the problem circuit"""
    qnode_kwargs = deepcopy(p.qnode_kwargs)
    qnode_kwargs.update({"expansion_strategy": "device"})
    qcompile(p, p.trial_params(0))
    return len(p.qcircuit.tape.operations)


SHOTS = None


def run_jax_lightning_qubit(N=6):
    """Test problem entry point"""

    import jax

    jax.config.update("jax_enable_x64", True)

    p = ProblemPL(dev=qml.device("lightning.qubit", wires=N, shots=SHOTS), interface="jax")

    @jax.jit
    def _main(params):
        qcompile(p, params)
        return workflow(p, params)

    params = p.trial_params(0)
    result = _main(params)
    print(f"Result: {result}")


def run_lightning_qubit(N=6):
    """Test problem entry point"""

    p = ProblemPL(dev=qml.device("lightning.qubit", wires=N, shots=SHOTS))

    def _main(params):
        qcompile(p, params)
        return workflow(p, params)

    params = p.trial_params(0)
    result = _main(params)
    print(f"Result: {result}")


def run_default_qubit(N=6):
    """Test problem entry point"""

    p = ProblemPL(dev=qml.device("default.qubit", wires=N, shots=SHOTS))

    def _main(params):
        qcompile(p, params)
        return workflow(p, params)

    params = p.trial_params(0)
    result = _main(params)
    print(f"Result: {result}")
