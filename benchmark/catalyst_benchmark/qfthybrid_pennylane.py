""" ChemVQE problem, PennyLane+Catalyst implementation """
from copy import deepcopy

import pennylane as qml
import pennylane.numpy as pnp
from .types import Problem
from .qft_pennylane import ProblemPL, qcompile, size


NSTEPS = 10


def workflow(p: ProblemPL, params):
    """Problem workflow"""
    for _ in range(NSTEPS):
        state = p.qcircuit(params)
        for v in state:
            params = params + v.real
    return state


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
