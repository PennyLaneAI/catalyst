""" ChemVQE problem, PennyLane+Catalyst implementation """
from dataclasses import dataclass

import pennylane as qml
import jax.numpy as jnp
from catalyst import qjit, for_loop
from jax.core import ShapedArray
from .types import Problem


@dataclass
class ProblemC(Problem):
    """Catalyst implementation details of the VQE problem"""

    def __init__(self, dev, nlayers=None, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        self.nlayers = nlayers if nlayers else 1
        self.qcircuit = None

    def trial_params(self, n: int):
        return jnp.array([1.0 / (2.0 * (1 + n) * jnp.pi)], dtype=jnp.float64)


def qcompile(p: ProblemC, params):
    """Compile the quantum parts of the problem"""

    def _circuit(params: ShapedArray(params.shape, dtype=jnp.complex64)):
        N = p.nqubits
        L = p.nlayers

        @for_loop(0, N, 1)
        def loop(i):
            qml.Hadamard(wires=i)

        loop()

        @for_loop(0, L, 1)
        def loop0(_):
            @for_loop(0, N, 1)
            def loop1(wG):
                @for_loop(wG + 1, N, 1)
                def loop2(wC):
                    phi = (2 ** (wC - wG)) / params[0]
                    qml.ctrl(qml.PhaseShift, control=wC)(phi=phi, wires=[wG])

                loop2()

            loop1()

        loop0()
        return qml.state()

    qcircuit = qml.QNode(_circuit, p.dev, **p.qnode_kwargs)
    # qcircuit.construct([params], {})
    # ^^^ AttributeError: 'AnnotatedQueue' object has no attribute 'jax_tape'
    p.qcircuit = qcircuit
    return p


def workflow(p: ProblemC, params):
    """Problem workflow"""
    return p.qcircuit(params)


SHOTS = None


def run_catalyst(N=6):
    """Test problem entry point"""
    p = ProblemC(
        dev=qml.device("lightning.qubit", wires=N, shots=SHOTS),
    )
    params = p.trial_params(0)

    @qjit
    def _main(params: ShapedArray(params.shape, params.dtype)):
        qcompile(p, params)
        return workflow(p, params)

    result = _main(p.trial_params(0))
    print(f"Result: {result}")
