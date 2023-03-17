""" ChemVQE problem, PennyLane+Catalyst implementation """
import pennylane as qml
from catalyst import qjit
from jax.core import ShapedArray
from .qft_catalyst import ProblemC, qcompile


NSTEPS = 10


def workflow(p: ProblemC, params):
    """Problem workflow"""
    for _ in range(NSTEPS):
        state = p.qcircuit(params)[0]
        for v in state:
            params = params + v.real
    return state


SHOTS = None


def run_catalyst(N=6):
    """Test problem entry point"""
    p = ProblemC(
        dev=qml.device("lightning.qubit", wires=N, shots=SHOTS),
    )
    params = p.trial_params(0)
    print(params)

    @qjit
    def _main(params: ShapedArray(params.shape, params.dtype)):
        qcompile(p, params)
        return workflow(p, params)

    result = _main(p.trial_params(0))
    print(f"Result: {result}")
