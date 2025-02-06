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
"""ChemVQE problem, PennyLane+Catalyst implementation"""

from dataclasses import dataclass

import jax.numpy as jnp
import pennylane as qml
from catalyst_benchmark.types import Problem
from jax.core import ShapedArray

from catalyst import for_loop, qjit


@dataclass
class ProblemC(Problem):
    """Catalyst implementation details of the VQE problem"""

    def __init__(self, dev, nlayers=None, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        self.nlayers = nlayers if nlayers else 1
        self.qcircuit = None

    def trial_params(self, i: int):
        return jnp.array([1.0 / (2.0 * (1 + i) * jnp.pi)], dtype=jnp.float64)


def qcompile(p: ProblemC, params):
    """Compile the quantum parts of the problem"""

    def _circuit(params: ShapedArray(params.shape, dtype=jnp.complex64)):
        # pylint: disable=no-value-for-parameter
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
