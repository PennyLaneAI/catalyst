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
""" ChemVQE problem, PennyLane/PennyLane+JAX implementation """

# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-instance-attributes

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from catalyst_benchmark.types import Problem
from pennylane import AllSinglesDoubles


@dataclass
class ProblemInfo:
    """ChemVQE problem specification"""

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


@dataclass
class ProblemCVQE(Problem):
    """PennyLane implementation details of the VQE problem"""

    def __init__(self, dev, diff_method, grad, nsteps=10, **qnode_kwargs):
        super().__init__(dev, **qnode_kwargs)
        self.nsteps = nsteps
        self.grad = grad
        self.diff_method = diff_method
        pi = PROBLEMS[self.nqubits]
        data = qml.data.load("qchem", molname=pi.name, basis="STO-3G", bondlength=pi.bond)[0]
        electrons = data.molecule.n_electrons
        qubits = data.molecule.n_orbitals * 2
        assert qubits == self.nqubits
        ham = data.hamiltonian
        self.hf_state = qml.qchem.hf_state(electrons, qubits)
        self.singles, self.doubles = qml.qchem.excitations(electrons, qubits)
        self.excitations = self.singles + self.doubles
        self.ham = ham
        self.qcircuit = None
        self.qgrad = None

    def trial_params(self, _: int) -> Any:
        return pnp.zeros(len(self.excitations), dtype=pnp.float64)


def qcompile_hybrid(p: ProblemCVQE, weights):
    """Compile the quantum parts of the problem"""

    def _circuit(params):
        AllSinglesDoubles(params, range(p.nqubits), p.hf_state, p.singles, p.doubles)
        return qml.expval(qml.Hamiltonian(np.array(p.ham.coeffs), p.ham.ops))

    qcircuit = qml.QNode(_circuit, p.dev, diff_method=p.diff_method, **p.qnode_kwargs)
    qcircuit.construct([weights], {})
    p.qcircuit = qcircuit


def qcompile(p: ProblemCVQE, weights):
    """Compile the quantum parts of the problem"""
    qcompile_hybrid(p, weights)
    qgrad = p.grad(p.qcircuit)
    p.qgrad = qgrad


def workflow(p: ProblemCVQE, params):
    """Problem workflow"""
    assert p.qcircuit is not None
    assert p.qgrad is not None

    stepsize = 0.5
    theta = params

    for _ in range(p.nsteps):
        dtheta = p.qgrad(theta)
        theta = theta - dtheta * stepsize
    return theta


def workflow_hybrid(p: ProblemCVQE, params):
    """Problem workflow"""
    assert p.qcircuit is not None

    stepsize = 0.5
    theta = params

    for _ in range(p.nsteps):
        dtheta = p.qcircuit(theta)
        theta = theta - pnp.mean(dtheta) * stepsize
    return theta


def size(p: ProblemCVQE) -> int:
    """Compute the size of the problem circuit"""
    with qml.tape.QuantumTape() as tape:
        AllSinglesDoubles(p.trial_params(0), range(p.nqubits), p.hf_state, p.singles, p.doubles)
    return len(qml.transforms.create_expand_fn(depth=5, device=p.dev)(tape))


SHOTS = None
DIFFMETHOD = "parameter-shift"
NSTEPS = 1


def run_default_qubit(N=6):
    """Test problem entry point"""
    p = ProblemCVQE(
        qml.device("default.qubit", wires=N, shots=SHOTS),
        grad=partial(qml.grad, argnum=0),
        nsteps=NSTEPS,
        diff_method=DIFFMETHOD,
    )
    print(f"Size: {size(p)}")

    def _main(params):
        qcompile_hybrid(p, params)
        return workflow_hybrid(p, params)

    theta = _main(p.trial_params(0))
    print(f"Resulting parameters: {theta}")


def run_lightning_qubit(N=6):
    """Test problem entry point"""
    p = ProblemCVQE(
        qml.device("lightning.qubit", wires=N, shots=SHOTS),
        grad=partial(qml.grad, argnum=0),
        nsteps=NSTEPS,
        diff_method=DIFFMETHOD,
    )
    print(f"Size: {size(p)}")

    def _main(params):
        qcompile(p, params)
        return workflow(p, params)

    theta = _main(p.trial_params(0))
    print(f"Resulting parameters: {theta}")


def run_jax_(devname, N=6):
    """Test problem entry point"""
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_array", True)

    p = ProblemCVQE(
        dev=qml.device(devname, wires=N, shots=SHOTS),
        grad=jax.grad,
        nsteps=NSTEPS,
        interface="jax",
        diff_method=DIFFMETHOD,
    )
    print(f"Size: {size(p)}")

    @jax.jit
    def _main(params):
        qcompile_hybrid(p, params)
        return workflow_hybrid(p, params)

    theta = _main(p.trial_params(0))
    print(f"Resulting parameters: {theta}")


def run_jax_default_qubit(N=6):
    """Test problem entry point"""
    return run_jax_("default.qubit.jax", N)


def run_jax_lightning_qubit(N=6):
    """Test problem entry point"""
    return run_jax_("lightning.qubit", N)
