# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests that the ``graph-decomposition`` pass decomposes ``qp.TrotterCDF`` into a target
gate set, using the decomposition rules collected from its numeric Hamiltonian.
"""

# pylint: disable = missing-function-docstring

# RUN: %PYTHON %s | FileCheck %s

import numpy as np
import pennylane as qp

import catalyst

N, L = 2, 1  # N orbitals -> 2N wires, L two-body fragments

rng = np.random.default_rng(42)


def random_orthogonal(dim):
    """A real orthogonal matrix, as the Trotter leaf tensors require."""
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))


CDF = qp.CDFHamiltonian(
    core_tensors=rng.standard_normal((L + 1, N, N)),
    leaf_tensors=np.stack([random_orthogonal(N) for _ in range(L + 1)]),
    nuc_constant=0.5,
)

GATE_SET = {"BasisRotation", "RZ", "IsingZZ", "GlobalPhase"}


@qp.qjit(capture=True)
@catalyst.passes.graph_decomposition(gate_set=GATE_SET)
@qp.qnode(qp.device("lightning.qubit", wires=2 * N))
def trotter_circuit():
    qp.TrotterCDF(
        evolution_time=1.0,
        num_trotter_steps=10,
        hamiltonian=CDF,
        wires=range(2 * N),
    )
    return qp.state()


# CHECK-NOT: TrotterCDF
# CHECK: BasisRotation
# CHECK: IsingZZ
# CHECK: RZ
specs = qp.specs(trotter_circuit, level=0)()["resources"].quantum_operations
print(dict(sorted(specs.items())))
