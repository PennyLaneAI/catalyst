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


import pennylane as qp
from jax import numpy as jnp
from pennylane.typing import Float, Wire

from catalyst.decomposition.python_decompositions import (
    python_decomposition,
)


class Alice(qp.core.Operator2):
    arg_specs = {"wires": Wire[-1]}

    def __init__(self, wires):
        super().__init__(wires=wires)


class Bob(qp.core.Operator2):

    arg_specs = {"wires": Wire[1], "other_wires": Wire[2], "phi": Float, "thetas": Float[2]}
    dynamic_argnames = ("phi", "thetas")
    wire_argnames = ("wires", "other_wires")
    compilable_argnames = ("bob_word",)

    def __init__(self, wires, other_wires, phi, thetas, bob_word):
        super().__init__(
            wires=wires, other_wires=other_wires, phi=phi, thetas=thetas, bob_word=bob_word
        )


def _resource_fn(wires):
    return {
        Bob(Wire[1], Wire[2], Float, Float[2], "legen"): 1,
        Bob(Wire[1], Wire[2], Float, Float[2], "dary"): 1,
    }


@qp.register_resources(_resource_fn)
def _A2B(wires):
    Bob([2], [0, 1], 0.5, jnp.array([3.6, 3.4]), bob_word="legen")
    Bob([5], [3, 4], 1.5, jnp.array([9.1, 100.2]), bob_word="dary")


qp.add_decomps(Alice, _A2B)


# print(BFS_decomp_rules("Alice", {"wires":Wire[2]}))
print(python_decomposition("Alice", "Alice[][2]{}", [], [2], {"wires": Wire[2]}))
