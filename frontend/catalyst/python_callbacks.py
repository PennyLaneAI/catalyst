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

"""
This module provides infrastructure for lowering decomposition rules using compiler callbacks.

Callback functions should adhere to the following specifications:
    The callback (python) function:
        - Is named `{op name}_callback_wrapper`.
        - Has a signature identical to the named parameters of the associated PL operator; dynamic
          arguments may be unused, but should still be included for compatibility.
        - Is able to AOT lower the decomposition rule to MLIR without invoking the compiler, e.g.
          using `target="mlir"`, AOT compilation and `QJIT.mlir_module`.
          See existing examples for further information.
        - Returns a string representation of an MLIR module, containing a FuncOp which represents
          the instantiated decomposition rule.

    The FuncOp decomposition rule in the returned string:
        - Is named `{op name}_decomp_rule`.
        - Is an MLIR representation of the PennyLane decomposition rule associated with the
          specified operator.
        - Is instantiated with the static data provided, and all other data remains dynamic.
        - Is compatible with the `decompose-lowering` pass, i.e. can be mapped to the MLIR operation
          it decomposes and inlined.
        - Is self-contained, and does not contain any device initialization, setup/teardown etc.
"""

# pylint: disable=protected-access,unused-argument

import jax.numpy as jnp
import pennylane as qp


def paulirot_callback_wrapper(theta, pauli_word, wires):
    """Wraps the paulirot decomp rule for compile-time lowering with a static pauli word.

    The decomposition rule is identifiable by the name `paulirot_decomp_rule`.
    """
    device = qp.device("null.qubit", wires=len(wires))
    wires = jnp.array(wires)

    def paulirot_decomp_rule(theta, wires):
        qp.ops.qubit.parametric_ops_multi_qubit._pauli_rot_decomposition._impl(
            theta, wires, pauli_word
        )

    paulirot_subroutine = qp.capture.subroutine(paulirot_decomp_rule)

    @qp.qjit(
        target="mlir",
        capture=True,
    )
    @qp.qnode(device=device)
    def circuit(theta: float):
        paulirot_subroutine(theta, wires)

    return str(circuit.mlir_module)
