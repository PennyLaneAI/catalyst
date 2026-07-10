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
This module provides infrastructure for compile-time lowering of decomposition rules via python.

Python decomposition wrappers should adhere to the following specifications:
    The wrapper:
        - Is named `{op name}_decomposition_wrapper`.
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


def python_decomposition_wrapper(op_name, op_id, dynamic_shape, wire_lens, static_data) -> str:
    """Generic decomposition wrapper."""
    device = qp.device("null.qubit", wires=sum(wire_lens))
    wires = tuple(jnp.array(range(length)) for length in wire_lens)

    def rule_to_subroutine(rule):
        def decomp_rule(*params, wires):
            rule._impl(*params, *wires, **static_data)

        # TODO remove this once we have unified lowering, we should be able to set target_gate and
        # stop relying on function names
        decomp_rule.__name__ = op_id + "_" + rule.name

        return qp.capture.subroutine(decomp_rule)

    # let this fail with the standard error message if the op is not found
    op_class = getattr(qp, op_name)

    subroutines = [rule_to_subroutine(rule) for rule in qp.decomposition.list_decomps(op_class)]

    @qp.qjit(
        target="mlir",
        capture=True,
    )
    @qp.qnode(device=device)
    def circuit():
        for subroutine in subroutines:
            # TODO I know this is dynamic, but we should probably have a better way of handling this
            # than hard-coded dummy values. Revisit this when unifying the decomp-rule lowering
            # pipeline
            subroutine(*[0.5 for _ in dynamic_shape], wires=wires)

    return str(circuit.mlir_module)
