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

"""This module provides infrastructure for lowering decomposition rules using compiler callbacks."""

import uuid
from typing import Callable, Dict

import jax.numpy as jnp
import pennylane as qp


def register_stopping_condition(cond: Callable[[qp.operation.Operator], bool]) -> str:
    """Register a stopping_condition predicate and return its UUID for the IR.
    """
    cid = str(uuid.uuid4())
    _STOPPING_CONDITIONS[cid] = cond
    return cid


def _reconstruct_op(op_name: str, params, wires) -> qp.operation.Operator:
    """TODO..."""

    cls = getattr(qp.ops, op_name, None) or getattr(qp, op_name, None)
    if cls is None:
        # Unknown op: predicate will likely reject. We pass through the name
        # via a generic Operator so any predicate that only inspects .name
        # still gets a sensible answer.
        raise ValueError(f"stopping_condition: unknown op name '{op_name}'")
    # 
    if params:
        # TODO: Op 2.0
        raise NotImplementedError("stopping_condition: parametric ops not yet supported in predicates")
    
    return cls(wires=list(wires))

_STOPPING_CONDITIONS: Dict[str, Callable[[qp.operation.Operator], bool]] = {}

def stopping_condition_wrapper(cond_id: str, op_name: str, params, wires) -> bool:
    """Bridge invoked from C++ per op-instance during graph decomposition.
    """
    cond = _STOPPING_CONDITIONS.get(cond_id)
    if cond is None:
        # Unknown condition:
        return False
    try:
        op = _reconstruct_op(op_name, list(params), list(wires))
    except Exception:
        return False
    return bool(cond(op))


#############################################
# PauliRot decomposition lowering
##############################################

def paulirot_callback_wrapper(theta, pauli_word, wires):
    """Wraps paulirot decomp rule to enable compile-time lowering."""
    # pylint: disable=protected-access
    print("running paulirot callbacks...")
    device = qp.device("null.qubit", wires=len(wires))

    @qp.qjit(
        target="mlir",
        capture=True,
        static_argnums=2,
    )
    @qp.qnode(device=device)
    def circuit(theta, wires, pauli_word):
        # declare subroutine
        my_subroutine = qp.capture.subroutine(
            qp.ops.qubit.parametric_ops_multi_qubit._pauli_rot_decomposition._impl, static_argnums=2
        )

        # call subroutine
        my_subroutine(theta, wires, pauli_word)

    circuit(theta, jnp.array(wires), pauli_word)
    return str(circuit.mlir)


def test_function():
    print("running python...")
    return "hello from python!"
