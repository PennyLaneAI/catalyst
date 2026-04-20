# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Conversion from control flow plxpr primitives.
"""

from copy import copy
from functools import partial

import jax
from jax.extend.core import ClosedJaxpr
from pennylane.capture.primitives import cond_prim as plxpr_cond_prim
from pennylane.capture.primitives import while_loop_prim as plxpr_while_loop_prim

from catalyst.from_plxpr.from_plxpr import (
    PLxPRToQuantumJaxprInterpreter,
    WorkflowInterpreter,
    _tuple_to_slice,
)
from catalyst.from_plxpr.qubit_handler import (
    _get_dynamically_allocated_qregs,
)
from catalyst.jax_extras import jaxpr_pad_consts
from catalyst.jax_primitives import cond_p


def _calling_convention(interpreter, closed_jaxpr, *args):
    # Launch a new interpreter for the body region
    converter = copy(interpreter)
    retvals = converter(closed_jaxpr, *args)
    return retvals


def _to_bool_if_not(arg):
    if getattr(arg, "dtype", None) == jax.numpy.bool:
        return arg
    return jax.numpy.bool(arg)


@WorkflowInterpreter.register_primitive(plxpr_cond_prim)
def workflow_cond(self, *plxpr_invals, jaxpr_branches, consts_slices, args_slice):
    """Handle the conversion from plxpr to Catalyst jaxpr for the cond primitive

    Args:
        consts_slices: List of tuples (start, stop, step) to slice consts for each branch
        args_slice: Tuple (start, stop, step) to slice args from plxpr_invals
    """
    args = plxpr_invals[_tuple_to_slice(args_slice)]
    converted_jaxpr_branches = []
    all_consts = []

    # Convert each branch from plxpr to jaxpr
    for const_slice, plxpr_branch in zip(consts_slices, jaxpr_branches):

        # Store all branches consts in a flat list
        branch_consts = plxpr_invals[_tuple_to_slice(const_slice)]

        evaluator = partial(copy(self).eval, plxpr_branch, branch_consts)
        new_jaxpr = jax.make_jaxpr(evaluator)(*args)
        all_consts = all_consts + new_jaxpr.consts

        converted_jaxpr_branches.append(new_jaxpr.jaxpr)

    predicate = [_to_bool_if_not(p) for p in plxpr_invals[: len(jaxpr_branches) - 1]]

    # Build Catalyst compatible input values
    cond_invals = [*predicate, *all_consts, *args]

    return cond_p.bind(
        *cond_invals,
        branch_jaxprs=jaxpr_pad_consts(converted_jaxpr_branches),
        num_implicit_outputs=0,
    )


@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_cond_prim)
def handle_cond(self, *plxpr_invals, jaxpr_branches, consts_slices, args_slice):
    """Handle the conversion from plxpr to Catalyst jaxpr for the cond primitive

    Args:
        consts_slices: List of tuples (start, stop, step) to slice consts for each branch
        args_slice: Tuple (start, stop, step) to slice args from plxpr_invals
    """
    args = plxpr_invals[_tuple_to_slice(args_slice)]
    self.init_qreg.insert_all_dangling_qubits()

    dynalloced_qregs, dynalloced_wire_global_indices = _get_dynamically_allocated_qregs(
        plxpr_invals, self.qubit_index_recorder, self.init_qreg
    )

    # Add the qregs to the args
    args_plus_qreg = [
        *args,
        *[dyn_qreg.get() for dyn_qreg in dynalloced_qregs],
        self.init_qreg.get(),
    ]

    converted_jaxpr_branches = []
    all_consts = []

    # Convert each branch from plxpr to jaxpr
    for const_slice, plxpr_branch in zip(consts_slices, jaxpr_branches):

        # Store all branches consts in a flat list
        branch_consts = plxpr_invals[_tuple_to_slice(const_slice)]

        converted_jaxpr_branch = None
        closed_jaxpr = ClosedJaxpr(plxpr_branch, branch_consts)

        f = partial(
            _calling_convention, self, closed_jaxpr, outer_dynqreg_handlers=dynalloced_qregs
        )
        converted_jaxpr_branch = jax.make_jaxpr(f)(*args_plus_qreg)

        all_consts += converted_jaxpr_branch.consts
        converted_jaxpr_branches.append(converted_jaxpr_branch.jaxpr)

    predicate = [_to_bool_if_not(p) for p in plxpr_invals[: len(jaxpr_branches) - 1]]

    # Build Catalyst compatible input values
    # strip global wire indices of dynamic wires
    all_consts = tuple(const for const in all_consts if const not in dynalloced_wire_global_indices)
    cond_invals = [*predicate, *all_consts, *args_plus_qreg]

    # Perform the binding
    outvals = cond_p.bind(
        *cond_invals,
        branch_jaxprs=jaxpr_pad_consts(converted_jaxpr_branches),
        num_implicit_outputs=None,
    )

    # Output structure:
    # First a list of dynamically allocated qregs, then the global qreg
    # Update the current qreg and remove it from the output values.
    self.init_qreg.set(outvals.pop())
    for dyn_qreg in reversed(dynalloced_qregs):
        dyn_qreg.set(outvals.pop())

    # Return only the output values that match the plxpr output values
    return outvals


# pylint: disable=too-many-arguments
@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_while_loop_prim)
def handle_while_loop(
    self,
    *plxpr_invals,
    jaxpr_body_fn,
    jaxpr_cond_fn,
    body_slice,
    cond_slice,
    args_slice,
):
    """Handle the conversion from plxpr to Catalyst jaxpr for the while loop primitive

    Args:
        body_slice: Tuple (start, stop, step) to slice body consts from plxpr_invals
        cond_slice: Tuple (start, stop, step) to slice cond consts from plxpr_invals
        args_slice: Tuple (start, stop, step) to slice args from plxpr_invals
    """
    self.init_qreg.insert_all_dangling_qubits()
    dynalloced_qregs, dynalloced_wire_global_indices = _get_dynamically_allocated_qregs(
        plxpr_invals, self.qubit_index_recorder, self.init_qreg
    )
    consts_body = plxpr_invals[_tuple_to_slice(body_slice)]
    consts_cond = plxpr_invals[_tuple_to_slice(cond_slice)]
    args = plxpr_invals[_tuple_to_slice(args_slice)]
    args_plus_qreg = [
        *args,
        *[dyn_qreg.get() for dyn_qreg in dynalloced_qregs],
        self.init_qreg.get(),
    ]  # Add the qreg to the args

    jaxpr = ClosedJaxpr(jaxpr_body_fn, consts_body)

    f = partial(_calling_convention, self, jaxpr, outer_dynqreg_handlers=dynalloced_qregs)
    converted_body_jaxpr = jax.make_jaxpr(f)(*args_plus_qreg)
    new_consts_body = converted_body_jaxpr.consts

    # Convert for condition from plxpr to Catalyst jaxpr
    # We need to be able to handle arbitrary plxpr here.
    # But we want to be able to create a state where:
    # * We do not pass the quantum register as an argument.
    # So let's just remove the quantum register here at the end
    jaxpr = ClosedJaxpr(jaxpr_cond_fn, consts_cond)

    f_remove_qreg = partial(
        _calling_convention, self, jaxpr, outer_dynqreg_handlers=dynalloced_qregs, return_qreg=False
    )

    converted_cond_jaxpr = jax.make_jaxpr(f_remove_qreg)(*args_plus_qreg)

    # Build Catalyst compatible input values
    new_consts_cond = converted_cond_jaxpr.consts
    new_consts_body = tuple(
        const for const in new_consts_body if const not in dynalloced_wire_global_indices
    )
    while_loop_invals = [*new_consts_cond, *new_consts_body, *args_plus_qreg]
    cond_slice = (0, len(new_consts_cond), 1)
    body_slice = (cond_slice[1], len(new_consts_body) + cond_slice[1], 1)
    args_slice = (body_slice[1], None, 1)

    # Perform the binding
    outvals = plxpr_while_loop_prim.bind(
        *while_loop_invals,
        jaxpr_body_fn=converted_body_jaxpr.jaxpr,
        jaxpr_cond_fn=converted_cond_jaxpr.jaxpr,
        body_slice=body_slice,
        cond_slice=cond_slice,
        args_slice=args_slice,
    )
    outvals = list(outvals)

    # We assume the last output value is the returned qreg.
    # Update the current qreg and remove it from the output values.
    self.init_qreg.set(outvals.pop())

    for dyn_qreg in reversed(dynalloced_qregs):
        dyn_qreg.set(outvals.pop())

    # Return only the output values that match the plxpr output values
    return outvals
