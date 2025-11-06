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
from jax.interpreters.partial_eval import convert_constvars_jaxpr
from pennylane.capture.primitives import cond_prim as plxpr_cond_prim
from pennylane.capture.primitives import for_loop_prim as plxpr_for_loop_prim
from pennylane.capture.primitives import while_loop_prim as plxpr_while_loop_prim

from catalyst.from_plxpr.from_plxpr import (
    PLxPRToQuantumJaxprInterpreter,
    WorkflowInterpreter,
    _tuple_to_slice,
)
from catalyst.from_plxpr.qubit_handler import QubitHandler, QubitIndexRecorder
from catalyst.jax_extras import jaxpr_pad_consts
from catalyst.jax_primitives import cond_p, for_p, while_p


def _calling_convention(interpreter, closed_jaxpr, *args_plus_qreg):
    # The last arg is the scope argument for the body jaxpr
    *args, qreg = args_plus_qreg

    # Launch a new interpreter for the body region
    # A new interpreter's root qreg value needs a new recorder
    converter = copy(interpreter)
    converter.qubit_index_recorder = QubitIndexRecorder()
    init_qreg = QubitHandler(qreg, converter.qubit_index_recorder)
    converter.init_qreg = init_qreg

    retvals = converter(closed_jaxpr, *args)
    init_qreg.insert_all_dangling_qubits()
    return *retvals, converter.init_qreg.get()


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
        nimplicit_outputs=0,
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
    args_plus_qreg = [*args, self.init_qreg.get()]  # Add the qreg to the args
    converted_jaxpr_branches = []
    all_consts = []

    # Convert each branch from plxpr to jaxpr
    for const_slice, plxpr_branch in zip(consts_slices, jaxpr_branches):

        # Store all branches consts in a flat list
        branch_consts = plxpr_invals[_tuple_to_slice(const_slice)]

        converted_jaxpr_branch = None
        closed_jaxpr = ClosedJaxpr(plxpr_branch, branch_consts)

        f = partial(_calling_convention, self, closed_jaxpr)
        converted_jaxpr_branch = jax.make_jaxpr(f)(*args_plus_qreg)

        all_consts += converted_jaxpr_branch.consts
        converted_jaxpr_branches.append(converted_jaxpr_branch.jaxpr)

    predicate = [_to_bool_if_not(p) for p in plxpr_invals[: len(jaxpr_branches) - 1]]

    # Build Catalyst compatible input values
    cond_invals = [*predicate, *all_consts, *args_plus_qreg]

    # Perform the binding
    outvals = cond_p.bind(
        *cond_invals,
        branch_jaxprs=jaxpr_pad_consts(converted_jaxpr_branches),
        nimplicit_outputs=None,
    )

    # We assume the last output value is the returned qreg.
    # Update the current qreg and remove it from the output values.
    self.init_qreg.set(outvals.pop())

    # Return only the output values that match the plxpr output values
    return outvals


# pylint: disable=unused-argument, too-many-arguments
@WorkflowInterpreter.register_primitive(plxpr_for_loop_prim)
def workflow_for_loop(
    self,
    start,
    stop,
    step,
    *plxpr_invals,
    jaxpr_body_fn,
    consts_slice,
    args_slice,
    abstract_shapes_slice,
):
    """Handle the conversion from plxpr to Catalyst jaxpr for the for loop primitive

    Args:
        consts_slice: Tuple (start, stop, step) to slice consts from plxpr_invals
        args_slice: Tuple (start, stop, step) to slice args from plxpr_invals
        abstract_shapes_slice: Tuple (start, stop, step) to slice abstract shapes
    """
    assert jaxpr_body_fn is not None
    args = plxpr_invals[_tuple_to_slice(args_slice)]

    consts = plxpr_invals[_tuple_to_slice(consts_slice)]

    converter = copy(self)
    evaluator = partial(converter.eval, jaxpr_body_fn, consts)

    converted_jaxpr_branch = jax.make_jaxpr(evaluator)(start, *args)
    converted_closed_jaxpr_branch = ClosedJaxpr(
        convert_constvars_jaxpr(converted_jaxpr_branch.jaxpr), ()
    )

    # Config additional for loop settings
    apply_reverse_transform = isinstance(step, int) and step < 0

    return for_p.bind(
        *converted_jaxpr_branch.consts,
        start,
        stop,
        step,
        start,
        *args,
        body_jaxpr=converted_closed_jaxpr_branch,
        body_nconsts=len(consts),
        apply_reverse_transform=apply_reverse_transform,
        nimplicit=0,
        preserve_dimensions=True,
    )


# pylint: disable=unused-argument, too-many-arguments
@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_for_loop_prim)
def handle_for_loop(
    self,
    start,
    stop,
    step,
    *plxpr_invals,
    jaxpr_body_fn,
    consts_slice,
    args_slice,
    abstract_shapes_slice,
):
    """Handle the conversion from plxpr to Catalyst jaxpr for the for loop primitive

    Args:
        consts_slice: Tuple (start, stop, step) to slice consts from plxpr_invals
        args_slice: Tuple (start, stop, step) to slice args from plxpr_invals
        abstract_shapes_slice: Tuple (start, stop, step) to slice abstract shapes
    """
    assert jaxpr_body_fn is not None
    args = plxpr_invals[_tuple_to_slice(args_slice)]

    # Add the iteration start and the qreg to the args
    self.init_qreg.insert_all_dangling_qubits()
    start_plus_args_plus_qreg = [
        start,
        *args,
        self.init_qreg.get(),
    ]

    consts = plxpr_invals[_tuple_to_slice(consts_slice)]

    jaxpr = ClosedJaxpr(jaxpr_body_fn, consts)

    f = partial(_calling_convention, self, jaxpr)
    converted_jaxpr_branch = jax.make_jaxpr(f)(*start_plus_args_plus_qreg)

    converted_closed_jaxpr_branch = ClosedJaxpr(
        convert_constvars_jaxpr(converted_jaxpr_branch.jaxpr), ()
    )

    # Build Catalyst compatible input values
    new_consts = converted_jaxpr_branch.consts
    for_loop_invals = [*new_consts, start, stop, step, *start_plus_args_plus_qreg]

    # Config additional for loop settings
    apply_reverse_transform = isinstance(step, int) and step < 0

    # Perform the binding
    outvals = for_p.bind(
        *for_loop_invals,
        body_jaxpr=converted_closed_jaxpr_branch,
        body_nconsts=len(new_consts),
        apply_reverse_transform=apply_reverse_transform,
        nimplicit=0,
        preserve_dimensions=True,
    )

    # We assume the last output value is the returned qreg.
    # Update the current qreg and remove it from the output values.
    self.init_qreg.set(outvals.pop())

    # Return only the output values that match the plxpr output values
    return outvals


# pylint: disable=too-many-arguments
@WorkflowInterpreter.register_primitive(plxpr_while_loop_prim)
def workflow_while_loop(
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
    consts_body = plxpr_invals[_tuple_to_slice(body_slice)]
    consts_cond = plxpr_invals[_tuple_to_slice(cond_slice)]
    args = plxpr_invals[_tuple_to_slice(args_slice)]

    evaluator_body = partial(copy(self).eval, jaxpr_body_fn, consts_body)
    new_body_jaxpr = jax.make_jaxpr(evaluator_body)(*args)
    evaluator_cond = partial(copy(self).eval, jaxpr_cond_fn, consts_cond)
    new_cond_jaxpr = jax.make_jaxpr(evaluator_cond)(*args)

    converted_body_closed_jaxpr_branch = ClosedJaxpr(
        convert_constvars_jaxpr(new_body_jaxpr.jaxpr), ()
    )
    converted_cond_closed_jaxpr_branch = ClosedJaxpr(
        convert_constvars_jaxpr(new_cond_jaxpr.jaxpr), ()
    )
    # Build Catalyst compatible input values
    while_loop_invals = [*new_cond_jaxpr.consts, *new_body_jaxpr.consts, *args]

    return while_p.bind(
        *while_loop_invals,
        cond_jaxpr=converted_cond_closed_jaxpr_branch,
        body_jaxpr=converted_body_closed_jaxpr_branch,
        cond_nconsts=len(new_cond_jaxpr.consts),
        body_nconsts=len(new_body_jaxpr.consts),
        nimplicit=0,
        preserve_dimensions=True,
    )


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
    consts_body = plxpr_invals[_tuple_to_slice(body_slice)]
    consts_cond = plxpr_invals[_tuple_to_slice(cond_slice)]
    args = plxpr_invals[_tuple_to_slice(args_slice)]
    args_plus_qreg = [*args, self.init_qreg.get()]  # Add the qreg to the args

    jaxpr = ClosedJaxpr(jaxpr_body_fn, consts_body)

    f = partial(_calling_convention, self, jaxpr)
    converted_body_jaxpr_branch = jax.make_jaxpr(f)(*args_plus_qreg)
    new_consts_body = converted_body_jaxpr_branch.consts
    converted_body_closed_jaxpr_branch = ClosedJaxpr(
        convert_constvars_jaxpr(converted_body_jaxpr_branch.jaxpr), ()
    )

    # Convert for condition from plxpr to Catalyst jaxpr
    # We need to be able to handle arbitrary plxpr here.
    # But we want to be able to create a state where:
    # * We do not pass the quantum register as an argument.

    # So let's just remove the quantum register here at the end

    jaxpr = ClosedJaxpr(jaxpr_cond_fn, consts_cond)

    def remove_qreg(*args_plus_qreg):
        # The last arg is the scope argument for the body jaxpr
        *args, qreg = args_plus_qreg

        # Launch a new interpreter for the body region
        # A new interpreter's root qreg value needs a new recorder
        converter = copy(self)
        converter.qubit_index_recorder = QubitIndexRecorder()
        init_qreg = QubitHandler(qreg, converter.qubit_index_recorder)
        converter.init_qreg = init_qreg

        return converter(jaxpr, *args)

    converted_cond_jaxpr_branch = jax.make_jaxpr(remove_qreg)(*args_plus_qreg)
    converted_cond_closed_jaxpr_branch = ClosedJaxpr(
        convert_constvars_jaxpr(converted_cond_jaxpr_branch.jaxpr), ()
    )

    new_consts_cond = converted_cond_jaxpr_branch.consts
    # Build Catalyst compatible input values
    while_loop_invals = [*new_consts_cond, *new_consts_body, *args_plus_qreg]

    # Perform the binding
    outvals = while_p.bind(
        *while_loop_invals,
        cond_jaxpr=converted_cond_closed_jaxpr_branch,
        body_jaxpr=converted_body_closed_jaxpr_branch,
        cond_nconsts=len(new_consts_cond),
        body_nconsts=len(new_consts_body),
        nimplicit=0,
        preserve_dimensions=True,
    )

    # We assume the last output value is the returned qreg.
    # Update the current qreg and remove it from the output values.
    self.init_qreg.set(outvals.pop())

    # Return only the output values that match the plxpr output values
    return outvals
