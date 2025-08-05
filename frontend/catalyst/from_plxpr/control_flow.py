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

from catalyst.from_plxpr.from_plxpr import PLxPRToQuantumJaxprInterpreter, WorkflowInterpreter
from catalyst.from_plxpr.qreg_manager import QregManager
from catalyst.jax_extras import jaxpr_pad_consts
from catalyst.jax_primitives import cond_p, for_p, while_p


@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_cond_prim)
def handle_cond(self, *plxpr_invals, jaxpr_branches, consts_slices, args_slice):
    """Handle the conversion from plxpr to Catalyst jaxpr for the cond primitive"""
    args = plxpr_invals[args_slice]
    self.qreg_manager.insert_all_dangling_qubits()
    args_plus_qreg = [*args, self.qreg_manager.get()]  # Add the qreg to the args
    converted_jaxpr_branches = []
    all_consts = []

    # Convert each branch from plxpr to jaxpr
    for const_slice, plxpr_branch in zip(consts_slices, jaxpr_branches):

        # Store all branches consts in a flat list
        branch_consts = plxpr_invals[const_slice]
        all_consts = all_consts + [*branch_consts]

        converted_jaxpr_branch = None

        if plxpr_branch is None:
            # Emit a new Catalyst jaxpr branch that simply returns a qreg
            converted_jaxpr_branch = jax.make_jaxpr(lambda x: x)(*args_plus_qreg).jaxpr
        else:

            closed_jaxpr = ClosedJaxpr(plxpr_branch, branch_consts)

            def calling_convention(*args_plus_qreg):
                *args, qreg = args_plus_qreg
                # `qreg` is the scope argument for the body jaxpr
                qreg_manager = QregManager(qreg)
                converter = copy(self)
                converter.qreg_manager = qreg_manager
                # pylint: disable-next=cell-var-from-loop
                retvals = converter(closed_jaxpr, *args)
                qreg_manager.insert_all_dangling_qubits()
                return *retvals, converter.qreg_manager.get()

            converted_jaxpr_branch = jax.make_jaxpr(calling_convention)(*args_plus_qreg).jaxpr

        converted_jaxpr_branches.append(converted_jaxpr_branch)

    # The slice [0,1) of the plxpr input values contains the true predicate of the plxpr cond,
    # whereas the slice [1,2) refers to the false predicate, which is always True.
    # We extract the true predicate and discard the false one.
    predicate_slice = slice(0, 1)
    predicate = plxpr_invals[predicate_slice]

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
    self.qreg_manager.set(outvals.pop())

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
    """Handle the conversion from plxpr to Catalyst jaxpr for the for loop primitive"""
    assert jaxpr_body_fn is not None
    args = plxpr_invals[args_slice]

    consts = plxpr_invals[consts_slice]

    converter = copy(self)
    evaluator = partial(converter.eval, jaxpr_body_fn, consts)

    converted_jaxpr_branch = jax.make_jaxpr(evaluator)(start, *args).jaxpr
    converted_closed_jaxpr_branch = ClosedJaxpr(convert_constvars_jaxpr(converted_jaxpr_branch), ())

    # Config additional for loop settings
    apply_reverse_transform = isinstance(step, int) and step < 0

    return for_p.bind(
        *consts,
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
    """Handle the conversion from plxpr to Catalyst jaxpr for the for loop primitive"""
    assert jaxpr_body_fn is not None
    args = plxpr_invals[args_slice]

    # Add the iteration start and the qreg to the args
    self.qreg_manager.insert_all_dangling_qubits()
    start_plus_args_plus_qreg = [
        start,
        *args,
        self.qreg_manager.get(),
    ]

    consts = plxpr_invals[consts_slice]

    jaxpr = ClosedJaxpr(jaxpr_body_fn, consts)

    def calling_convention(*args_plus_qreg):
        *args, qreg = args_plus_qreg
        # `qreg` is the scope argument for the body jaxpr
        qreg_manager = QregManager(qreg)
        converter = copy(self)
        converter.qreg_manager = qreg_manager
        retvals = converter(jaxpr, *args)
        qreg_manager.insert_all_dangling_qubits()
        return *retvals, converter.qreg_manager.get()

    converted_jaxpr_branch = jax.make_jaxpr(calling_convention)(*start_plus_args_plus_qreg).jaxpr
    converted_closed_jaxpr_branch = ClosedJaxpr(convert_constvars_jaxpr(converted_jaxpr_branch), ())

    # Build Catalyst compatible input values
    for_loop_invals = [*consts, start, stop, step, *start_plus_args_plus_qreg]

    # Config additional for loop settings
    apply_reverse_transform = isinstance(step, int) and step < 0

    # Perform the binding
    outvals = for_p.bind(
        *for_loop_invals,
        body_jaxpr=converted_closed_jaxpr_branch,
        body_nconsts=len(consts),
        apply_reverse_transform=apply_reverse_transform,
        nimplicit=0,
        preserve_dimensions=True,
    )

    # We assume the last output value is the returned qreg.
    # Update the current qreg and remove it from the output values.
    self.qreg_manager.set(outvals.pop())

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
    """Handle the conversion from plxpr to Catalyst jaxpr for the while loop primitive"""
    consts_body = plxpr_invals[body_slice]
    consts_cond = plxpr_invals[cond_slice]
    args = plxpr_invals[args_slice]

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
    while_loop_invals = [*consts_cond, *consts_body, *args]

    return while_p.bind(
        *while_loop_invals,
        cond_jaxpr=converted_cond_closed_jaxpr_branch,
        body_jaxpr=converted_body_closed_jaxpr_branch,
        cond_nconsts=len(consts_cond),
        body_nconsts=len(consts_body),
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
    """Handle the conversion from plxpr to Catalyst jaxpr for the while loop primitive"""
    self.qreg_manager.insert_all_dangling_qubits()
    consts_body = plxpr_invals[body_slice]
    consts_cond = plxpr_invals[cond_slice]
    args = plxpr_invals[args_slice]
    args_plus_qreg = [*args, self.qreg_manager.get()]  # Add the qreg to the args

    jaxpr = ClosedJaxpr(jaxpr_body_fn, consts_body)

    def calling_convention(*args_plus_qreg):
        *args, qreg = args_plus_qreg
        # `qreg` is the scope argument for the body jaxpr
        qreg_manager = QregManager(qreg)
        converter = copy(self)
        converter.qreg_manager = qreg_manager
        retvals = converter(jaxpr, *args)
        qreg_manager.insert_all_dangling_qubits()
        return *retvals, converter.qreg_manager.get()

    converted_body_jaxpr_branch = jax.make_jaxpr(calling_convention)(*args_plus_qreg).jaxpr
    converted_body_closed_jaxpr_branch = ClosedJaxpr(
        convert_constvars_jaxpr(converted_body_jaxpr_branch), ()
    )

    # Convert for condition from plxpr to Catalyst jaxpr
    # We need to be able to handle arbitrary plxpr here.
    # But we want to be able to create a state where:
    # * We do not pass the quantum register as an argument.

    # So let's just remove the quantum register here at the end

    jaxpr = ClosedJaxpr(jaxpr_cond_fn, consts_cond)

    def remove_qreg(*args_plus_qreg):
        *args, qreg = args_plus_qreg
        # `qreg` is the scope argument for the body jaxpr
        qreg_manager = QregManager(qreg)
        converter = copy(self)
        converter.qreg_manager = qreg_manager

        return converter(jaxpr, *args)

    converted_cond_jaxpr_branch = jax.make_jaxpr(remove_qreg)(*args_plus_qreg).jaxpr
    converted_cond_closed_jaxpr_branch = ClosedJaxpr(
        convert_constvars_jaxpr(converted_cond_jaxpr_branch), ()
    )

    # Build Catalyst compatible input values
    while_loop_invals = [*consts_cond, *consts_body, *args_plus_qreg]

    # Perform the binding
    outvals = while_p.bind(
        *while_loop_invals,
        cond_jaxpr=converted_cond_closed_jaxpr_branch,
        body_jaxpr=converted_body_closed_jaxpr_branch,
        cond_nconsts=len(consts_cond),
        body_nconsts=len(consts_body),
        nimplicit=0,
        preserve_dimensions=True,
    )

    # We assume the last output value is the returned qreg.
    # Update the current qreg and remove it from the output values.
    self.qreg_manager.set(outvals.pop())

    # Return only the output values that match the plxpr output values
    return outvals
