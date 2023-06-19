# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the :class:`~.JaxTape`, a PennyLane :class:`~pennylane.QuantumTape`
that supports capturing classical computations and control flow of quantum operations
that occur within the circuit.
"""

import jax
import pennylane as qml
from jax.interpreters.partial_eval import (
    DynamicJaxprTrace,
    JaxprStackFrame,
    extend_jaxpr_stack,
)
from jax.tree_util import tree_flatten, tree_unflatten
from pennylane.tape import QuantumTape

from catalyst import jax_tracer
from catalyst.param_evaluator import ParamEvaluator


# pylint: disable=too-many-instance-attributes
class JaxTape:
    """A Quantum Tape that can additionally capture classical computations and control flow
    that occur within the circuit.
    """

    device = None

    def __init__(self, *args, **kwargs):
        # jax specific attributes
        self.main_cm = None
        self.frame = None
        self.extended_frame_cm = None
        self.reset_cm = None
        self.trace = None
        self.closed_jaxpr = None

        # everything else
        self.return_value = None
        self.output_trees = None
        self.quantum_tape = QuantumTape(*args, **kwargs)
        self.quantum_tape.jax_tape = self

    def __enter__(self):
        """
        Corresponds to the with statement that would look like:

        .. code-block:: python

            with jax.core.new_main(DynamicJaxprTrace, dynamic=True) as main:
                with extended_jaxpr_stack(main, frame):
                    with QuantumTape():
                        ...
        """
        # corresponds to `with jax.core.new_main(DynamicJaxprTrace, dynamic=True) as main:`
        self.main_cm = jax.core.new_main(DynamicJaxprTrace, dynamic=True)
        main = self.main_cm.__enter__()
        main.jaxpr_stack = ()

        self.frame = JaxprStackFrame()

        # corresponds to `with extend_jaxpr_stack(main, self.frame):`
        self.extended_frame_cm = extend_jaxpr_stack(main, self.frame)
        self.extended_frame_cm.__enter__()

        self.trace = DynamicJaxprTrace(main, jax.core.cur_sublevel())

        return self

    def __exit__(self, e_type, e_value, traceback):
        """
        Closes the context managers in the reverse ordering that they were opened in.

        That is,

        ..code-block:: python

            with jax.core.new_main(DynamicJaxprTrace, dynamic=True) as main:
                with extended_jaxpr_stack(main, frame):
        """

        if e_type is None:
            # no exception was raised

            def get_params_from_op_or_m_process(op):
                """
                Helper function to produce relevant parameters for tracing
                from different classes.
                """
                if isinstance(op, (qml.Hermitian, qml.QubitUnitary)):
                    # Can I subscript here? Or should I pass everything?
                    return op.data[0], op.wires.tolist()
                if isinstance(op, qml.Hamiltonian):
                    return op.coeffs
                if op.__class__.__name__ == "Cond":
                    return op.preds, op.consts
                if op.__class__.__name__ == "WhileLoop":
                    return op.cond_consts, op.body_consts, op.iter_args
                if op.__class__.__name__ == "ForLoop":
                    return op.loop_bounds, op.body_consts, op.iter_args
                # Named observables and other ops fall here.
                return op.parameters, op.wires.tolist()

            out_trees_and_tracers = []
            insts = get_insts(self)
            for inst in insts:
                params = get_params_from_op_or_m_process(inst)
                flat_tree = tree_flatten(params)
                out_trees_and_tracers.append(flat_tree)

            if self.return_value is not None:
                out_trees_and_tracers += [tree_flatten(self.return_value)]

            flat_params = [p for p_list in out_trees_and_tracers for p in p_list[0]]
            self.output_trees = [p[1] for p in out_trees_and_tracers]
            jaxpr, const_vals = self.frame.to_jaxpr([self.trace.full_raise(p) for p in flat_params])
            self.closed_jaxpr = jax.core.ClosedJaxpr(jaxpr, const_vals)

        # tear down jax tracing logic
        self.trace = None
        self.extended_frame_cm.__exit__(e_type, e_value, traceback)
        self.extended_frame_cm = None
        self.frame = None
        self.main_cm.__exit__(e_type, e_value, traceback)
        self.main_cm = None

    def eval(self, *args):
        """Provide mid circuit measurement results and loop outcomes and get
        the set of circuit parameters.

        Args:
            args (list): mid-circuit measurements and loop results.

        Returns:
            circuit parameters
        """
        return jax.core.eval_jaxpr(self.closed_jaxpr.jaxpr, self.closed_jaxpr.literals, *args)

    def create_tracer(self, tree, avals):
        """
        Create JAX tracers for the given abstract arrays.
        """
        return tree_unflatten(tree, map(self.trace.new_arg, avals))

    def get_parameter_evaluator(self):
        """Create an instance of a ParameterEvaluator

        Returns:
            ParamEvaluator for the current tape.
        """
        return ParamEvaluator(self.closed_jaxpr, self.output_trees)

    def set_return_val(self, return_value):
        """Set the return value for the tape.

        Args:
            return_value: The return value.
        """
        self.return_value = return_value


class EmptyObservable:
    """Denotes an observable equivalent to a given set of wires.

    This class is used only for consistency of how observables are handled.
    """

    def __init__(self, wires):
        self.wires = wires
        self.parameters = []


def get_insts(tape):
    """Get instructions.

    Instructions here are equivalent to operations and observables.
    """
    yield from tape.quantum_tape.operations

    for meas in tape.quantum_tape.measurements:
        for item in get_observables_dependency_tree(meas.obs):
            if item:
                yield item
            else:
                yield EmptyObservable(meas.wires)


def get_observables_dependency_tree(obs):
    """Get observable dependency tree.

    Observables are flattened out in pre-order.
    """
    if obs is None:
        yield obs
    elif isinstance(obs, jax_tracer.KNOWN_NAMED_OBS):
        yield obs
    elif isinstance(obs, qml.Hermitian):
        yield obs
    elif isinstance(obs, qml.operation.Tensor):
        yield obs
        for o in obs.obs:
            yield from get_observables_dependency_tree(o)
    elif isinstance(obs, qml.Hamiltonian):
        yield obs
        for o in obs.ops:
            yield from get_observables_dependency_tree(o)
    else:
        raise ValueError("Unsupported observable")
