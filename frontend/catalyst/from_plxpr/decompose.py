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
A transform for the new MLIT-based Catalyst decomposition system.
"""


from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import jax
import pennylane as qml

# Support ctrl ops in decomposition (adapted from PL's DecomposeInterpreter)
from pennylane.capture.primitives import ctrl_transform_prim

# GraphSolutionInterpreter:
from pennylane.decomposition import DecompositionGraph
from pennylane.decomposition.utils import translate_op_alias
from pennylane.operation import Operator


# pylint: disable=too-few-public-methods
class GraphSolutionInterpreter(qml.capture.PlxprInterpreter):
    """Interpreter for getting the decomposition graph solution
    from a jaxpr when program capture is enabled.

    This interpreter should be used after the PreMlirDecomposeInterpreter.
    """

    def __init__(
        self,
        *,
        operations=[],
        gate_set=None,
        fixed_decomps=None,
        alt_decomps=None,
    ):  # pylint: disable=too-many-arguments

        if not qml.decomposition.enabled_graph():
            raise TypeError(
                "The GraphSolutionInterpreter can only be used when"
                "graph-based decomposition is enabled."
            )

        self._operations = operations
        self._decomp_graph_solution = {}
        self._target_gate_names = None
        self._fixed_decomps, self._alt_decomps = fixed_decomps, alt_decomps

        gate_set, _ = _resolve_gate_set(gate_set)
        self._gate_set = gate_set
        self._env = {}

    # pylint: disable=too-many-branches, too-many-locals
    def eval(self, jaxpr: "jax.extend.core.Jaxpr", consts: Sequence, *args) -> list:
        """Evaluate a jaxpr.

        Args:
            jaxpr (jax.extend.core.Jaxpr): the jaxpr to evaluate
            consts (list[TensorLike]): the constant variables for the jaxpr
            *args (tuple[TensorLike]): The arguments for the jaxpr.

        Returns:
            list[TensorLike]: the results of the execution.

        """
        self._env = {}
        self.setup()

        for arg, invar in zip(args, jaxpr.invars, strict=True):
            self._env[invar] = arg
        for const, constvar in zip(consts, jaxpr.constvars, strict=True):
            self._env[constvar] = const

        if self._operations and not self._decomp_graph_solution:

            self._decomp_graph_solution = _solve_decomposition_graph(
                self._operations,
                self._gate_set,
                fixed_decomps=self._fixed_decomps,
                alt_decomps=self._alt_decomps,
            )

        for eqn in jaxpr.eqns:
            primitive = eqn.primitive
            custom_handler = self._primitive_registrations.get(primitive, None)

            if custom_handler:
                invals = [self.read(invar) for invar in eqn.invars]
                outvals = custom_handler(self, *invals, **eqn.params)
            elif getattr(primitive, "prim_type", "") == "operator":
                outvals = self.interpret_operation_eqn(eqn)
            elif getattr(primitive, "prim_type", "") == "measurement":
                outvals = self.interpret_measurement_eqn(eqn)
            else:
                invals = [self.read(invar) for invar in eqn.invars]
                subfuns, params = primitive.get_bind_params(eqn.params)
                outvals = primitive.bind(*subfuns, *invals, **params)

            if not primitive.multiple_results:
                outvals = [outvals]
            for outvar, outval in zip(eqn.outvars, outvals, strict=True):
                self._env[outvar] = outval

        # Read the final result of the Jaxpr from the environment
        outvals = []
        for var in jaxpr.outvars:
            outval = self.read(var)
            if isinstance(outval, qml.operation.Operator):
                outvals.append(self.interpret_operation(outval))
            else:
                outvals.append(outval)
        self.cleanup()
        self._env = {}
        return outvals


# pylint: disable=too-many-arguments
@GraphSolutionInterpreter.register_primitive(ctrl_transform_prim)
def _(self, *invals, n_control, jaxpr, control_values, work_wires, n_consts):
    consts = invals[:n_consts]
    args = invals[n_consts:-n_control]
    control_wires = invals[-n_control:]

    unroller = ControlTransformInterpreter(
        control_wires, control_values=control_values, work_wires=work_wires
    )

    def wrapper(*inner_args):
        return unroller.eval(jaxpr, consts, *inner_args)

    jaxpr = jax.make_jaxpr(wrapper)(*args)
    return self.eval(jaxpr.jaxpr, jaxpr.consts, *args)


class ControlTransformInterpreter(qml.capture.PlxprInterpreter):
    """Interpreter for replacing control transforms with individually controlled ops."""

    def __init__(self, control_wires, control_values=None, work_wires=None):
        super().__init__()
        self.control_wires = control_wires
        self.control_values = control_values
        self.work_wires = work_wires

    def interpret_operation(self, op):
        """Interpret operation."""
        with qml.capture.pause():
            ctrl_op = qml.ctrl(
                op,
                self.control_wires,
                control_values=self.control_values,
                work_wires=self.work_wires,
            )
        super().interpret_operation(ctrl_op)


def _resolve_gate_set(
    gate_set: set[type | str] | dict[type | str, float] = None,
    stopping_condition: Callable[[qml.operation.Operator], bool] = None,
) -> tuple[set[type | str] | dict[type | str, float], Callable[[qml.operation.Operator], bool]]:
    """Resolve the gate set and the stopping condition from arguments."""

    if gate_set is None:
        gate_set = set(qml.ops.__all__)

    if isinstance(gate_set, (str, type)):
        gate_set = {gate_set}

    if isinstance(gate_set, dict):

        if any(v < 0 for v in gate_set.values()):
            raise ValueError("Negative gate weights provided to gate_set are not supported.")

    if isinstance(gate_set, Iterable):

        gate_types = tuple(gate for gate in gate_set if isinstance(gate, type))
        gate_names = {translate_op_alias(gate) for gate in gate_set if isinstance(gate, str)}

        def gate_set_contains(op: Operator) -> bool:
            return (op.name in gate_names) or isinstance(op, gate_types)

    elif isinstance(gate_set, Callable):  # pylint:disable=isinstance-second-argument-not-valid-type

        gate_set_contains = gate_set

    else:
        raise TypeError("Invalid gate_set type. Must be an iterable, dictionary, or function.")

    if stopping_condition:

        # Even when the user provides a stopping condition, we still need to check
        # whether an operator belongs to the target gate set. This is to prevent
        # the case of an operator missing the stopping condition but doesn't have
        # a decomposition assigned due to being in the target gate set.
        def _stopping_condition(op):
            return gate_set_contains(op) or stopping_condition(op)

    else:
        # If the stopping condition is not explicitly provided, the default is to simply check
        # whether an operator belongs to the target gate set.
        _stopping_condition = gate_set_contains

    return gate_set, _stopping_condition


# pylint: disable=protected-access
def _solve_decomposition_graph(operations, gate_set, fixed_decomps, alt_decomps):
    """Get the decomposition graph solution for the given operations and gate set."""

    # decomp_graph_solution
    decomp_graph_solution = {}

    decomp_graph = DecompositionGraph(
        operations,
        gate_set,
        fixed_decomps=fixed_decomps,
        alt_decomps=alt_decomps,
    )

    # Find the efficient pathways to the target gate set
    solutions = decomp_graph.solve()

    def is_solved_for(op):
        return (
            op in solutions._all_op_indices
            and solutions._all_op_indices[op] in solutions._visitor.distances
        )

    for (
        op_node,
        op_node_idx,
    ) in solutions._all_op_indices.items():

        if is_solved_for(op_node) and op_node_idx in solutions._visitor.predecessors:
            d_node_idx = solutions._visitor.predecessors[op_node_idx]
            decomp_graph_solution[op_node] = solutions._graph[d_node_idx].rule._impl

    return decomp_graph_solution
