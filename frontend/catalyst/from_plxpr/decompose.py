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

import inspect
from collections.abc import Callable
from copy import copy
from typing import get_type_hints

import jax
import pennylane as qml

# GraphSolutionInterpreter:
from pennylane.decomposition import DecompositionGraph
from pennylane.measurements import MidMeasureMP
from pennylane.wires import WiresLike

from catalyst.jax_primitives import decomposition_rule


def create_decomposition_rule(func: Callable, op_name: str, num_wires: int):
    """Create a decomposition rule from a function."""

    sig_func = inspect.signature(func)
    type_hints = get_type_hints(func)

    args = {}
    for name in sig_func.parameters.keys():
        typ = type_hints.get(name, None)

        # Skip tailing kwargs in the rules
        if name == "__":
            continue

        if typ is float or name in ("phi", "theta", "omega", "delta"):
            args[name] = float
        elif typ is int:
            args[name] = int
        elif typ is WiresLike or name == "wires":
            args[name] = qml.math.array([0] * num_wires, like="jax")
        else:
            raise ValueError(
                f"Unsupported type annotation {typ} for parameter {name} in func {func}."
            )

    # Update the name of decomposition rule
    rule_name = "_rule" if func.__name__[0] == "_" else "_rule_"
    func.__name__ = op_name + rule_name + func.__name__

    return decomposition_rule(func)(**args)


# pylint: disable=too-few-public-methods
class GraphSolutionInterpreter(qml.capture.PlxprInterpreter):
    """Interpreter for getting the decomposition graph solution
    from a jaxpr when program capture is enabled.
    """

    def __init__(
        self,
        *,
        gate_set=None,
        fixed_decomps=None,
        alt_decomps=None,
    ):  # pylint: disable=too-many-arguments

        if not qml.decomposition.enabled_graph():
            raise TypeError(
                "The GraphSolutionInterpreter can only be used when"
                "graph-based decomposition is enabled."
            )

        self._gate_set = gate_set
        self._fixed_decomps = fixed_decomps
        self._alt_decomps = alt_decomps

        self._captured = False
        self._operations = set()
        self._decomp_graph_solution = {}

    def interpret_operation(self, op: "qml.operation.Operator"):
        """Interpret a PennyLane operation instance.

        Args:
            op (Operator): a pennylane operator instance

        Returns:
            Any

        This method is only called when the operator's output is a dropped variable,
        so the output will not affect later equations in the circuit.

        We cache the list of operations seen during the interpretation
        to build the decomposition graph in the later stages.

        See also: :meth:`~.interpret_operation_eqn`.

        """

        self._operations.add(op)
        data, struct = jax.tree_util.tree_flatten(op)
        return jax.tree_util.tree_unflatten(struct, data)

    def interpret_measurement(self, measurement: "qml.measurement.MeasurementProcess"):
        """Interpret a measurement process instance.

        Args:
            measurement (MeasurementProcess): a measurement instance.

        See also :meth:`~.interpret_measurement_eqn`.

        """

        if not self._captured and not isinstance(measurement, MidMeasureMP):
            self._captured = True
            if self._fixed_decomps:
                for op, rule in self._fixed_decomps.items():
                    # TODO: number of wires for multi-wires custom gates
                    create_decomposition_rule(
                        rule._impl, op if isinstance(op, str) else op.__name__, num_wires=1
                    )

            self._decomp_graph_solution = _solve_decomposition_graph(
                self._operations,
                self._gate_set,
                fixed_decomps=self._fixed_decomps,
                alt_decomps=self._alt_decomps,
            )

            captured_ops = copy(self._operations)
            for op, rule in self._decomp_graph_solution.items():
                for o in captured_ops:
                    if o.name == op.op.name:
                        create_decomposition_rule(rule, op_name=op.op.name, num_wires=len(o.wires))
                        captured_ops.remove(o)
                        break
                else:
                    # else query the number of wires by name
                    create_decomposition_rule(rule, op_name=op.op.name, num_wires=1)

        data, struct = jax.tree_util.tree_flatten(measurement)
        return jax.tree_util.tree_unflatten(struct, data)


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
