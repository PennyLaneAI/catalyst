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

Note: this transform will be merged with the PennyLane decomposition transform as part of
the PennyLane <> Catalyst unification project.
"""


from __future__ import annotations

import warnings
from collections import ChainMap
from collections.abc import Callable, Generator, Iterable, Sequence
from functools import partial

import jax
import pennylane as qml

from pennylane.decomposition.collect_resource_ops import CollectResourceOps

# Support ctrl ops in decomposition (adapted from PL's DecomposeInterpreter)
from pennylane.capture.primitives import ctrl_transform_prim

# GraphSolutionInterpreter:
from pennylane.decomposition import DecompositionGraph
from pennylane.decomposition.decomposition_graph import DecompGraphSolution
from pennylane.decomposition.utils import translate_op_alias
from pennylane.operation import Operator


CollectResourceOps = CollectResourceOps


# pylint: disable=too-many-instance-attributes
class PreMlirDecomposeInterpreter(qml.capture.PlxprInterpreter):
    """Plxpr Interpreter for applying the Catalyst compiler-specific decomposition transform
    to callables or jaxpr when program capture is enabled.

    TODO:
    - Enable graph-based for pre-mlir decomposition
        (not priority for this stage -- needs further maintenance in PennyLane/decomposition)
    - Add a more optimized support for PL's templates

    Note:
    - This interpreter shares common code with PL's DecomposeInterpreter.
      We will merge the two in the future near the completion of the unification project.
    """

    def __init__(
        self,
        *,
        gate_set=None,
        max_expansion=None,
    ):  # pylint: disable=too-many-arguments

        self.max_expansion = max_expansion
        self._current_depth = 0
        self._target_gate_names = None

        # We use a ChainMap to store the environment frames, which allows us to push and pop
        # environments without copying the interpreter instance when we evaluate a jaxpr of
        # a dynamic decomposition. The name is different from the _env in the parent class
        # (a dictionary) to avoid confusion.
        self._env_map = ChainMap()

        gate_set, stopping_condition = _resolve_gate_set(gate_set)
        self._gate_set = gate_set
        self._stopping_condition = stopping_condition

    def setup(self) -> None:
        """Setup the environment for the interpreter by pushing a new environment frame."""

        # This is the local environment for the jaxpr evaluation, on the top of the stack,
        # from which the interpreter reads and writes variables.
        # ChainMap writes to the first dictionary in the chain by default.
        self._env_map = self._env_map.new_child()

    def cleanup(self) -> None:
        """Cleanup the environment by popping the top-most environment frame."""

        # We delete the top-most environment frame after the evaluation is done.
        self._env_map = self._env_map.parents

    def read(self, var):
        """Extract the value corresponding to a variable."""
        return var.val if isinstance(var, jax.extend.core.Literal) else self._env_map[var]

    def stopping_condition(self, op: qml.operation.Operator) -> bool:
        """Function to determine whether an operator needs to be decomposed or not.

        Args:
            op (qml.operation.Operator): Operator to check.

        Returns:
            bool: Whether ``op`` is valid or needs to be decomposed. ``True`` means
                that the operator does not need to be decomposed.
        """

        if not op.has_decomposition:
            if not self._stopping_condition(op):
                warnings.warn(
                    f"Operator {op.name} does not define a decomposition and was not "
                    f"found in the target gate set. To remove this warning, add the operator "
                    f"name ({op.name}) or type ({type(op)}) to the gate set.",
                    UserWarning,
                )
            return True

        return self._stopping_condition(op)

    def decompose_operation(self, op: qml.operation.Operator):
        """Decompose a PennyLane operation instance if it does not satisfy the
        provided gate set.

        Args:
            op (Operator): a pennylane operator instance

        This method is only called when the operator's output is a dropped variable,
        so the output will not affect later equations in the circuit.

        See also: :meth:`~.interpret_operation_eqn`, :meth:`~.interpret_operation`.
        """

        if self._stopping_condition(op):
            return self.interpret_operation(op)

        max_expansion = (
            self.max_expansion - self._current_depth if self.max_expansion is not None else None
        )

        with qml.capture.pause():
            decomposition = list(
                _operator_decomposition_gen(
                    op,
                    self.stopping_condition,
                    max_expansion=max_expansion,
                )
            )

        return [self.interpret_operation(decomp_op) for decomp_op in decomposition]

    def _evaluate_jaxpr_decomposition(self, op: qml.operation.Operator):
        """Creates and evaluates a Jaxpr of the plxpr decomposition of an operator."""

        if self._stopping_condition(op):
            return self.interpret_operation(op)

        if self.max_expansion is not None and self._current_depth >= self.max_expansion:
            return self.interpret_operation(op)

        compute_qfunc_decomposition = op.compute_qfunc_decomposition

        args = (*op.parameters, *op.wires)

        jaxpr_decomp = qml.capture.make_plxpr(
            partial(compute_qfunc_decomposition, **op.hyperparameters)
        )(*args)

        self._current_depth += 1
        # We don't need to copy the interpreter here, as the jaxpr of the decomposition
        # is evaluated with a new environment frame placed on top of the stack.
        out = self.eval(jaxpr_decomp.jaxpr, jaxpr_decomp.consts, *args)
        self._current_depth -= 1

        return out

    # pylint: disable=too-many-branches
    def eval(self, jaxpr: jax.extend.core.Jaxpr, consts: Sequence, *args) -> list:
        """
        Evaluates a jaxpr, which can also be generated by a dynamic decomposition.

        Args:
            jaxpr_decomp (jax.extend.core.Jaxpr): the Jaxpr to evaluate
            consts (list[TensorLike]): the constant variables for the jaxpr
            *args: the arguments to use in the evaluation
        """

        self.setup()

        for arg, invar in zip(args, jaxpr.invars, strict=True):
            self._env_map[invar] = arg
        for const, constvar in zip(consts, jaxpr.constvars, strict=True):
            self._env_map[constvar] = const

        for eq in jaxpr.eqns:

            prim_type = getattr(eq.primitive, "prim_type", "")
            custom_handler = self._primitive_registrations.get(eq.primitive, None)

            if custom_handler:

                invals = [self.read(invar) for invar in eq.invars]
                outvals = custom_handler(self, *invals, **eq.params)

            elif prim_type == "operator":
                outvals = self.interpret_operation_eqn(eq)
            elif prim_type == "measurement":
                outvals = self.interpret_measurement_eqn(eq)
            else:
                invals = [self.read(invar) for invar in eq.invars]
                subfuns, params = eq.primitive.get_bind_params(eq.params)
                outvals = eq.primitive.bind(*subfuns, *invals, **params)

            if not eq.primitive.multiple_results:
                outvals = [outvals]

            for outvar, outval in zip(eq.outvars, outvals, strict=True):
                self._env_map[outvar] = outval

        outvals = []
        for var in jaxpr.outvars:
            outval = self.read(var)
            if isinstance(outval, qml.operation.Operator):
                outvals.append(self.interpret_operation(outval))
            else:
                outvals.append(outval)

        self.cleanup()

        return outvals

    def interpret_operation_eqn(self, eqn: jax.extend.core.JaxprEqn):
        """Interpret an equation corresponding to an operator.

        If the operator has a dynamic decomposition defined, this method will
        create and evaluate the jaxpr of the decomposition using the :meth:`~.eval` method.

        Args:
            eqn (jax.extend.core.JaxprEqn): a jax equation for an operator.

        See also: :meth:`~.interpret_operation`.

        """

        invals = (self.read(invar) for invar in eqn.invars)

        with qml.QueuingManager.stop_recording():
            op = eqn.primitive.impl(*invals, **eqn.params)

        if not eqn.outvars[0].__class__.__name__ == "DropVar":
            return op

        return self.decompose_operation(op)


# pylint: disable=too-many-arguments
@PreMlirDecomposeInterpreter.register_primitive(ctrl_transform_prim)
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

class GraphSolutionInterpreter(qml.capture.PlxprInterpreter):
    """Interpreter for getting the decomposition graph solution
    from a jaxpr when program capture is enabled.

    This interpreter should be used after the PreMlirDecomposeInterpreter.
    """

    def __init__(
        self,
        *,
        operations,
        gate_set=None,
        fixed_decomps=None,
        alt_decomps=None,
    ):  # pylint: disable=too-many-arguments

        if not qml.decomposition.enabled_graph():
            raise TypeError(
                "The GraphSolutionInterpreter can only be used when graph-based decomposition is enabled."
            )

        self._operations = operations
        self._decomp_graph_solution = {}
        self._target_gate_names = None
        self._fixed_decomps, self._alt_decomps = fixed_decomps, alt_decomps

        gate_set, _ = _resolve_gate_set(gate_set)
        self._gate_set = gate_set

    # pylint: disable=too-many-branches
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
 
            decomp_graph = DecompositionGraph(
                self._operations,
                self._gate_set,
                fixed_decomps=self._fixed_decomps,
                alt_decomps=self._alt_decomps,
            )

            # Find the efficient pathways to the target gate set
            solutions = decomp_graph.solve()

            def is_solved_for(op):
                return op in solutions._all_op_indices and solutions._all_op_indices[op] in solutions._visitor.distances

            for op_node, op_node_idx in solutions._all_op_indices.items():

                if is_solved_for(op_node) and op_node_idx in solutions._visitor.predecessors:
                    d_node_idx = solutions._visitor.predecessors[op_node_idx]
                    self._decomp_graph_solution[op_node] = solutions._graph[d_node_idx].rule._impl

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


def _operator_decomposition_gen(
    op: qml.operation.Operator,
    acceptance_function: Callable[[qml.operation.Operator], bool],
    max_expansion: int | None = None,
    current_depth=0,
    decomp_graph_solution: DecompGraphSolution | None = None,
) -> Generator[qml.operation.Operator]:
    """A generator that yields the next operation that is accepted."""

    max_depth_reached = False
    decomp = []

    if max_expansion is not None and max_expansion <= current_depth:
        max_depth_reached = True

    if acceptance_function(op) or max_depth_reached:
        yield op
    elif decomp_graph_solution is not None and decomp_graph_solution.is_solved_for(op):
        op_rule = decomp_graph_solution.decomposition(op)
        with qml.queuing.AnnotatedQueue() as decomposed_ops:
            op_rule(*op.parameters, wires=op.wires, **op.hyperparameters)
        decomp = decomposed_ops.queue
        current_depth += 1
    else:
        decomp = op.decomposition()
        current_depth += 1

    for sub_op in decomp:
        yield from _operator_decomposition_gen(
            sub_op,
            acceptance_function,
            max_expansion=max_expansion,
            current_depth=current_depth,
            decomp_graph_solution=decomp_graph_solution,
        )


def _resolve_gate_set(
    gate_set: set[type | str] | dict[type | str, float] = None,
    stopping_condition: Callable[[qml.operation.Operator], bool] = None,
) -> tuple[set[type | str] | dict[type | str, float], Callable[[qml.operation.Operator], bool]]:
    """Resolve the gate set and the stopping condition from arguments.

    The ``gate_set`` can be provided in various forms, and the ``stopping_condition`` may or
    may not be provided. This function will resolve the gate set and the stopping condition
    to the following standardized form:

    - The ``gate_set`` is set of operator **types** and/or names, or a dictionary mapping operator
      types and/or names to their respective costs. This is only used by the DecompositionGraph
    - The ``stopping_condition`` is a function that takes an operator **instances** and returns
      ``True`` if the operator does not need to be decomposed. This is used during decomposition.

    """

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

