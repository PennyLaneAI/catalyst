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
A transform for the new MLIR-based Catalyst decomposition system.
"""


from __future__ import annotations

import functools
import inspect
import types
import warnings
from collections.abc import Callable
from typing import get_type_hints

import jax
import pennylane as qml
from pennylane.decomposition import DecompositionGraph
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

from catalyst.jax_primitives import decomposition_rule

# A mapping from operation names to the number of wires they act on
# and the number of parameters they have.
# This is used when the operation is not in the captured operations
# but we still need to create a decomposition rule for it.
#
# Note that some operations have a variable number of wires,
# e.g., MultiRZ, GlobalPhase. For these, we set the number
# of wires to -1 to indicate a variable number.
#
# This will require a copy of the function to be made
# when creating the decomposition rule to avoid mutating
# the original function with attributes like num_wires.

# A list of operations that can be represented
# in the Catalyst compiler. This will be a superset of
# the operations supported by the runtime.

# FIXME: ops with OpName(params, wires) signatures can be
# represented in the Catalyst compiler. Unfortunately,
# the signature info is not sufficient as there are
# templates with the same signature that should be
# disambiguated.
COMPILER_OPS_FOR_DECOMPOSITION: dict[str, tuple[int, int]] = {
    "CNOT": (2, 0),
    "ControlledPhaseShift": (2, 1),
    "CRot": (2, 3),
    "CRX": (2, 1),
    "CRY": (2, 1),
    "CRZ": (2, 1),
    "CSWAP": (3, 0),
    "CY": (2, 0),
    "CZ": (2, 0),
    "Hadamard": (1, 0),
    "Identity": (1, 0),
    "IsingXX": (2, 1),
    "IsingXY": (2, 1),
    "IsingYY": (2, 1),
    "IsingZZ": (2, 1),
    "SingleExcitation": (2, 1),
    "DoubleExcitation": (4, 1),
    "ISWAP": (2, 0),
    "PauliX": (1, 0),
    "PauliY": (1, 0),
    "PauliZ": (1, 0),
    "PauliRot": (-1, 1),
    "PauliMeasure": (-1, 1),
    "PhaseShift": (1, 1),
    "PSWAP": (2, 1),
    "Rot": (1, 3),
    "RX": (1, 1),
    "RY": (1, 1),
    "RZ": (1, 1),
    "S": (1, 0),
    "SWAP": (2, 0),
    "T": (1, 0),
    "Toffoli": (3, 0),
    "U1": (1, 1),
    "U2": (1, 2),
    "U3": (1, 3),
    "MultiRZ": (-1, 1),
    "GlobalPhase": (-1, 1),
}


class DecompRuleInterpreter(qml.capture.PlxprInterpreter):
    """Interpreter for getting the decomposition graph solution
    from a jaxpr when program capture is enabled.

    This interpreter captures all operations seen during the interpretation
    and builds a decomposition graph to find efficient decomposition pathways
    to a target gate set.

    This interpreter should be used with `qml.decomposition.enable_graph()`
    to enable graph-based decomposition.

    Note that this doesn't actually decompose the operations during interpretation.
    It only captures the operations and builds the decomposition graph.
    The actual decomposition is done later in the MLIR decomposition pass.

    See also: :class:`~.DecompositionGraph`.

    Args:
        ag_enabled (bool): Whether to enable autograph in the decomposition rules.
        gate_set (set[Operator] or None): The target gate set to decompose to
        fixed_decomps (dict or None): A dictionary of fixed decomposition rules
            to use in the decomposition graph.
        alt_decomps (dict or None): A dictionary of alternative decomposition rules
            to use in the decomposition graph.

    Raises:
        TypeError: if graph-based decomposition is not enabled.
    """

    def __init__(
        self,
        *,
        ag_enabled=False,
        gate_set=None,
        fixed_decomps=None,
        alt_decomps=None,
    ):

        if not qml.decomposition.enabled_graph():  # pragma: no cover
            raise TypeError(
                "The DecompRuleInterpreter can only be used when"
                "graph-based decomposition is enabled."
            )

        self._ag_enabled = ag_enabled
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

    def cleanup(self):
        """Cleanup after interpretation."""

        # Solve the decomposition graph to get the decomposition rules
        # for all the captured operations
        # I know it's a bit hacky to do this here, but it's the only
        # place where we can be sure that we have seen all operations
        # in the circuit before the measurement.
        # TODO: Find a better way to do this.
        self._decomp_graph_solution = _solve_decomposition_graph(
            self._operations,
            self._gate_set,
            fixed_decomps=self._fixed_decomps,
            alt_decomps=self._alt_decomps,
        )

        # Create decomposition rules for each operation in the solution
        # and compile them to Catalyst JAXPR decomposition rules
        for op, rule in self._decomp_graph_solution.items():
            # Get number of wires if exists
            op_num_wires = op.op.params.get("num_wires", None)
            if (
                o := next(
                    (
                        o
                        for o in self._operations
                        if o.name == op.op.name and len(o.wires) == op_num_wires
                    ),
                    None,
                )
            ) is not None:
                num_wires, num_params = COMPILER_OPS_FOR_DECOMPOSITION[op.op.name]
                _create_decomposition_rule(
                    rule,
                    op_name=op.op.name,
                    num_wires=len(o.wires),
                    num_params=num_params,
                    requires_copy=num_wires == -1,
                    ag_enabled=self._ag_enabled,
                )
            elif op.op.name in COMPILER_OPS_FOR_DECOMPOSITION:
                # In this part, we need to handle the case where an operation in
                # the decomposition graph solution is not in the captured operations.
                # This can happen if the operation is not directly called
                # in the circuit, but is used inside a decomposition rule.
                # In this case, we fall back to using the COMPILER_OPS_FOR_DECOMPOSITION
                # dictionary to get the number of wires.
                num_wires, num_params = COMPILER_OPS_FOR_DECOMPOSITION[op.op.name]
                pauli_word = op.op.params.get("pauli_word", None)
                requires_copy = num_wires == -1

                if op.op.name in ("PauliRot", "PauliMeasure"):
                    num_wires = len(pauli_word)
                elif num_wires == -1 and op_num_wires is not None:
                    num_wires = op_num_wires

                _create_decomposition_rule(
                    rule,
                    op_name=op.op.name,
                    num_wires=num_wires,
                    num_params=num_params,
                    requires_copy=requires_copy,
                    ag_enabled=self._ag_enabled,
                    pauli_word=pauli_word,
                )
            elif not any(
                keyword in getattr(op.op, "name", "")
                for keyword in (
                    "Adjoint",
                    "Controlled",
                    "TemporaryAND",
                    "ChangeOpBasis",
                    "Prod",
                )
            ):  # pragma: no cover
                # Note that the graph-decomposition returns abstracted rules
                # for Adjoint and Controlled operations, so we skip them here.
                # These abstracted rules cannot be captured and lowered.
                # We use MLIR AdjointOp and ControlledOp primitives
                # to deal with decomposition of symbolic operations at PLxPR.
                raise ValueError(f"Could not capture {op} without the number of wires.")


# pylint: disable=too-many-arguments, too-many-positional-arguments
def _create_decomposition_rule(
    func: Callable,
    op_name: str,
    num_wires: int,
    num_params: int,
    requires_copy: bool = False,
    ag_enabled: bool = False,
    pauli_word: str | None = None,
):
    """Create a decomposition rule from a callable.
    See also: :func:`~.decomposition_rule`.
    Args:
        func (Callable): The decomposition function.
        op_name (str): The name of the operation to decompose.
        num_wires (int): The number of wires the operation acts on.
        num_params (int): The number of parameters the operation takes.
        requires_copy (bool): Whether to create a copy of the function
            to avoid mutating the original. This is required for operations
            with a variable number of wires (e.g., MultiRZ, GlobalPhase).
        ag_enabled (bool): Whether to enable autograph in the decomposition rule.
    """

    sig_func = inspect.signature(func)
    type_hints = get_type_hints(func)

    args = []

    for name in sig_func.parameters.keys():
        typ = type_hints.get(name, None)

        # Skip tailing args or kwargs in the rules
        if name in ("__", "_"):
            continue

        # TODO: This is a temporary solution until all rules have proper type annotations.
        # Why? Because we need to pass the correct types to the decomposition_rule
        # function to capture the rule correctly with JAX.
        possible_names_for_single_param = {
            "param",
            "angle",
            "phi",
            "omega",
            "theta",
            "weight",
        }
        possible_names_for_multi_params = {
            "params",
            "angles",
            "weights",
        }

        # TODO: Support work-wires when it's supported in Catalyst.
        possible_names_for_wires = {"wires", "wire", "control_wires", "target_wires"}

        if typ is TensorLike or name in possible_names_for_multi_params:
            args.append(qml.math.array([0.0] * num_params, like="jax", dtype=float))
        elif typ is float or name in possible_names_for_single_param:
            # TensorLike is a Union of float, int, array-like, so we use float here
            # to cover the most common case as the JAX tracer doesn't like Union types
            # and we don't have the actual values at this point.
            args.append(float)
        elif typ is WiresLike or name in possible_names_for_wires:
            # Pass a dummy array of zeros with the correct number of wires
            # This is required for the decomposition_rule to work correctly
            # as it expects an array-like input for wires
            args.append(qml.math.array([0] * num_wires, like="jax"))
        elif typ is int:  # pragma: no cover
            # This is only for cases where the rule has an int parameter
            # e.g., dimension in some gates. Not that common though!
            # We cover this when adding end-to-end tests for rules
            # in the MLIR PR.
            args.append(int)
        elif pauli_word is not None and typ is str:
            pass
        else:  # pragma: no cover
            raise ValueError(
                f"Unsupported type annotation {typ} for parameter {name} in func {func}."
            )

    func_cp = make_def_copy(func) if requires_copy else func

    if requires_copy:
        # Include number of wires in the function name to avoid name clashes
        # when the same rule is compiled multiple times with different number of wires
        # (e.g., MultiRZ, GlobalPhase)
        func_cp.__name__ += f"_wires_{num_wires}"

    if ag_enabled:
        from pennylane.capture.autograph import (  # pylint: disable=import-outside-toplevel
            run_autograph,
        )

        # Capture the function with autograph
        func_cp = run_autograph(func_cp)

    # Set custom attributes for the decomposition rule
    # These attributes are used in the MLIR decomposition pass
    # to identify the target gate and the number of wires
    setattr(func_cp, "target_gate", op_name)
    setattr(func_cp, "num_wires", num_wires)

    # Note that we shouldn't pass args as kwargs to decomposition_rule
    # JAX doesn't like it and it may fail to preserve the order of args.
    return decomposition_rule(func_cp, pauli_word=pauli_word)(*args)


# pylint: disable=protected-access
def _solve_decomposition_graph(operations, gate_set, fixed_decomps, alt_decomps):
    """Get the decomposition graph solution for the given operations and gate set.

    TODO: Extend `DecompGraphSolution` API and avoid accessing protected members
    directly in this function.

    Args:
        operations (set[Operator]): The set of operations to decompose.
        gate_set (set[Operator]): The target gate set to decompose to.
        fixed_decomps (dict or None): A dictionary of fixed decomposition rules
            to use in the decomposition graph.
        alt_decomps (dict or None): A dictionary of alternative decomposition rules
            to use in the decomposition graph.

    Returns:
        dict: A dictionary mapping operations to their decomposition rules.
    """

    # decomp_graph_solution
    decomp_graph_solution = {}

    decomp_graph = DecompositionGraph(
        operations,
        gate_set,
        fixed_decomps=fixed_decomps,
        alt_decomps=alt_decomps,
    )

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", UserWarning)
        solutions = decomp_graph.solve()

    # Check if the graph-based decomposition failed for any operation
    # We shall do the check after the context manager of warnings.catch_warnings
    # to be able to check and re-emit the warnings to the user.
    graph_failed = False

    for wi in captured_warnings:
        # Re-emit all captured warnings to the user
        warnings.showwarning(wi.message, wi.category, wi.filename, wi.lineno)

        # TODO: use a custom warning class for this in PennyLane to remove this
        # string matching and make it more robust.
        if "The graph-based decomposition system is unable" in str(wi.message):  # pragma: no cover
            graph_failed = True

    if graph_failed:
        # Note that this warning is already issued in the DecompositionGraph.solve()
        # method, but we capture it here to make it more visible to the user
        # that we are falling back to the standard PennyLane decomposition.
        # This is important because we cannot use `op.decomposition()` in the
        # Catalyst MLIR decomposition pass, as it may introduce new unsupported ops
        # so we need to inform the user that if some operations could not be
        # decomposed to the target gate set using the graph-based approach,
        # we need to fallback to the legacy approach without MLIR decomposition.
        return {}

    def is_solved_for(op):
        return (
            op in solutions._all_op_indices
            and solutions._all_op_indices[op] in solutions._visitor.distances
        )

    for op_node, op_node_idx in solutions._all_op_indices.items():
        if is_solved_for(op_node) and op_node_idx in solutions._visitor.predecessors:
            d_node_idx = solutions._visitor.predecessors[op_node_idx]
            decomp_graph_solution[op_node] = solutions._graph[d_node_idx].rule._impl

    return decomp_graph_solution


def make_def_copy(func):
    """Create a copy of a Python definition to avoid mutating the original.

    This is especially useful when compiling decomposition rules with
    parametric number of wires (e.g., MultiRZ, GlobalPhase) multiple times,
    as the compilation process may add attributes to the function that
    can interfere with subsequent compilations.

    Args:
        func (Callable): The function to copy.

    Returns:
        Callable: A copy of the original function with the same attributes.
    """
    # Create a new function object with the same code, globals, name, defaults, and closure
    func_copy = types.FunctionType(
        func.__code__,
        func.__globals__,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=func.__closure__,
    )

    # Now, we create and update the wrapper to copy over attributes like docstring, module, etc.
    return functools.update_wrapper(func_copy, func)
