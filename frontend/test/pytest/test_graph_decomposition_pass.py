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

"""Unit tests for the ``graph_decomposition`` decorator: its pass option builder in
builtin_passes.py, and its trace-time handling of controlled / controlled-adjoint Operator2 ops."""

from pathlib import Path

import numpy as np
import pennylane as qp
import pytest
from pennylane.decomposition import DecompositionRule, register_resources
from pennylane.typing import Wire

from catalyst import qjit
from catalyst.from_plxpr import decompose as decompose_module
from catalyst.from_plxpr.decompose import (
    DecompRuleInterpreter,
    _create_decomposition_rule,
    _resource_num_wires,
)
from catalyst.passes.builtin_passes import graph_decomposition_setup_inputs

# Dummy lib paths so building the options dict never hits the environment / installed libraries.
_DUMMY_LIBS = {"libQPD_path": Path("/dummy/libQPD"), "libpython_path": Path("/dummy/libpython")}


def _setup(**kwargs):
    """Call graph_decomposition_setup_inputs with a minimal gate set + dummy lib paths, returning
    just the options dict."""
    _, options = graph_decomposition_setup_inputs({qp.RX}, **_DUMMY_LIBS, **kwargs)
    return options


def _plain_rule():
    """A bare decomposition function (has ``__name__``, is not a ``DecompositionRule``)."""

    def x_to_rx(wire):  # pylint: disable=unused-argument
        ...

    return x_to_rx


def _registered_rule():
    """A PennyLane ``DecompositionRule`` (from ``@register_resources``; has ``.name``, no
    ``__name__``)."""

    @register_resources(lambda: {})
    def h_to_rz(wire):  # pylint: disable=unused-argument
        ...

    return h_to_rz


class TestRuleRefName:
    """Cover every branch of the ``rule_ref_name`` closure."""

    def test_string_rule_reference(self):
        """A rule given as a name string is passed through unchanged."""
        options = _setup(fixed_decomps={qp.PauliX: "custom_x_rule"})
        assert options["fixed_decomps"] == {"PauliX": "custom_x_rule"}

    def test_decomposition_rule_reference(self):
        """A ``DecompositionRule`` resolves via ``.name``."""
        rule = _registered_rule()
        assert isinstance(rule, DecompositionRule)
        assert not hasattr(rule, "__name__")

        options = _setup(fixed_decomps={qp.Hadamard: rule})
        assert options["fixed_decomps"] == {"Hadamard": "h_to_rz"}

    def test_plain_function_reference(self):
        """A bare decomposition function resolves via its ``__name__`` (the final fallback
        branch)."""
        options = _setup(fixed_decomps={qp.PauliX: _plain_rule()})
        assert options["fixed_decomps"] == {"PauliX": "x_to_rx"}


class TestFixedDecompsOption:
    """Cover the ``if fixed_decomps:`` block."""

    def test_absent_when_not_provided(self):
        """No ``fixed_decomps`` key is emitted when the argument is omitted (falsy branch)."""
        assert "fixed_decomps" not in _setup()
        assert "fixed_decomps" not in _setup(fixed_decomps={})

    def test_maps_ops_and_rules_by_name(self):
        """Each operator and its single rule are name-resolved into the ``fixed_decomps`` option,
        across all three rule-reference kinds."""
        options = _setup(
            fixed_decomps={
                qp.PauliX: _plain_rule(),  # -> __name__
                qp.Hadamard: _registered_rule(),  # -> DecompositionRule.name
                qp.T: "custom_t_rule",  # -> str
            }
        )
        assert options["fixed_decomps"] == {
            "PauliX": "x_to_rx",
            "Hadamard": "h_to_rz",
            "T": "custom_t_rule",
        }
        assert "alt_decomps" not in options


class TestAltDecompsOption:
    """Cover the ``if alt_decomps:`` block."""

    def test_absent_when_not_provided(self):
        """No ``alt_decomps`` key is emitted when the argument is omitted (falsy branch)."""
        assert "alt_decomps" not in _setup()
        assert "alt_decomps" not in _setup(alt_decomps={})

    def test_maps_op_to_tuple_of_rule_names(self):
        """An operator maps to a tuple of name-resolved alternative rules, mixing all three
        rule-reference kinds."""
        options = _setup(
            alt_decomps={qp.Hadamard: [_plain_rule(), _registered_rule(), "custom_rule"]}
        )
        assert options["alt_decomps"] == {"Hadamard": ("x_to_rx", "h_to_rz", "custom_rule")}
        assert "fixed_decomps" not in options


class TestResourceReps:
    """Tests for resource representations that use abstract wires (``Wire[n]``)."""

    def test_abstract_wire_operator2_resource_rep(self):
        """``_resource_num_wires`` returns the total wire count of an ``Operator2`` resource
        representation built from abstract wires (``pennylane.typing.Wire[n]``)."""
        op_rep = qp.SemiAdder(Wire[2], Wire[2], Wire[1])

        assert isinstance(op_rep, qp.core.Operator2)
        assert _resource_num_wires(op_rep) == 5


class TestOperator2SolutionNodeCapture:
    """Capturing decomposition solutions that contain an ``Operator2`` (abstract-wire) node."""

    def test_operator2_solution_node_is_captured(self, use_capture_dgraph):
        """An ``Operator2`` such as ``SemiAdder`` appearing in the decomposition-graph solution
        is captured rather than aborting: its wire count is materialized from the abstract
        resource rep and its rule-internal optional parameters (e.g. ``carry_flip``) fall back to
        their defaults.

        The ``target="jaxpr"`` capture runs the frontend graph-decomposition cleanup (where the
        ``Operator2`` node is handled) without triggering the downstream C++ decompose-lowering
        pass. Capture failures are swallowed by ``aot_compile`` and leave ``jaxpr`` as ``None``,
        so a populated ``jaxpr`` is the signal that the ``Operator2`` node was captured.
        """

        @qjit(capture=True, target="jaxpr")
        @qp.transforms.decompose(gate_set={"CNOT", "Toffoli", "X", "Hadamard", "PhaseShift"})
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit():
            qp.SemiAdder(x_wires=[0], y_wires=[1])
            return qp.expval(qp.Z(0))

        assert circuit.jaxpr is not None

    def test_unknown_operator2_argument_raises(self):
        """A rule parameter that is neither a param/wire/compilable arg of the ``Operator2`` nor an
        optional rule-internal parameter (no default) cannot be materialized, so building the rule
        raises. This is the fallback of the ``Operator2`` argument-resolution branch."""
        op_rep = qp.SemiAdder(Wire[2], Wire[2], Wire[1])
        assert isinstance(op_rep, qp.core.Operator2)

        def bad_rule(not_an_arg):  # required param not described by the resource rep
            del not_an_arg

        with pytest.raises(ValueError, match="Unknown Operator2 argument"):
            _create_decomposition_rule(
                bad_rule,
                op_name=op_rep.name,
                op_rep=op_rep,
                num_wires=_resource_num_wires(op_rep),
                num_params=op_rep.num_params,
            )

    def test_uncapturable_solution_node_raises(self, monkeypatch):
        """A solution node that is not a compiler op, not a skippable symbolic op, and not an
        ``Operator2`` has no way to recover its wire count, so ``cleanup`` raises rather than
        silently emitting a bad rule."""

        class _UncapturableOp:
            name = "TotallyUnknownOp"
            params = {}  # `_resource_num_wires` reads `params["num_wires"]` -> None

        class _SolutionNode:
            op = _UncapturableOp()

        interpreter = DecompRuleInterpreter(gate_set={"CNOT"})
        monkeypatch.setattr(
            decompose_module,
            "_solve_decomposition_graph",
            lambda *args, **kwargs: {_SolutionNode(): (lambda: None)},
        )

        with pytest.raises(ValueError, match="Could not capture"):
            interpreter.cleanup()


if __name__ == "__main__":
    pytest.main(["-x", __file__])
