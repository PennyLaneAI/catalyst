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

from catalyst.from_plxpr.decompose import _resource_num_wires
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
    """Regression tests for resource representations that use abstract wires (``Wire[n]``)."""

    def test_abstract_wire_operator2_resource_rep(self):
        """A decomposition rule may describe its resources with an ``Operator2`` built from abstract
        wires (``pennylane.typing.Wire[n]``), whose ``.wires`` is an ``AbstractWires`` that exposes
        ``__len__`` but not ``.num_wires``. ``_resource_num_wires`` must read the wire count via
        ``len`` rather than ``.num_wires`` (regression test for #3163)."""
        op_rep = qp.SemiAdder(Wire[2], Wire[2], Wire[1])

        assert isinstance(op_rep, qp.core.Operator2)
        # The exact access pattern that used to crash: AbstractWires has no ``num_wires``.
        with pytest.raises(AttributeError):
            _ = op_rep.wires.num_wires

        assert _resource_num_wires(op_rep) == 5


if __name__ == "__main__":
    pytest.main(["-x", __file__])
