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
    """A PennyLane ``DecompositionRule`` (from ``@register_resources``.

    Its ``.name`` is the reference; as of PennyLane #10144, it also has a ``.__name__``
    which isn't always interchangable with ``.name``.
    """

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

        options = _setup(fixed_decomps={qp.Hadamard: rule})
        assert options["fixed_decomps"] == {"Hadamard": "h_to_rz"}

    def test_decomposition_rule_name_wins_over_dunder_name(self):
        """Test that .name is preferred over .__name__."""

        rule = _registered_rule()
        rule.name = "h_to_rz_renamed"
        assert rule.__name__ == "h_to_rz"

        options = _setup(fixed_decomps={qp.Hadamard: rule})
        # Renamed .name is used
        assert options["fixed_decomps"] == {"Hadamard": "h_to_rz_renamed"}

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


class TestVerboseOption:
    """Cover the ``verbose`` option."""

    def test_absent_when_not_requested(self):
        """Test the option is omitted unless asked for, so an ordinary compilation emits the same pass
        options it did before this option existed."""
        assert "verbose" not in _setup()
        assert "verbose" not in _setup(verbose=False)

    def test_present_when_requested(self):
        """Test ``verbose=True`` is carried into the pass options."""
        assert _setup(verbose=True)["verbose"] is True

    def test_coexists_with_other_options(self):
        """Test requesting verbosity leaves the rest of the pass options untouched."""
        options = _setup(verbose=True, fixed_decomps={qp.PauliX: "custom_x_rule"})
        assert options["verbose"] is True
        assert options["fixed_decomps"] == {"PauliX": "custom_x_rule"}
        assert options["gate_set"] == {"RX": 1.0}


class TestGateSetOption:
    """Cover how gate-set names are carried into the pass options."""

    def test_modifier_wrapped_name_is_preserved(self):
        """Test a modifier-wrapped gate-set name (e.g. ``Adjoint(TemporaryAND)`` will be
        serialized correctly down to the pass options.
        """
        _, options = graph_decomposition_setup_inputs(
            {"Adjoint(TemporaryAND)", "TemporaryAND"}, **_DUMMY_LIBS
        )
        assert options["gate_set"] == {"Adjoint(TemporaryAND)": 1.0, "TemporaryAND": 1.0}


if __name__ == "__main__":
    pytest.main(["-x", __file__])
