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

"""Unit tests for the ``graph_decomposition`` pass option builder."""

from pathlib import Path

import pennylane as qp
import pytest
from pennylane.decomposition import register_resources

from catalyst.passes.builtin_passes import graph_decomposition_setup_inputs

# Dummy lib paths so building the options dict never hits the environment / installed libraries.
_DUMMY_LIBS = {"libQPD_path": Path("/dummy/libQPD"), "libpython_path": Path("/dummy/libpython")}


def _setup(**kwargs):
    """Call graph_decomposition_setup_inputs with a minimal gate set + dummy lib paths, returning
    just the options dict."""
    _, options = graph_decomposition_setup_inputs({qp.RX}, **_DUMMY_LIBS, **kwargs)
    return options


class TestRuleRefName:
    """Test accepted and rejected fixed rule references."""

    def test_string_rule_reference(self):
        """A rule given as a name string is passed through unchanged."""
        with qp.decomposition.local_decomps():

            @register_resources({})
            def custom_x_rule(wire):
                del wire

            qp.add_decomps(qp.PauliX, custom_x_rule)
            options = _setup(fixed_decomps={qp.PauliX: "custom_x_rule"})
            assert options["fixed_decomps"] == {"PauliX": "custom_x_rule"}

    def test_decomposition_rule_reference(self):
        """A ``DecompositionRule`` resolves via ``.name``."""
        with qp.decomposition.local_decomps():

            @register_resources({})
            def h_to_rz(wire):
                del wire

            qp.add_decomps(qp.Hadamard, h_to_rz)
            options = _setup(fixed_decomps={qp.Hadamard: h_to_rz})
            assert options["fixed_decomps"] == {"Hadamard": "h_to_rz"}

    def test_plain_function_reference(self):
        """A bare function raises an exception."""
        with pytest.raises(AssertionError, match="must be"):

            def function(wire):
                del wire

            _setup(fixed_decomps={qp.PauliX: function})


class TestFixedDecompsOption:
    """Cover the ``fixed_decomps`` option."""

    def test_omitted_and_empty_options(self):
        """Omitted options are absent, while an explicit empty mapping is preserved."""
        assert "fixed_decomps" not in _setup()
        assert "fixed_decomps" not in _setup(fixed_decomps=None)
        assert _setup(fixed_decomps={})["fixed_decomps"] == {}

    def test_maps_ops_and_rules_by_name(self):
        """Registered rule objects and names are converted to pass options."""
        with qp.decomposition.local_decomps():

            @register_resources({})
            def x_to_rx(wire):
                del wire

            @register_resources({})
            def h_to_rz(wire):
                del wire

            qp.add_decomps(qp.PauliX, x_to_rx)
            qp.add_decomps(qp.Hadamard, h_to_rz)
            options = _setup(fixed_decomps={qp.PauliX: x_to_rx.name, qp.Hadamard: h_to_rz})

        assert options["fixed_decomps"] == {
            "PauliX": "x_to_rx",
            "Hadamard": "h_to_rz",
        }


class TestExcludeDecompsOption:
    """Cover the ``exclude_decomps`` option."""

    def test_omitted_and_empty_options(self):
        """Omitted options are absent, while an explicit empty mapping is preserved."""
        assert "exclude_decomps" not in _setup()
        assert "exclude_decomps" not in _setup(exclude_decomps=None)
        assert _setup(exclude_decomps={})["exclude_decomps"] == {}

    def test_maps_op_to_tuple_of_rule_names(self):
        """Excluded rule names are converted to a tuple in the pass options."""
        options = _setup(exclude_decomps={qp.Hadamard: ["x_to_rx", "h_to_rz", "custom_rule"]})
        assert options["exclude_decomps"] == {"Hadamard": ("x_to_rx", "h_to_rz", "custom_rule")}

    def test_modifier_wrapped_name_is_preserved(self):
        """Symbolic operator names are preserved for the MLIR pass."""
        options = _setup(exclude_decomps={"Adjoint(Hadamard)": ["custom_rule"]})
        assert options["exclude_decomps"] == {"Adjoint(Hadamard)": ("custom_rule",)}

    def test_coexists_with_fixed_decomps(self):
        """Exclusions and fixed rule selections are separate pass options."""
        with qp.decomposition.local_decomps():

            @register_resources({})
            def h_to_rz(wire):
                del wire

            qp.add_decomps(qp.Hadamard, h_to_rz)
            options = _setup(
                fixed_decomps={qp.Hadamard: h_to_rz.name},
                exclude_decomps={qp.Hadamard: ["custom_rule"]},
            )
        assert options["fixed_decomps"] == {"Hadamard": "h_to_rz"}
        assert options["exclude_decomps"] == {"Hadamard": ("custom_rule",)}


class TestVerboseOption:
    """Cover the ``verbose`` option."""

    def test_absent_when_not_requested(self):
        """The verbosity option is omitted unless requested."""
        assert "verbose" not in _setup()
        assert "verbose" not in _setup(verbose=False)

    def test_present_when_requested(self):
        """Test ``verbose=True`` is carried into the pass options."""
        assert _setup(verbose=True)["verbose"] is True

    def test_coexists_with_other_options(self):
        """Test requesting verbosity leaves the rest of the pass options untouched."""
        with qp.decomposition.local_decomps():

            @register_resources({})
            def custom_x_rule(wire):
                del wire

            qp.add_decomps(qp.PauliX, custom_x_rule)
            options = _setup(verbose=True, fixed_decomps={qp.PauliX: custom_x_rule.name})
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
