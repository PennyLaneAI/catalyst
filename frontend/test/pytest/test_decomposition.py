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

"""Unit tests for the python decompositions module."""

from pathlib import Path

import pennylane as qp
import pytest

from catalyst.compiler import _quantum_opt
from catalyst.decomposition.precompile_decomposition_rules import (
    get_abstract_args,
    precompile_decomp_rules,
)
from catalyst.decomposition.python_decompositions import compile_decomposition_rules_wrapper
from catalyst.utils.runtime_environment import BYTECODE_FILE_PATH


class TestGenericUtilities:
    """Tests for common decomposition rule lowering utilities."""

    def test_paulirot(self):
        """Test that the QPD wrapper correctly returns the IR as a string."""
        result = compile_decomposition_rules_wrapper(
            "PauliRot", "PauliRot[f64][3]{pauli_word:XZZ}", ["i32"], [3], {"pauli_word": "XZZ"}
        )
        assert isinstance(result, str)
        assert "_pauli_rot_decomposition" in result
        assert 'target_gate = "PauliRot[f64][3]{pauli_word:XZZ}"' in result
        assert "Hadamard" in result
        assert "multirz" in result

    def test_multiple_rules(self):
        """Test that the python decomposition wrapper supports multiple rules."""
        with qp.decomposition.local_decomps():

            def test_resources(pauli_word):  # pylint: disable=unused-argument
                return {qp.X: 1}

            @qp.register_resources(test_resources)
            def test_decomp(angle, wires, pauli_word):  # pylint: disable=unused-argument
                qp.RX(angle, wires[0])

            qp.add_decomps(qp.PauliRot, test_decomp)

            result = compile_decomposition_rules_wrapper(
                "PauliRot", "PauliRot[f64][3]{pauli_word:XYX}", ["f64"], [3], {"pauli_word": "XYX"}
            )

            assert "test_decomp" in result
            assert "_pauli_rot_decomp" in result
            assert 'target_gate = "PauliRot[f64][3]{pauli_word:XYX}"' in result


class TestPrecompiled:
    """Tests for precompiled decomposition rules."""


class TestTraceTime:
    """Placeholder for future tests of trace-time decomposition rule lowering."""


class TestOnDemand:
    """
    Test the python wrapper functions used for on-demand, compile-time decomposition rule lowering.
    """


if __name__ == "__main__":
    pytest.main(["-x", __file__])
