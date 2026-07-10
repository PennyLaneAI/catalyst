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

import pennylane as qp
import pytest

from catalyst.device.python_decompositions import python_decomposition_wrapper


class TestQPD:
    """Test the python wrapper functions used for compile-time decomposition rule lowering."""

    def test_paulirot_wrapper(self):
        """Test that the paulirot QPD wrapper correctly returns the IR as a string."""
        result = python_decomposition_wrapper(
            "PauliRot", "PauliRot[f64][3]{pauli_word:XZZ}", [0.4], [3], {"pauli_word": "XZZ"}
        )
        assert isinstance(result, str)
        assert "PauliRot[f64][3]{pauli_word:XZZ}__pauli_rot_decomposition" in result
        assert "Hadamard" in result
        assert "multirz" in result

    def test_multiple_rules(self):
        """Test that the python decomposition wrapper supports multiple rules."""
        with qp.decomposition.local_decomps():

            def test_resources():
                return {qp.X: 1}

            @qp.register_resources(test_resources)
            def test_decomp(angle, wires, pauli_word):
                qp.RX(angle, wires[0])

            qp.add_decomps(qp.PauliRot, test_decomp)

            result = python_decomposition_wrapper(
                "PauliRot", "PauliRot[f64][3]{pauli_word:XYX}", [float], [3], {"pauli_word": "XYX"}
            )

            assert "PauliRot[f64][3]{pauli_word:XYX}_test_decomp" in result
            assert "test_decomp" in result


if __name__ == "__main__":
    pytest.main(["-x", __file__])
