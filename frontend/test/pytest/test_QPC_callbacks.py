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

"""Unit tests for the python callbacks module."""

from catalyst.python_callbacks import paulirot_callback_wrapper


class TestQPCCallbacks:
    """Test the python wrapper functions used for decomposition callbacks."""

    def test_paulirot_wrapper(self):
        """Test that the QPC paulirot callback wrapper correctly returns the IR as a string."""
        result = paulirot_callback_wrapper(0.4, "XZZ", [0, 1, 2])
        assert isinstance(result, str)
        assert "paulirot_decomp_rule" in result
        assert "Hadamard" in result
        assert "multirz" in result
