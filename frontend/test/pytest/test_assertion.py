# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for the runtime assertion feature."""

import pytest

from catalyst import debug_assert, qjit


class TestAssertion:
    """Test that the runtime assertion works correctly in different cases."""

    def test_static_assertion_true(self):
        """Test that the static true assertion always passes."""

        @qjit
        def circuit(x):
            debug_assert(True, "Always pass")
            return x * 8

        assert circuit(5) == 40

    def test_static_assertion_false(self):
        """Test that the static false assertion always fails."""

        @qjit
        def circuit(x):
            debug_assert(False, "Always fail")
            return x * 8

        with pytest.raises(RuntimeError, match="Always fail"):
            circuit(5)

    def test_dynamic_assertion(self):
        """Test that the dynamic assertions work."""

        @qjit
        def circuit(x):
            debug_assert(x < 6, "x greater than 6")
            return x * 8

        assert circuit(5) == 40
        with pytest.raises(RuntimeError, match="x greater than 6"):
            circuit(7)

    def test_disabling_assertions_static(self):
        """Test that disabling static assertion works."""

        @qjit(disable_assertions=True)
        def circuit(x):
            debug_assert(False, "x greater than 6")
            return x * 8

        assert circuit(5) == 40

    def test_disabling_assertions_dynamic(self):
        """Test that disabling static assertion works."""

        @qjit(disable_assertions=True)
        def circuit(x):
            debug_assert(x < 6, "x greater than 6")
            return x * 8

        assert circuit(7) == 56
        assert circuit(5) == 40

    def test_disable_assertions_pass(self):
        """Test that disabling and enabling disable_assertions pass works properly."""

        def circuit():
            debug_assert(False, "Always raise")
            return True

        with pytest.raises(RuntimeError, match="Always raise"):
            qjit()(circuit)()

        assert qjit(disable_assertions=True)(circuit)() == True

        with pytest.raises(RuntimeError, match="Always raise"):
            qjit()(circuit)()

        assert qjit(disable_assertions=True)(circuit)() == True


if __name__ == "__main__":
    pytest.main(["-x", __file__])
