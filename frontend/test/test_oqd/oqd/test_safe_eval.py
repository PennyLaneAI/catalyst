# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest

from catalyst.third_party.oqd.safe_eval import safe_eval


class TestSafeEval:
    """Test suite for the safe_eval function"""

    def test_safe_eval_single_value(self):
        """Test safe_eval on expressions containing single numeric values of various types."""
        assert safe_eval("1") == 1
        assert type(safe_eval("1")) == int

        assert math.isclose(safe_eval("1.0"), 1.0)
        assert type(safe_eval("1.0")) == float

        assert safe_eval("+1") == 1
        assert type(safe_eval("+1")) == int

        assert math.isclose(safe_eval("+1.0"), 1.0)
        assert type(safe_eval("+1.0")) == float

        assert safe_eval("-1") == -1
        assert type(safe_eval("-1")) == int

        assert math.isclose(safe_eval("-1.0"), -1.0)
        assert type(safe_eval("-1.0")) == float

        assert math.isclose(safe_eval("1e3"), 1e3)
        assert math.isclose(safe_eval("1e-3"), 1e-3)

    def test_safe_eval_addition(self):
        """Test safe_eval on expressions containing addition operations."""
        assert safe_eval("1 + 1") == 2
        assert type(safe_eval("1 + 1")) == int

        assert math.isclose(safe_eval("1.0 + 1.0"), 2.0)
        assert type(safe_eval("1.0 + 1.0")) == float

    def test_safe_eval_subtraction(self):
        """Test safe_eval on expressions containing subtraction operations."""
        assert safe_eval("2 - 1") == 1
        assert type(safe_eval("2 - 1")) == int

        assert math.isclose(safe_eval("2.0 - 1.0"), 1.0)
        assert type(safe_eval("2.0 - 1.0")) == float

        assert safe_eval("1 - 2") == -1
        assert math.isclose(safe_eval("1.0 - 2.0"), -1.0)

    def test_safe_eval_multiplication(self):
        """Test safe_eval on expressions containing multiplication operations."""
        assert safe_eval("2 * 2") == 4
        assert type(safe_eval("2 * 2")) == int

        assert math.isclose(safe_eval("2.0 * 2.0"), 4.0)
        assert type(safe_eval("2.0 * 2.0")) == float

        assert safe_eval("2 * -2") == -4
        assert math.isclose(safe_eval("2.0 * -2.0"), -4.0)

    def test_safe_eval_division(self):
        """Test safe_eval on expressions containing division operations."""
        assert math.isclose(safe_eval("4 / 2"), 2.0)
        assert type(safe_eval("4 / 2")) == float

        assert math.isclose(safe_eval("4.0 / 2.0"), 2.0)
        assert type(safe_eval("4.0 / 2.0")) == float

        assert math.isclose(safe_eval("4.0 / -2.0"), -2.0)
        assert math.isclose(safe_eval("-4.0 / 2.0"), -2.0)

        assert math.isclose(safe_eval("1.0 / 2.0"), 0.5)

    def test_safe_eval_exponentiation(self):
        """Test safe_eval on expressions containing exponentiation operations."""
        assert math.isclose(safe_eval("2 ** 2"), 4)
        assert type(safe_eval("2 ** 2")) == int

        assert math.isclose(safe_eval("2.0 ** 2"), 4.0)
        assert type(safe_eval("2.0 ** 2")) == float

    def test_safe_eval_math_constants(self):
        """Test safe_eval on expressions containing constants from the math module."""
        assert math.isclose(safe_eval("math.pi"), math.pi)
        assert math.isclose(safe_eval("math.e"), math.e)
        assert math.isinf(safe_eval("math.inf"))
        assert math.isnan(safe_eval("math.nan"))

    def test_safe_eval_math_functions(self):
        """Test safe_eval on expressions containing functions from the math module."""
        assert math.isclose(safe_eval("math.sin(0.5)"), math.sin(0.5))
        assert math.isclose(safe_eval("math.cos(0.5)"), math.cos(0.5))
        assert math.isclose(safe_eval("math.tan(0.5)"), math.tan(0.5))
        assert math.isclose(safe_eval("math.asin(0.5)"), math.asin(0.5))
        assert math.isclose(safe_eval("math.acos(0.5)"), math.acos(0.5))
        assert math.isclose(safe_eval("math.atan(0.5)"), math.atan(0.5))
        assert math.isclose(safe_eval("math.sinh(0.5)"), math.sinh(0.5))
        assert math.isclose(safe_eval("math.cosh(0.5)"), math.cosh(0.5))
        assert math.isclose(safe_eval("math.tanh(0.5)"), math.tanh(0.5))
        assert math.isclose(safe_eval("math.asinh(0.5)"), math.asinh(0.5))
        assert math.isclose(safe_eval("math.acosh(1.5)"), math.acosh(1.5))
        assert math.isclose(safe_eval("math.atanh(0.5)"), math.atanh(0.5))
        assert math.isclose(safe_eval("math.log(0.5)"), math.log(0.5))
        assert math.isclose(safe_eval("math.log10(0.5)"), math.log10(0.5))
        assert math.isclose(safe_eval("math.log2(0.5)"), math.log2(0.5))

    def test_safe_eval_complex_numbers(self):
        """Test safe_eval on expressions containing complex numbers."""
        assert safe_eval("1 + 1j") == 1 + 1j
        assert safe_eval("(1 + 1j) * (1 - 1j)") == 2 + 0j

    def test_safe_eval_long_expr(self):
        """Test safe_eval on reasonably long, non-trivial expressions."""
        assert math.isclose(
            safe_eval("(1.602e-19) ** 2 / (4 * math.pi * 8.854e-12 * 1.054e-34 * 2.998e8)"),
            1 / 137.036,
            rel_tol=1e-3,
        )

    def test_safe_eval_invalid(self):
        """Test that safe_eval raises a ValueError on invalid expressions."""
        with pytest.raises(ValueError, match="Invalid expression"):
            safe_eval("1 +")

        with pytest.raises(ValueError, match="Invalid expression"):
            safe_eval("1 + (2 + 3")

        with pytest.raises(ValueError, match="Invalid expression"):
            safe_eval("* 2")

        with pytest.raises(ValueError, match="Invalid expression"):
            safe_eval("1 + x")

        with pytest.raises(ValueError, match="Invalid expression"):
            safe_eval("1 + math.sin")

        with pytest.raises(ValueError, match="Invalid expression"):
            program = "for i in [1, 2, 3]:" "    print(i)"
            safe_eval(program)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
