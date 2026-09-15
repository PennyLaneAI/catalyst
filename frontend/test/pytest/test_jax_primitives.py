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

"""
Unit tests for Catalyst primitives in JAX.
"""

import platform
from collections import namedtuple

import jax
import pytest
from jax import make_jaxpr
from jax._src.lib.mlir import ir
from jax.interpreters.mlir import ir_constant, make_ir_context

from catalyst.jax_primitives import (
    _qextract_lowering,
    _qinsert_lowering,
    extract_scalar,
    get_call_jaxpr,
    safe_cast_to_f64,
    set_estimated_iterations_attr,
    unconditional_to_conditional_if_probs,
)

# Fake some arguments used by functions to be tested.
JAXCTX = namedtuple("jax_ctx", ["module_context"])
MODCTX = namedtuple("module_context", ["context"])
VALUE = namedtuple("qubit", ["type"])


@pytest.mark.skipif(platform.system() == "Linux", reason="undiagnosed segfault")
class TestLowering:
    """Test lowering methods for JAX primitives to MLIR."""

    @pytest.mark.parametrize("test_input", (1.0, 2j))
    def test_extract_wire_type_error(self, test_input):
        """Test that unsupported wire types raise an appropriate error."""

        ctx = make_ir_context()
        jax_ctx = JAXCTX(MODCTX(ctx))
        with ir.Location.unknown(ctx):
            index_value = ir_constant(test_input)
            qreg_value = VALUE(ir.OpaqueType.get("quantum", "reg"))
            with pytest.raises(TypeError, match="Operator wires expected to be integers"):
                _qextract_lowering(jax_ctx, qreg_value, index_value)

    @pytest.mark.parametrize("test_input", (1.0, 2j))
    def test_insert_wire_type_error(self, test_input):
        """Test that unsupported wire types raise an appropriate error."""

        ctx = make_ir_context()
        jax_ctx = JAXCTX(MODCTX(ctx))
        with ir.Location.unknown(ctx):
            index_value = ir_constant(test_input)
            qreg_value = VALUE(ir.OpaqueType.get("quantum", "reg"))
            qbit_value = VALUE(ir.OpaqueType.get("quantum", "bit"))
            with pytest.raises(TypeError, match="Operator wires expected to be integers"):
                _qinsert_lowering(jax_ctx, qreg_value, index_value, qbit_value)


@pytest.mark.skipif(platform.system() in ["Linux", "Darwin"], reason="undiagnosed segfault")
class TestHelpers:
    """Test helper methods for primitive lowerings."""

    @pytest.mark.parametrize(
        "test_input",
        (
            True,
            0,
            1.0,
            jax.numpy.array(2, dtype=jax.numpy.int8),
            jax.numpy.array(3, dtype=jax.numpy.int64),
            jax.numpy.array(4.0, dtype=jax.numpy.float32),
            jax.numpy.array(5.0, dtype=jax.numpy.float64),
        ),
    )
    def test_float_casting(self, test_input):
        """Test that float conversion works."""

        ctx = make_ir_context()
        with ir.Location.unknown(ctx):
            ir_value = ir_constant(test_input)
            casted_value = safe_cast_to_f64(ir_value, "TestOp")
            assert ir.RankedTensorType.isinstance(casted_value.type)
            tensor_ty = ir.RankedTensorType(casted_value.type)
            assert ir.FloatType.isinstance(tensor_ty.element_type)
            float_ty = ir.FloatType(tensor_ty.element_type)
            assert float_ty.width == 64

    @pytest.mark.parametrize(
        "test_input",
        (
            jax.numpy.array(0j, dtype=jax.numpy.complex64),
            jax.numpy.array(1j, dtype=jax.numpy.complex128),
        ),
    )
    def test_float_casting_error(self, test_input):
        """Test that float conversion raises an appropriate error."""

        ctx = make_ir_context()
        with ir.Location.unknown(ctx):
            ir_value = ir_constant(test_input)
            with pytest.raises(TypeError, match="Operator TestOp expected a float64 value"):
                safe_cast_to_f64(ir_value, "TestOp", "value")

    @pytest.mark.parametrize("test_input", (0, jax.numpy.array(1), jax.numpy.array([2])))
    def test_scalar_extraction(self, test_input):
        """Test that scalar extraction works."""

        ctx = make_ir_context()
        with ir.Location.unknown(ctx):
            ir_value = ir_constant(test_input)
            extracted_value = extract_scalar(ir_value, "TestOp")
            assert ir.IntegerType.isinstance(extracted_value.type)

    @pytest.mark.parametrize("test_input", (jax.numpy.array([[3]]), jax.numpy.array([4, 5])))
    def test_scalar_extraction_error(self, test_input):
        """Test that scalar extraction raises an appropriate error."""

        ctx = make_ir_context()
        with ir.Location.unknown(ctx):
            ir_value = ir_constant(test_input)
            with pytest.raises(TypeError, match="Operator TestOp expected a scalar value"):
                extract_scalar(ir_value, "TestOp", "value")


class TestUnconditionalToConditionalIfProbs:
    """Unit tests for the unconditional->conditional branch-probability conversion used when
    lowering ``cond``/``switch`` resource hints to nested ``scf.if`` probabilities."""

    def test_none_passthrough(self):
        """``None`` (no hint provided) is passed through unchanged."""
        assert unconditional_to_conditional_if_probs(None) is None

    def test_empty(self):
        """An empty sequence converts to an empty tuple."""
        assert unconditional_to_conditional_if_probs(()) == ()

    @pytest.mark.parametrize(
        "unconditional, expected",
        [
            ((0.3,), (0.3,)),
            ((0.2, 0.6, 0.1), (0.2, 0.75, 0.5)),
            ((0.5, 0.25), (0.5, 0.5)),
        ],
    )
    def test_conversion(self, unconditional, expected):
        """Unconditional per-branch probabilities convert to the expected conditional ones."""
        result = unconditional_to_conditional_if_probs(unconditional)
        assert result == pytest.approx(expected)

    def test_exhausted_mass_yields_zero(self):
        """Once the probability mass is fully consumed, later branches are unreachable and
        receive a conditional probability of 0 (rather than dividing by zero)."""
        result = unconditional_to_conditional_if_probs((0.5, 0.5, 0.3))
        assert result == pytest.approx((0.5, 1.0, 0.0))

    @pytest.mark.parametrize(
        "unconditional",
        [(0.2, 0.6, 0.1), (0.5, 0.25), (0.1, 0.1, 0.1, 0.1), (1.0,)],
    )
    def test_roundtrip_recovers_unconditional(self, unconditional):
        """Applying the conditional probabilities as a cascade (the way the resource analysis
        weights branches) must recover the original unconditional probabilities."""
        conditional = unconditional_to_conditional_if_probs(unconditional)

        recovered = []
        remaining = 1.0
        for cond_prob in conditional:
            weight = remaining * cond_prob
            recovered.append(weight)
            remaining -= weight

        assert tuple(recovered) == pytest.approx(unconditional)


def test_set_estimated_iterations_attr_rejects_negative():
    """A negative ``estimated_iterations`` is rejected before any MLIR attribute is set, so no
    operation is required to trigger the guard."""
    with pytest.raises(ValueError, match="'estimated_iterations' must be non-negative"):
        set_estimated_iterations_attr(None, -1.0)


def test_get_call_jaxpr():
    """Test get_call_jaxpr raises AsserionError if no function primitive exists."""

    def f(x):
        return x * x

    jaxpr = make_jaxpr(f)(2.0)
    with pytest.raises(AssertionError, match="No call_jaxpr found in the JAXPR"):
        _ = get_call_jaxpr(jaxpr)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
