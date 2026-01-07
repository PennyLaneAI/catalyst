# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for xDSL utilities."""
# pylint: disable=line-too-long

import pytest
from xdsl.dialects import arith, builtin, tensor, test

from catalyst.python_interface.dialects.stablehlo import ConstantOp as hloConstantOp
from catalyst.python_interface.utils import get_constant_from_ssa, get_pyval_from_xdsl_attr

pytestmark = pytest.mark.xdsl


class TestGetConstantFromSSA:
    """Unit tests for ``get_constant_from_ssa``."""

    def test_non_constant(self):
        """Test that ``None`` is returned if the input is not a constant."""
        val = test.TestOp(result_types=(builtin.Float64Type(),)).results[0]
        assert get_constant_from_ssa(val) is None

    @pytest.mark.parametrize(
        "const, attr_type, dtype",
        [
            (11, builtin.IntegerAttr, builtin.IntegerType(64)),
            (5, builtin.IntegerAttr, builtin.IndexType()),
            (2.5, builtin.FloatAttr, builtin.Float64Type()),
        ],
    )
    def test_scalar_constant_arith(self, const, attr_type, dtype):
        """Test that constants created by ``arith.constant`` are returned correctly."""
        const_attr = attr_type(const, dtype)
        val = arith.ConstantOp(value=const_attr).results[0]

        assert get_constant_from_ssa(val) == const

    @pytest.mark.parametrize(
        "const, elt_type",
        [
            (11, builtin.IntegerType(64)),
            (9, builtin.IndexType()),
            (2.5, builtin.Float64Type()),
            (-1.1 + 2.3j, builtin.ComplexType(builtin.Float64Type())),
        ],
    )
    @pytest.mark.parametrize("constant_op", [arith.ConstantOp, hloConstantOp])
    def test_scalar_constant_extracted_from_rank0_tensor(self, const, elt_type, constant_op):
        """Test that constants created by ``stablehlo.constant`` are returned correctly."""
        data = const
        if isinstance(const, complex):
            # For complex numbers, the number must be split into a 2-tuple containing
            # the real and imaginary part when initializing a dense elements attr.
            data = (const.real, const.imag)

        dense_attr = builtin.DenseIntOrFPElementsAttr.from_list(
            type=builtin.TensorType(element_type=elt_type, shape=()),
            data=(data,),
        )
        tensor_ = constant_op(value=dense_attr).results[0]
        val = tensor.ExtractOp(tensor=tensor_, indices=[], result_type=elt_type).results[0]

        assert get_constant_from_ssa(val) == const

    def test_tensor_constant_arith(self):
        """Test that ``None`` is returned if the input is a tensor created by ``arith.constant``."""
        dense_attr = builtin.DenseIntOrFPElementsAttr.from_list(
            type=builtin.TensorType(element_type=builtin.Float64Type(), shape=(3,)),
            data=(1, 2, 3),
        )
        val = arith.ConstantOp(value=dense_attr).results[0]

        assert get_constant_from_ssa(val) is None

    def test_tensor_constant_stablehlo(self):
        """Test that ``None`` is returned if the input is a tensor created by ``stablehlo.constant``."""
        dense_attr = builtin.DenseIntOrFPElementsAttr.from_list(
            type=builtin.TensorType(element_type=builtin.Float64Type(), shape=(3,)),
            data=(1.0, 2.0, 3.0),
        )
        val = hloConstantOp(value=dense_attr).results[0]

        assert get_constant_from_ssa(val) is None

    def test_extract_scalar_from_constant_tensor_stablehlo(self):
        """Test that ``None`` is returned if the input is a scalar constant, but it was extracted
        from a non-scalar constant."""
        # Index SSA value to be used for extracting a value from a tensor
        dummy_index = test.TestOp(result_types=(builtin.IndexType(),)).results[0]

        dense_attr = builtin.DenseIntOrFPElementsAttr.from_list(
            type=builtin.TensorType(element_type=builtin.Float64Type(), shape=(3,)),
            data=(1.0, 2.0, 3.0),
        )
        tensor_ = hloConstantOp(value=dense_attr).results[0]
        val = tensor.ExtractOp(
            tensor=tensor_, indices=[dummy_index], result_type=builtin.Float64Type()
        ).results[0]
        # val is a value that we got by indexing into a constant tensor with rank >= 1
        assert isinstance(val.type, builtin.Float64Type)

        assert get_constant_from_ssa(val) is None


class TestGetPyvalFromXdslAttr:
    """Unit tests for ``get_pyval_from_xdsl_attr``."""

    @pytest.mark.parametrize("bitwidth", [16, 32, 64])
    def test_int(self, bitwidth):
        """Test that integer attributes are converted correctly."""
        in_val = 1234
        attr = builtin.IntegerAttr(in_val, bitwidth)
        val = get_pyval_from_xdsl_attr(attr)

        assert isinstance(val, int)
        assert val == in_val

    @pytest.mark.parametrize("in_val", [True, False])
    def test_bool(self, in_val):
        """Test that boolean attributes are converted correctly."""
        attr = builtin.IntegerAttr.from_bool(in_val)
        val = get_pyval_from_xdsl_attr(attr)

        assert isinstance(val, bool)
        assert val == in_val

    @pytest.mark.parametrize("bitwidth", [16, 32, 64])
    def test_float(self, bitwidth):
        """Test that float attributes are converted correctly."""
        in_val = 1.56
        attr = builtin.FloatAttr(in_val, bitwidth)
        val = get_pyval_from_xdsl_attr(attr)

        assert isinstance(val, float)
        assert round(val, 2) == in_val

    def test_string(self):
        """Test that string attributes are converted correctly."""
        in_val = "test_string"
        attr = builtin.StringAttr(in_val)
        val = get_pyval_from_xdsl_attr(attr)

        assert isinstance(val, str)
        assert val == in_val

    def test_array(self):
        """Test that array attributes are converted correctly."""
        in_val = (1, 2, 3, 4)
        attr = builtin.ArrayAttr([builtin.IntegerAttr(v, 64) for v in in_val])
        val = get_pyval_from_xdsl_attr(attr)

        assert isinstance(val, tuple)
        assert all(isinstance(v, int) for v in val)
        assert val == in_val

    def test_dict(self):
        """Test that dict attributes are converted correctly."""
        in_val = {"a": 1, "b": 2, "c": 3}
        attr = builtin.DictionaryAttr({k: builtin.IntegerAttr(v, 64) for k, v in in_val.items()})
        val = get_pyval_from_xdsl_attr(attr)

        assert isinstance(val, dict)
        assert all(isinstance(k, str) for k in val.keys())
        assert all(isinstance(v, int) for v in val.values())
        assert val == in_val

    def test_nested_containers(self):
        """Test that nested container attributes are converted correctly."""
        expected_val = {"a": (1, 2, 3), "b": {"c": 1.5, "d": (False, True)}, "e": "test_string"}
        attr = builtin.DictionaryAttr(
            {
                "a": builtin.ArrayAttr(
                    [
                        builtin.IntegerAttr(1, 64),
                        builtin.IntegerAttr(2, 64),
                        builtin.IntegerAttr(3, 64),
                    ]
                ),
                "b": builtin.DictionaryAttr(
                    {
                        "c": builtin.FloatAttr(1.5, 64),
                        "d": builtin.ArrayAttr(
                            [
                                builtin.IntegerAttr.from_bool(False),
                                builtin.IntegerAttr.from_bool(True),
                            ]
                        ),
                    }
                ),
                "e": builtin.StringAttr("test_string"),
            }
        )
        val = get_pyval_from_xdsl_attr(attr)

        # Verify types
        assert isinstance(val, dict)

        assert isinstance(val["a"], tuple)
        assert all(isinstance(v, int) for v in val["a"])

        assert isinstance(val["b"], dict)
        assert isinstance(val["b"]["c"], float)
        assert isinstance(val["b"]["d"], tuple)
        assert all(isinstance(v, bool) for v in val["b"]["d"])

        assert isinstance(val["e"], str)

        assert val == expected_val

    def test_unsupported_attr(self):
        """Test that trying to convert an unsupported attribute raises an error."""
        attr = builtin.DenseIntElementsAttr.from_list(
            builtin.TensorType(builtin.IndexType(), ()), [123]
        )
        with pytest.raises(ValueError, match="cannot be converted to a Python value"):
            _ = get_pyval_from_xdsl_attr(attr)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
