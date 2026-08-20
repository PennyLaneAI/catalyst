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

import jax.numpy as jnp
import pennylane as qp
import pytest
from jax.core import ShapedArray
from pennylane import qjit, qnode
from pennylane.decomposition import add_decomps, local_decomps, register_resources
from pennylane.typing import Float, Wire

from catalyst.compiler import _quantum_opt
from catalyst.decomposition.decomposition_rules import (
    compile_decomposition_rules_wrapper,
)
from catalyst.decomposition.precompile_decomposition_rules import (
    get_abstract_args,
    precompile_decomp_rules,
)
from catalyst.decomposition.type_utils import (
    get_dummy_values_for_arg,
    mlir_stringify_type,
)
from catalyst.utils.runtime_environment import BYTECODE_FILE_PATH


class TestGenericUtilities:
    """Tests for common decomposition rule lowering utilities."""

    @pytest.mark.parametrize(
        "input, dtype, shape",
        [
            # python-type scalar tests
            (int, "int64", ()),
            (float, "float64", ()),
            (jnp.dtype("int32"), "int32", ()),
            (bool, "bool", ()),
            (complex, "complex128", ()),
            # mlir-type scalar tests
            ("i1", "bool", ()),
            ("i32", "int32", ()),
            ("f64", "float64", ()),
            ("complex<f64>", "complex128", ()),
            ("complex<f32>", "complex64", ()),
            # python-type shaped tests
            (bool, "bool", ()),
            ([float, float], "float64", (2,)),
            ([int], "int64", (1,)),
            (ShapedArray((4,), "int32"), "int32", (4,)),
            # mlir-type shaped tests
            ("i32", "int32", ()),
            (["f64", "f64"], "float64", (2,)),
            (["i1", "i1", "i1"], "bool", (3,)),
        ],
    )
    def test_get_dummy_values_types(self, input, dtype, shape):
        """Test that get_dummy_values_for_container handles MLIR and Python types correctly."""
        result = get_dummy_values_for_arg(input)
        assert result.dtype == dtype
        assert result.shape == shape

    @pytest.mark.parametrize(
        "dtype, expected",
        [
            (qp.typing.Float, "[f64]"),
            (qp.typing.Int, "[i64]"),
            (qp.typing.Bool, "[i1]"),
            (qp.typing.Complex, "[complex<f64>]"),
            (qp.typing.AbstractArray((2,), "int32"), "[i32,i32]"),
        ],
    )
    def test_mlir_stringify_type(self, dtype, expected):
        """Test mlir_stringify_type."""
        assert mlir_stringify_type(dtype) == expected

    def test_wrapper_operator(self, mocker):
        """Test that compile_decomposition_rules_wrapper doesn't error on Operator1 instances."""
        mock_decomp = mocker.MagicMock()
        mock_decomp._impl.__name__ = "FakeRuleName"
        mock_decomp.compute_resources.side_effect = ValueError("Fake Resource Related Error")

        mocker.patch("pennylane.decomposition.list_decomps", return_value=[mock_decomp])

        with pytest.warns(match="Failed to get resources"):
            res = compile_decomposition_rules_wrapper(
                "MockOp", 'MockOp{}{"wires":1}{}', {}, {"wires": 1}, {}
            )
        assert isinstance(res, str)


class TestPrecompiled:
    """Tests for precompiled decomposition rules."""


class TestTraceTime:
    """Placeholder for future tests of trace-time decomposition rule lowering."""


class TestOnDemand:
    """Test the python wrapper functions used for on-demand, compile-time decomposition rule lowering."""


if __name__ == "__main__":
    pytest.main(["-x", __file__])
