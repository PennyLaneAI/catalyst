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

from catalyst.compiler import _quantum_opt
from catalyst.decomposition.decomposition_rules import (
    compile_decomposition_rules_wrapper,
)
from catalyst.decomposition.precompile_decomposition_rules import (
    get_abstract_args,
    precompile_decomp_rules,
)
from catalyst.decomposition.type_utils import get_dummy_values_for_container, mlir_stringify_type
from catalyst.utils.runtime_environment import BYTECODE_FILE_PATH


class TestGenericUtilities:
    """Tests for common decomposition rule lowering utilities."""

    def test_get_dummy_values_types(self):
        """Test that get_dummy_values_for_container handles MLIR and Python types correctly."""
        python_types = [int, float, jnp.dtype("int32"), bool, complex]
        result = get_dummy_values_for_container(python_types)

        assert result[0].dtype == "int64"
        assert result[1].dtype == "float64"
        assert result[2].dtype == "int32"
        assert result[3].dtype == "bool"
        assert result[4].dtype == "complex128"

        mlir_types = ["i1", "i32", "f64", "complex<f64>", "complex<f128>"]
        result = get_dummy_values_for_container(mlir_types)

        assert result[0].dtype == "bool"
        assert result[1].dtype == "int32"
        assert result[2].dtype == "float64"
        assert result[3].dtype == "complex64"
        assert result[4].dtype == "complex128"

        string_type = "f64"
        result = get_dummy_values_for_container(string_type)

        assert result.dtype == "float64"

    def test_get_dummy_values_shapes(self):
        """Test that get_dummy_values_for_container handles MLIR and python shapes correctly."""
        python_shapes = [bool, [float, float], [int], ShapedArray((4,), "int32")]
        result = get_dummy_values_for_container(python_shapes)

        assert result[0].shape == ()
        assert result[1].shape == (2,)
        assert result[2].shape == (1,)
        assert result[3].shape == (4,)

        mlir_types = ["i32", ["f64", "f64"], ["i1", "i1", "i1"]]
        result = get_dummy_values_for_container(mlir_types)

        assert result[0].shape == ()
        assert result[1].shape == (2,)
        assert result[2].shape == (3,)

    def test_mlir_stringify_type(self):
        """Test mlir_stringify_type."""
        assert mlir_stringify_type(qp.typing.Float) == "[f64]"
        assert mlir_stringify_type(qp.typing.Int) == "[i64]"
        assert mlir_stringify_type(qp.typing.Bool) == "[i1]"
        assert mlir_stringify_type(qp.typing.Complex) == "[complex<f128>]"
        assert mlir_stringify_type(qp.typing.AbstractArray((2,), "int32")) == "[i32,i32]"


class TestPrecompiled:
    """Tests for precompiled decomposition rules."""

    def test_bytecode_file(self):
        """Test that the bytecode file is generated correctly."""
        # orig_bcfile = Path(BYTECODE_FILE_PATH)
        # tmp_bcfile = None
        #
        # if orig_bcfile.exists():
        #     tmp_bcfile = orig_bcfile.replace(BYTECODE_FILE_PATH + ".tmpbackup")
        #
        # try:
        #     precompile_decomp_rules()
        #     assert orig_bcfile.exists()
        #
        # finally:
        #     if tmp_bcfile:
        #         tmp_bcfile = tmp_bcfile.replace(orig_bcfile)
        #     else:
        #         orig_bcfile.unlink(missing_ok=True)
        #
        # # NOTE: empty pass is needed to prevent running default pipeline
        # rules = _quantum_opt("--empty", BYTECODE_FILE_PATH)
        #
        # assert "_isingxy_to_h_cy" in rules
        # assert "_doublexcit" in rules
        # assert "_pauliz_to_ps" in rules
        # assert "_cphase_to_ppr" in rules
        # assert "_crot" in rules
        pass


class TestTraceTime:
    """Placeholder for future tests of trace-time decomposition rule lowering."""


class TestOnDemand:
    """
    Test the python wrapper functions used for on-demand, compile-time decomposition rule lowering.
    """


if __name__ == "__main__":
    pytest.main(["-x", __file__])
