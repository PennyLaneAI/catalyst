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
from catalyst.decomposition.type_utils import get_dummy_values_for_arg, mlir_stringify_type
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
            ("complex<f64>", "complex64", ()),
            ("complex<f128>", "complex128", ()),
            # shaped tests
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

    def test_for_loop(self):
        class TestOp(qp.core.Operator2):
            dynamic_argnames = ("angles",)

            def __init__(self, angles, wires):
                super().__init__(angles, wires)

        class TestRX(qp.core.Operator2):
            dynamic_argnames = ("theta",)
            wires_argnames = ("wires",)

            arg_specs = {"theta": Float, "wires": Wire[1]}

            def __init__(self, theta, wires):
                super().__init__(theta, wires)

        @register_resources(lambda angles, wires: {TestRX(Float, Wire[1]): len(wires)})
        def test_rule(angles, wires):
            @qp.for_loop(len(wires))
            def l(i):
                TestRX(angles[i], wires[i])

            l()  # pylint: disable=no-value-for-parameter

        with local_decomps():
            add_decomps(TestOp, test_rule)

            assert "scf.for" in compile_decomposition_rules_wrapper(
                "TestOp", "TestID", {"angles": ["f64", "f64", "f64"]}, {"wires": 3}, {}
            )


if __name__ == "__main__":
    pytest.main(["-x", __file__])
