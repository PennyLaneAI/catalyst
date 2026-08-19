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

import jax.numpy as jnp
import pennylane as qp
import pytest
from jax.core import ShapedArray
from operator2_dummy_gates import (
    CompilableData,
    HybridOpArg,
    HybridWires,
    MultiParams,
    MultipleRegisters,
    NoParams,
    NoParamsCustomOp,
    SingleParam,
    StaticData,
)
from pennylane.typing import Complex, Float, Int
from pennylane.wires import Wires

from catalyst.decomposition.decomposition_rules import (
    compile_decomposition_rules_wrapper,
)
from catalyst.decomposition.graph_op_id import GraphOpID
from catalyst.decomposition.type_utils import (
    convert_types_to_mlir_strings,
    get_dummy_values_for_arg,
)


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
            ([["f64", "f64"], ["f64", "f64"]], "float64", (2, 2)),
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
            ({"name": qp.typing.Float}, {"name": ["f64"]}),
            ({"name": qp.typing.Int}, {"name": ["i64"]}),
            ({"test": qp.typing.Bool}, {"test": ["i1"]}),
            ({"r": qp.typing.Complex}, {"r": ["complex<f64>"]}),
            ({"A": qp.typing.AbstractArray((2,), "int32")}, {"A": ["i32", "i32"]}),
        ],
    )
    def test_mlir_stringify_type(self, dtype, expected):
        """Test mlir_stringify_type."""
        assert convert_types_to_mlir_strings(dtype) == expected

    @pytest.mark.parametrize(
        "op, id",
        [
            (NoParams(Wires(0)), "NoParams{}{reg:1}{}"),
            (NoParamsCustomOp(Wires([0, 1])), "NoParamsCustomOp{}{wires:2}{}"),
            (SingleParam(Float, Wires([2, 3])), "SingleParam{x:[[f64]]}{reg:2}{}"),
            (
                CompilableData(True, 3.14, "string", Wires([0, 1])),
                "CompilableData{}{wires:2}{a:True,b:3.14,thing:string}",
            ),
            (
                MultipleRegisters(Wires([0, 1, 2]), Wires([3, 4])),
                "MultipleRegisters{}{reg1:3,reg2:2}{}",
            ),
            (
                MultiParams(Wires([0, 2, 3]), Complex, Int, Float[2]),
                "MultiParams{a:[[complex<f64>]],b:[[i64]],c:[[f64,f64]]}{reg:3}",
            ),
            (qp.MultiRZ(Float, Wires([0, 2, 3, 4])), "MultiRZ{theta:[f64]}{wires:4}{}"),
            (
                qp.PauliRot(Float, "XYZ", Wires([1, 2, 3])),
                "PauliRot{theta:[f64]}{wires:3}{pauli_word:XYZ}",
            ),
            (StaticData("mylabel", Wires([0, 1])), "StaticData{}{reg:2}{}["),
            (HybridWires(Wires([0, 1, 2])), "HybridWires{}{}{}["),  # NOTE: open brace to match uid
            (
                HybridOpArg(Float, StaticData("innerop", Wires(0)), Wires([2, 3]), 12),
                "HybridOpArg{angle:[[f64]]}{cwires:2}{}[",  # NOTE: open brace to match uid
            ),
        ],
    )
    def test_GraphOpId(self, op, id):
        """Test that GraphOpIds are generated correctly by the frontend."""
        # NOTE: use startswith to match ops with uids/extra_data
        assert GraphOpID(op).getGraphOpId().startswith(id)

    def test_wrapper_operator(self):
        """Test that compile_decomposition_rules_wrapper doesn't error on Operator1 instances."""
        # TODO: keep this up to date with an operator that is not migrated, and decomposes to
        # un-migrated operators until migration is complete.
        with pytest.warns(match="Failed to get resources"):
            compile_decomposition_rules_wrapper(
                "PauliX", 'PauliX{}{"wires":1}{}', {}, {"wires": 1}, {}
            )


class TestPrecompiled:
    """Tests for precompiled decomposition rules."""


class TestTraceTime:
    """Placeholder for future tests of trace-time decomposition rule lowering."""


class TestOnDemand:
    """Test the python wrapper functions used for on-demand, compile-time decomposition rule lowering."""


if __name__ == "__main__":
    pytest.main(["-x", __file__])
