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
    """Tests of trace-time decomposition rule lowering."""

    @staticmethod
    def _base_and_adjoint_rules():
        from operator2_dummy_gates import SingleParam

        def base_resource_fn(reg):
            return {SingleParam(x=Float, reg=Wire[2]): 1}

        @register_resources(base_resource_fn)
        def base_rule(reg):
            SingleParam(x=0.1, reg=reg[0:2])

        def adj_resource_fn(reg):
            return {SingleParam(x=Float, reg=Wire[2]): 2}

        @register_resources(adj_resource_fn)
        def adj_rule(reg):
            SingleParam(x=0.2, reg=reg[0:2])
            SingleParam(x=0.3, reg=reg[0:2])

        return base_rule, adj_rule

    def test_plain_gate_captures_base_and_adjoint(self):
        """Lowering a plain gate captures the rules registered against both the gate
        and its adjoint."""
        from operator2_dummy_gates import NoParams

        base_rule, adj_rule = self._base_and_adjoint_rules()
        with local_decomps():
            add_decomps(NoParams, base_rule)
            add_decomps("Adjoint(NoParams)", adj_rule)

            @qjit(capture=True, target="mlir")
            @qnode(qp.device("null.qubit", wires=3))
            def circuit():
                NoParams(reg=[0, 1])
                return qp.state()

            mlir = circuit.mlir

        assert 'target_gate = "NoParams{}{reg:2}{}"' in mlir
        assert 'target_gate = "Adjoint(NoParams{}{reg:2}{})"' in mlir

    def test_adjoint_gate_captures_base_and_adjoint(self):
        """Lowering the Adjoint of a gate captures the rules registered against both the plain gate
        and its adjoint."""
        from operator2_dummy_gates import NoParams

        base_rule, adj_rule = self._base_and_adjoint_rules()
        with local_decomps():
            add_decomps(NoParams, base_rule)
            add_decomps("Adjoint(NoParams)", adj_rule)

            @qjit(capture=True, target="mlir")
            @qnode(qp.device("null.qubit", wires=3))
            def circuit():
                qp.adjoint(NoParams(reg=[0, 1]))
                return qp.state()

            mlir = circuit.mlir

        assert 'qref.operator "NoParams"() adj' in mlir
        assert 'target_gate = "NoParams{}{reg:2}{}"' in mlir
        assert 'target_gate = "Adjoint(NoParams{}{reg:2}{})"' in mlir

    def test_no_adjoint_rule_registered(self):
        """When no Adjoint(Op) rule is registered, only the base rule is lowered (default)."""
        from operator2_dummy_gates import NoParams

        base_rule, _ = self._base_and_adjoint_rules()
        with local_decomps():
            add_decomps(NoParams, base_rule)

            @qjit(capture=True, target="mlir")
            @qnode(qp.device("null.qubit", wires=3))
            def circuit():
                NoParams(reg=[0, 1])
                return qp.state()

            mlir = circuit.mlir

        assert 'target_gate = "NoParams{}{reg:2}{}"' in mlir
        assert "Adjoint(" not in mlir


class TestOnDemand:
    """Test the python wrapper functions used for on-demand, compile-time decomposition rule lowering."""


if __name__ == "__main__":
    pytest.main(["-x", __file__])
