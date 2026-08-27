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

from unittest.mock import MagicMock

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
from pennylane import qnode
from pennylane.decomposition import add_decomps, local_decomps, register_resources
from pennylane.typing import Bool, Complex, Float, Int, Wire
from pennylane.wires import Wires

from catalyst import qjit
from catalyst.decomposition.decomposition_rules import (
    compile_decomposition_rules_wrapper,
    name_unwrap_adjoint,
    name_wrap_adjoint,
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
            (
                HybridWires(Wires([0, 1, 2])),
                "HybridWires{}{}{}[",
            ),  # NOTE: open brace to match uid
            (
                HybridOpArg(Float, StaticData("innerop", Wires(0)), Wires([2, 3]), 12),
                "HybridOpArg{angle:[[f64]]}{cwires:2}{}[",  # NOTE: open brace to match uid
            ),
            (
                qp.Rot(Bool, Int, Float, Wires(0)),
                "Rot{0:[f64],1:[f64],2:[f64]}{wires:1}{}",
            ),  # custom ops should be promoted to f64
        ],
    )
    def test_GraphOpId(self, op, id):
        """Test that GraphOpIds are generated correctly by the frontend."""
        # NOTE: use startswith to match ops with uids/extra_data
        assert GraphOpID(op).getGraphOpId().startswith(id)

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

    def test_wrapper_passes_compilable_data_to_conditions(self, monkeypatch):
        """Test that decomposition conditions receive compilable operator data."""
        mock_decomp = MagicMock()
        mock_decomp._impl.__name__ = "FakeRuleName"
        mock_decomp.compute_resources.return_value.gate_counts = {}
        mock_decomp.is_applicable.side_effect = (
            lambda *, wires, a, b, thing: a and b == 3.14 and thing == "string"
        )

        monkeypatch.setattr(qp.decomposition, "list_decomps", lambda _: [mock_decomp])

        res = compile_decomposition_rules_wrapper(
            "CompilableData",
            "CompilableData{}{wires:2}{a:True,b:3.14,thing:string}",
            {},
            {"wires": 2},
            {"a": True, "b": 3.14, "thing": "string"},
        )

        assert isinstance(res, str)

class TestPrecompiled:
    """Tests for precompiled decomposition rules."""


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
        assert 'target_gate = "Adjoint(NoParams){}{reg:2}{}"' in mlir

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
        assert 'target_gate = "Adjoint(NoParams){}{reg:2}{}"' in mlir

    def test_distribution_rule_synthesized_from_base_only(self):
        """With only a base rule registered (no Adjoint(Op) rule), lowering still synthesizes a rule
        for Adjoint(Op) by distributing the base rule over adjoint (case 3): its resources are the
        base resources adjointed and its body is an adjoint region."""
        from operator2_dummy_gates import NoParams

        base_rule, _ = self._base_and_adjoint_rules()
        with local_decomps():
            add_decomps(NoParams, base_rule)  # only a base rule, no Adjoint(NoParams) rule

            @qjit(capture=True, target="mlir")
            @qnode(qp.device("null.qubit", wires=3))
            def circuit():
                NoParams(reg=[0, 1])
                return qp.state()

            mlir = circuit.mlir

        assert 'target_gate = "NoParams{}{reg:2}{}"' in mlir
        # A distribution rule for Adjoint(NoParams) is synthesized even though none was registered.
        assert 'target_gate = "Adjoint(NoParams){}{reg:2}{}"' in mlir
        assert (
            'resources = {operations = {"Adjoint(SingleParam){x:[[f64]]}{reg:2}{}" = 1 : i64}'
            in mlir
        )
        assert "qref.adjoint" in mlir

    def test_no_distribution_rule_for_non_invertible_body(self):
        """A distribution rule is NOT synthesized when the base rule body is non-invertible (contains
        a mid-circuit measurement): the base rule is still lowered, but no Adjoint(Op) rule."""
        from operator2_dummy_gates import NoParams, SingleParam

        def base_resource_fn(reg):
            return {SingleParam(x=Float, reg=Wire[2]): 1}

        @register_resources(base_resource_fn)
        def measuring_rule(reg):
            SingleParam(x=0.1, reg=reg[0:2])
            qp.measure(reg[0])

        with local_decomps():
            add_decomps(NoParams, measuring_rule)

            @qjit(capture=True, target="mlir")
            @qnode(qp.device("null.qubit", wires=3))
            def circuit():
                NoParams(reg=[0, 1])
                return qp.state()

            mlir = circuit.mlir

        assert 'target_gate = "NoParams{}{reg:2}{}"' in mlir
        assert 'target_gate = "Adjoint(NoParams){}{reg:2}{}"' not in mlir


class TestOnDemand:
    """Test the python wrapper functions used for on-demand,
    compile-time decomposition rule lowering.
    """

    @pytest.mark.parametrize(
        "op_id, expected",
        [
            # Normal case: only the name (prefix before the first `{`) is wrapped.
            ("S{}{wires:1}{}", "Adjoint(S){}{wires:1}{}"),
            ("RX{0:[f64]}{wires:1}{}", "Adjoint(RX){0:[f64]}{wires:1}{}"),
            # `[uid]` suffix is carried through untouched.
            ("HybridOp{a:[[f64]]}{w:1}{}[42]", "Adjoint(HybridOp){a:[[f64]]}{w:1}{}[42]"),
            # No `{` present (split == -1): the whole id is the name and gets wrapped as-is.
            ("RX", "Adjoint(RX)"),
            ("", "Adjoint()"),
        ],
    )
    def test_name_wrap_adjoint(self, op_id, expected):
        """name_wrap_adjoint wraps only the operator name; when the id has no `{` (split == -1) the
        whole string is treated as the name and wrapped as-is."""
        assert name_wrap_adjoint(op_id) == expected

    @pytest.mark.parametrize(
        "op_name, op_id, expected",
        [
            ("S", "S{}{wires:1}{}", "S{}{wires:1}{}"),
            ("S", "Adjoint(S){}{wires:1}{}", "S{}{wires:1}{}"),
            ("RX", "Adjoint(RX){0:[f64]}{wires:1}{}", "RX{0:[f64]}{wires:1}{}"),
        ],
    )
    def test_name_unwrap_adjoint(self, op_name, op_id, expected):
        """name_unwrap_adjoint recovers the base op's id from an adjoint graphOpId, and is the
        inverse of name_wrap_adjoint for a base id."""
        if op_id.startswith("Adjoint("):
            assert name_unwrap_adjoint(op_name, op_id) == expected
            assert name_wrap_adjoint(expected) == op_id
        else:
            with pytest.raises(ValueError, match="not an adjoint id"):
                name_unwrap_adjoint(op_name, op_id)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
