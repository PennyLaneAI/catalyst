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
import numpy as np
import pennylane as qp
import pytest
from jax.core import ShapedArray
from operator2_dummy_gates import (
    ArrayData,
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
from pennylane.core.operator import abstractify
from pennylane.decomposition import (
    add_decomps,
    local_decomps,
    register_condition,
    register_resources,
)
from pennylane.ops.op_math.adjoint2 import Adjoint2
from pennylane.ops.op_math.controlled2 import ControlledOp2
from pennylane.typing import Bool, Complex, Float, Int, Wire
from pennylane.wires import Wires

from catalyst import qjit
from catalyst.decomposition import GraphOpID, RuleLoweringWarning
from catalyst.decomposition.decomposition_rules import (
    _MODIFIER_CANONICAL_ORDER,
    _control_modifier,
    _leading_modifier_kind,
    _modifier_kind,
    collect_symbolic_resources,
    compile_decomposition_rules_wrapper,
    compile_reachable_decomposition_rules_wrapper,
    compile_registered_symbolic_rules,
    get_rule_strings_from_module,
    name_unwrap_adjoint,
    name_unwrap_control,
    name_wrap_adjoint,
    prepare_dynamic_op_kwargs,
    wrap_modifier_id,
)
from catalyst.decomposition.graph_op_id import GraphOpID, build_graph_op_id
from catalyst.decomposition.type_utils import (
    convert_item_to_mlir_type,
    get_dummy_values_for_arg,
    replace_wires_with_placeholder_wires,
)
from catalyst.passes import graph_decomposition
from catalyst.utils.exceptions import CompileError


class TestGenericUtilities:
    """Tests for common decomposition rule lowering utilities."""

    def test_probe_wires_dont_overlap(self):
        """Test that the helper for generating probe arguments doesnt create
        overlapping wires.

        NOTE: Regression test for the accumulator change made in #3225.
        """

        kwargs = prepare_dynamic_op_kwargs({}, wire_lens={"target": 3, "control": 2})
        assert len(kwargs["target"]) == 3
        assert len(kwargs["control"]) == 2

        combined_wires = np.concatenate(
            [np.asarray(kwargs["target"]), np.asarray(kwargs["control"])]
        )
        # Assert they are all negative and unique
        assert np.all(combined_wires < 0)
        assert len(np.unique(combined_wires)) == 5

    def test_wires_replacement_doesnt_create_overlapping_wire_labels(self):
        """Test that the helper does not create overlapping wire labels which create
        validation failures when the operator is unflattened.

        NOTE: Regression test for the accumulator change made in #3214.
        """

        op = qp.ctrl(qp.S(Wire[1]), Wire[1])
        new_op = replace_wires_with_placeholder_wires(op)
        assert new_op == qp.ctrl(qp.S(-2), -1)

    def test_build_graph_op_id(self):
        """The shared builder canonicalizes every frontend identity component."""
        op_id = build_graph_op_id(
            "Example",
            {"z": ["f64"], "a": ["i1"]},
            {"right": 2, "left": 1},
            {"label": "value"},
            adjoint=True,
            num_controls=2,
            uid=7,
        )

        assert op_id == '2C(Adjoint(Example)){a:[i1],z:[f64]}{left:1,right:2}{label = "value"}[7]'

    def test_wires_replacement_doesnt_mutate_operator(self):
        """Test that the wires replacement helper does not mutate the incoming operator."""

        original_wires = qp.wires.Wires([0])
        op = qp.RX(0.5, original_wires)
        original_op_wires = op.wires

        # NOTE: PennyLane re-wraps wire arguments on construction, so the stored wires are
        # a different object than original wires.
        original_wire_arg = op.arguments["wires"]

        new_op = replace_wires_with_placeholder_wires(op)

        # Check that the new op received the placeholder wires
        assert new_op.wires == qp.wires.Wires([-1])
        assert new_op is not op

        # Check original operator is not mutated
        assert op.wires == original_wires
        assert op.wires is original_op_wires
        assert op.arguments["wires"] is original_wire_arg

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
            # mlir-type tensor tests
            ("tensor<i1>", "bool", ()),
            ("tensor<1xi1>", "bool", (1,)),
            ("tensor<2xi64>", "int64", (2,)),
            ("tensor<3x4xf64>", "float64", (3, 4)),
            ("tensor<3x4xcomplex<f64>>", "complex128", (3, 4)),
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
            (["tensor<3x4xcomplex<f64>>"], "complex128", (3, 4)),
        ],
    )
    def test_get_dummy_values_types(self, input, dtype, shape):
        """Test that get_dummy_values_for_container handles MLIR and Python types correctly."""
        result = get_dummy_values_for_arg(input)
        assert result.dtype == dtype
        assert result.shape == shape

    @pytest.mark.parametrize(
        "item, is_special_lowering, mlir_type",
        [
            (Float, True, "f64"),  # custom op is always float
            (Float, False, "tensor<f64>"),
            (Int, False, "tensor<i64>"),
            (Bool, False, "tensor<i1>"),
            (Complex, False, "tensor<complex<f64>>"),
            (Float[1], False, "tensor<1xf64>"),
            (Float[2], False, "tensor<2xf64>"),
            (Int[3], False, "tensor<3xi64>"),
            (Bool[4], False, "tensor<4xi1>"),
            (Complex[5], False, "tensor<5xcomplex<f64>>"),
            (Float[2, 2], False, "tensor<2x2xf64>"),
            (Complex[3, 4, 5], False, "tensor<3x4x5xcomplex<f64>>"),
        ],
    )
    def test_convert_item_to_mlir_type(self, item, is_special_lowering, mlir_type):
        assert convert_item_to_mlir_type(item, is_special_lowering) == mlir_type

    @pytest.mark.parametrize(
        "op, id",
        [
            (NoParams(Wires(0)), "NoParams{}{reg:1}{}"),
            (NoParamsCustomOp(Wires([0, 1])), "NoParamsCustomOp{}{wires:2}{}"),
            (SingleParam(Float, Wires([2, 3])), "SingleParam{x:[tensor<f64>]}{reg:2}{}"),
            (
                CompilableData(True, 3.14, "string", Wires([0, 1])),
                'CompilableData{}{wires:2}{a = true, b = 3.140000e+00 : f64, thing = "string"}',
            ),
            (
                ArrayData(np.array([1, 2, 3]), Wires([0, 1])),
                "ArrayData{}{wires:2}{angles = dense<[1, 2, 3]> : tensor<3xi64>}",
            ),
            (
                # An array of one repeated value takes MLIR's splat shorthand in the id too.
                ArrayData(np.array([[7, 7], [7, 7]], dtype=np.int8), Wires([0, 1])),
                "ArrayData{}{wires:2}{angles = dense<7> : tensor<2x2xi8>}",
            ),
            (
                MultipleRegisters(Wires([0, 1, 2]), Wires([3, 4])),
                "MultipleRegisters{}{reg1:3,reg2:2}{}",
            ),
            (
                MultiParams(Wires([0, 2, 3]), Complex, Int, Float[2]),
                "MultiParams{a:[tensor<complex<f64>>],b:[tensor<i64>],c:[tensor<2xf64>]}{reg:3}",
            ),
            (qp.MultiRZ(Float, Wires([0, 2, 3, 4])), "MultiRZ{theta:[f64]}{wires:4}{}"),
            (
                qp.PauliRot(Float, "XYZ", Wires([1, 2, 3])),
                'PauliRot{theta:[f64]}{wires:3}{pauli_word = "XYZ"}',
            ),
            (StaticData("mylabel", Wires([0, 1])), "StaticData{}{reg:2}{}["),
            (
                HybridWires(Wires([0, 1, 2])),
                "HybridWires{}{}{}[",
            ),  # NOTE: open brace to match uid
            (
                HybridOpArg(Float, StaticData("innerop", Wires(0)), Wires([2, 3]), 12),
                "HybridOpArg{angle:[tensor<f64>]}{cwires:2}{}[",  # NOTE: open brace to match uid
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

    @pytest.mark.parametrize(
        "op, expected_adjoint, expected_controls",
        [
            (qp.S(Wire[1]), False, 0),
            (qp.adjoint(qp.S(Wire[1])), True, 0),
            (qp.adjoint(qp.adjoint(qp.S(Wire[1]))), False, 0),  # adjoints cancel
            (qp.ctrl(qp.S(Wire[1]), Wire[2]), False, 2),
            (qp.ctrl(qp.adjoint(qp.S(Wire[1])), Wire[1]), True, 1),
            (qp.adjoint(qp.ctrl(qp.S(Wire[1]), Wire[1])), True, 1),  # the non-canonical nesting
            # A concrete controlled class is an operator in its own right, not a wrapper.
            (qp.CH(Wire[2]), False, 0),
        ],
    )
    def test_peel_modifiers(self, op, expected_adjoint, expected_controls):
        """Test GraphOpID separates a symbolic operator into its base and the modifiers on it."""

        base, adjoint, num_controls = GraphOpID.peel_modifiers(op)

        assert (adjoint, num_controls) == (expected_adjoint, expected_controls)
        assert not isinstance(base, (Adjoint2, ControlledOp2))

    def test_peeled_identity_describes_the_base(self):
        """Test the identity components of a symbolic operator are the base operator's, since the
        base is what the compiler names and what owns the decomposition rules."""

        wrapped = GraphOpID(qp.ctrl(qp.adjoint(qp.RX(Float, Wire[1])), Wire[2]))
        base = GraphOpID(qp.RX(Float, Wire[1]))

        assert wrapped.operator_name == base.operator_name == "RX"
        assert wrapped.wire_lens == base.wire_lens == {"wires": 1}
        assert wrapped.dynamic_shape == base.dynamic_shape
        assert wrapped.is_custom_op == base.is_custom_op
        assert wrapped.getBaseGraphOpId() == base.getGraphOpId() == "RX{0:[f64]}{wires:1}{}"

    @pytest.mark.parametrize(
        "op, caller_adjoint, caller_controls, expected",
        [
            # The operator's own modifiers, with the caller applying none.
            (qp.adjoint(qp.S(Wire[1])), False, 0, "Adjoint(S){}{wires:1}{}"),
            # The caller's modifiers, on a plain operator.
            (qp.S(Wire[1]), True, 2, "2C(Adjoint(S)){}{wires:1}{}"),
            # Both, composed: control outermost, and the two adjoints cancel.
            (qp.adjoint(qp.S(Wire[1])), True, 1, "C(S){}{wires:1}{}"),
            (qp.ctrl(qp.S(Wire[1]), Wire[1]), False, 1, "2C(S){}{wires:1}{}"),
            # Either nesting of the same operator spells the one canonical id.
            (qp.ctrl(qp.adjoint(qp.S(Wire[1])), Wire[1]), False, 0, "C(Adjoint(S)){}{wires:1}{}"),
            (qp.adjoint(qp.ctrl(qp.S(Wire[1]), Wire[1])), False, 0, "C(Adjoint(S)){}{wires:1}{}"),
        ],
    )
    def test_graph_op_id_composes_modifiers(self, op, caller_adjoint, caller_controls, expected):
        """Test the modifiers the caller applies compose with the ones the operator carried rather
        than wrapping an already-wrapped id a second time."""

        op_id = GraphOpID(op).getGraphOpId(adjoint=caller_adjoint, num_controls=caller_controls)

        assert op_id == expected

    def test_wrapper_operator(self, mocker):
        """Test that compile_decomposition_rules_wrapper doesn't error on Operator1 instances."""
        mock_decomp = mocker.MagicMock()
        mock_decomp.name = "FakeRuleName"
        mock_decomp.compute_resources.side_effect = ValueError("Fake Resource Related Error")

        mocker.patch("pennylane.decomposition.list_decomps", return_value=[mock_decomp])

        with pytest.warns(RuleLoweringWarning, match="Failed to get resources"):
            res = compile_decomposition_rules_wrapper(
                "MockOp", 'MockOp{}{"wires":1}{}', {}, {"wires": 1}, {}
            )
        assert isinstance(res, str)

    def test_wrapper_passes_compilable_data_to_conditions(self, mocker):
        """Test that decomposition conditions receive compilable operator data."""
        mock_decomp = mocker.MagicMock()
        mock_decomp.name = "FakeRuleName"
        mock_decomp.compute_resources.return_value.gate_counts = {}
        mock_decomp.is_applicable.side_effect = (
            lambda *, wires, a, b, thing: a and b == 3.14 and thing == "string"
        )

        mocker.patch("pennylane.decomposition.list_decomps", return_value=[mock_decomp])

        res = compile_decomposition_rules_wrapper(
            "CompilableData",
            'CompilableData{}{wires:2}{a = true, b = 3.140000e+00 : f64, thing = "string"}',
            {},
            {"wires": 2},
            {"a": True, "b": 3.14, "thing": "string"},
        )

        assert "FakeRuleName" in res
        mock_decomp.is_applicable.assert_called_once()
        call_kwargs = mock_decomp.is_applicable.call_args.kwargs
        assert call_kwargs["a"] is True
        assert call_kwargs["b"] == 3.14
        assert call_kwargs["thing"] == "string"

        probe_wires = np.asarray(call_kwargs["wires"])
        assert probe_wires.shape == (2,)
        assert np.all(probe_wires < 0)
        assert len(np.unique(probe_wires)) == 2


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
            'resources = {operations = {"Adjoint(SingleParam){x:[tensor<f64>]}{reg:2}{}" = 1 : i64}'
            in mlir
        )
        assert "qref.adjoint" in mlir

    @pytest.mark.filterwarnings("ignore::catalyst.decomposition.RuleLoweringWarning")
    def test_array_static_data_reaches_the_rules(self):
        """An operator holding array static data lowers with its array spelled as a dense
        attribute, and two rules emitting different arrays stay two distinct operators: the search
        for reachable rules compares arrays by contents rather than asking a whole array whether it
        is true."""

        def resource_fn_a(angles, wires):
            return {ArrayData(angles=np.array([1, 2]), wires=Wire[2]): 1}

        @register_resources(resource_fn_a)
        def rule_a(angles, wires):
            ArrayData(angles=np.array([1, 2]), wires=wires)

        def resource_fn_b(angles, wires):
            return {ArrayData(angles=np.array([3, 4]), wires=Wire[2]): 1}

        @register_resources(resource_fn_b)
        def rule_b(angles, wires):
            ArrayData(angles=np.array([3, 4]), wires=wires)

        with local_decomps():
            add_decomps(ArrayData, rule_a, rule_b)

            @qjit(capture=True, target="mlir")
            @qnode(qp.device("null.qubit", wires=2))
            def circuit():
                ArrayData(angles=np.array([5, 5, 5]), wires=[0, 1])
                return qp.state()

            mlir = circuit.mlir

        # The op carries the array exactly as the id spells it, which is what lets the compiler
        # print an id for the op that matches the rules compiled here.
        assert "static_data = {angles = dense<5> : tensor<3xi64>}" in mlir
        assert 'target_gate = "ArrayData{}{wires:2}{angles = dense<5> : tensor<3xi64>}"' in mlir
        assert (
            'target_gate = "ArrayData{}{wires:2}{angles = dense<[1, 2]> : tensor<2xi64>}"' in mlir
        )
        assert (
            'target_gate = "ArrayData{}{wires:2}{angles = dense<[3, 4]> : tensor<2xi64>}"' in mlir
        )

    @pytest.mark.filterwarnings("ignore::catalyst.decomposition.RuleLoweringWarning")
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

    def test_work_wire_rule_that_doesnt_apply_lowers_without_warning(self, recwarn):
        """Tests the whole trace-time path: capture, id generation, and rule
        lowering for an operator whose cheaper rule needs two borrowed work wires lowers cleanly
        when only one is available.

        This is the shape of PennyLane's real work-wire rules (``Select``, ``MultiControlledX``):
        the rule declares its requirement as a condition, and its resource function derives a
        work-wire count by subtraction, which is only meaningful once that condition holds.
        """

        def borrow_resources(reg1, reg2):
            # Two work wires are consumed, any extras are passed on to the sub-decomposition.
            return {qp.X(Wire[1]): 2 + len(Wire[len(reg2) - 2])}

        @register_condition(lambda reg1, reg2: len(reg2) >= 2)
        @register_resources(borrow_resources)
        def borrow_two_work_wires(reg1, reg2):
            qp.X(reg2[0:1])
            qp.X(reg1[0:1])

        @register_resources(lambda reg1, reg2: {qp.X(Wire[1]): 2})
        def no_work_wires(reg1, reg2):
            qp.X(reg1[0:1])
            qp.X(reg1[0:1])

        with local_decomps():
            add_decomps(MultipleRegisters, borrow_two_work_wires, no_work_wires)

            @qjit(capture=True, target="mlir")
            @qnode(qp.device("null.qubit", wires=3))
            def circuit():
                MultipleRegisters(reg1=[0, 1], reg2=[2])
                return qp.state()

            mlir = circuit.mlir

        lowering_warnings = [
            str(w.message) for w in recwarn if issubclass(w.category, RuleLoweringWarning)
        ]

        assert not any("borrow_two_work_wires" in message for message in lowering_warnings)
        assert 'target_gate = "MultipleRegisters{}{reg1:2,reg2:1}{}"' in mlir
        assert "no_work_wires" in mlir
        assert "borrow_two_work_wires" not in mlir


class TestOnDemand:
    """Test the python wrapper functions used for on-demand,
    compile-time decomposition rule lowering.
    """

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

    @pytest.mark.parametrize(
        "op_name, op_id, expected_base_id, expected_n_ctrl",
        [
            ("RX", "C(RX){0:[f64]}{wires:1}{}", "RX{0:[f64]}{wires:1}{}", 1),
            ("RX", "2C(RX){0:[f64]}{wires:1}{}", "RX{0:[f64]}{wires:1}{}", 2),
            ("S", "10C(S){}{wires:1}{}", "S{}{wires:1}{}", 10),  # multi-digit control count
        ],
    )
    def test_name_unwrap_control(self, op_name, op_id, expected_base_id, expected_n_ctrl):
        """name_unwrap_control recovers the base op's id (with its bare name re-prepended) and the
        control count from a controlled graphOpId, and round-trips through wrap_modifier_id."""

        assert name_unwrap_control(op_name, op_id) == (expected_base_id, expected_n_ctrl)
        assert wrap_modifier_id(expected_base_id, _control_modifier(expected_n_ctrl)) == op_id

    def test_name_unwrap_control_rejects_non_control_id(self):
        """A non-controlled id is rejected for the given base op."""

        with pytest.raises(ValueError, match="not a control id"):
            name_unwrap_control("RX", "Adjoint(RX){0:[f64]}{wires:1}{}")

    @pytest.mark.parametrize(
        "op_id, extra_ctrl_target",
        [
            ("C(S){}{wires:1}{}", None),
            # A multi-controlled id recovers n_ctrl=2 and additionally synthesizes the n=1 variant.
            ("2C(S){}{wires:1}{}", 'target_gate = "C(S){}{wires:1}{}"'),
        ],
    )
    def test_reachable_wrapper_controlled_op(self, op_id, extra_ctrl_target):
        """compile_reachable_decomposition_rules_wrapper routes a controlled op-id through
        name_unwrap_controland returns a module that holds both the base op's
        and the ``<n>C(...)`` rule closure."""

        module_str = compile_reachable_decomposition_rules_wrapper(
            "S", op_id, {}, {"wires": 1}, {}, is_custom_op=True
        )
        assert module_str.lstrip().startswith("module")

        assert f'target_gate = "{op_id}"' in module_str
        assert 'target_gate = "S{}{wires:1}{}"' in module_str
        if extra_ctrl_target is not None:
            assert extra_ctrl_target in module_str

    def test_multi_controlled_resource_gets_its_rules(self):
        """A rule whose resource is a multi-controlled op pulls the rules for that ``<n>C(...)``
        node into the closure.

        The closure explores a symbolic resource through its base, so the control count the
        resource carries has to be carried over with it; only ``C(...)`` would be synthesized
        otherwise, leaving the ``2C(...)`` node the resource names without any rule.
        """

        with local_decomps():

            @register_resources(lambda reg: {qp.ctrl(qp.S(Wire[1]), Wire[2]): 1})
            def two_controlled_s(reg):
                qp.ctrl(qp.S(reg[2]), control=[reg[0], reg[1]])

            add_decomps(NoParams, two_controlled_s)

            module_str = compile_reachable_decomposition_rules_wrapper(
                "NoParams", "NoParams{}{reg:3}{}", {}, {"reg": 3}, {}
            )

        # the resource the rule declares ...
        assert '"2C(S){}{wires:1}{}" = 1 : i64' in module_str
        # ... and the rules that decompose it
        assert 'target_gate = "2C(S){}{wires:1}{}"' in module_str

    def test_control_variant_warns_and_skips_on_failure(self, mocker):
        """control_variant_rule_strings warns and skips a rule when it fails to compile."""

        from catalyst.decomposition import decomposition_rules as dr

        mocker.patch.object(dr, "compile_decomposition_rules", side_effect=ValueError("boom"))
        with pytest.warns(RuleLoweringWarning, match="control rules"):
            out = dr.control_variant_rule_strings(
                "S", "S{}{wires:1}{}", [1], {}, {"wires": 1}, {}, is_custom_op=True
            )
        assert out == []

    def test_compile_rules_reports_missing_mlir_module(self, mocker):
        """A failed qjit compilation should not cause a secondary NoneType error."""

        from catalyst.decomposition import decomposition_rules as dr

        mocker.patch.object(dr, "collect_resources_for_op", return_value=({}, {}, []))
        failed_qjit = mocker.MagicMock(mlir_module=None)
        mocker.patch.object(dr.qp, "qjit", return_value=lambda _circuit: failed_qjit)

        with pytest.raises(
            CompileError,
            match="Failed to generate an MLIR module while compiling decomposition rules for S",
        ):
            dr.compile_decomposition_rules("S", "S{}{wires:1}{}", {}, {"wires": 1}, {})


class TestModifierIds:
    """Unit tests for the op-level modifier name-wrapping helpers (Adjoint / C canonicalization)."""

    @pytest.mark.parametrize(
        "modifier, expected",
        [
            ("Adjoint", "Adjoint"),
            ("C", "C"),
            ("2C", "C"),  # multi-control normalises to the "C" kind
            ("10C", "C"),
        ],
    )
    def test_modifier_kind(self, modifier, expected):
        """A modifier token normalises to its canonical kind (any ``<n>C`` -> ``C``)."""
        assert _modifier_kind(modifier) == expected

    @pytest.mark.parametrize(
        "op_id, expected",
        [
            ("Adjoint(RX){0:[f64]}{wires:1}{}", "Adjoint"),
            ("C(RX){0:[f64]}{wires:1}{}", "C"),
            ("2C(RX){0:[f64]}{wires:1}{}", "C"),  # exercises the leading-digit scan
            ("10C(RX){0:[f64]}{wires:1}{}", "C"),  # multi-digit control count
            ("RX{0:[f64]}{wires:1}{}", None),  # bare id, no modifier
            ("", None),  # empty edge case
        ],
    )
    def test_leading_modifier_kind(self, op_id, expected):
        """The outermost modifier of an id is detected (including ``<n>C(...)``), or None if bare."""
        assert _leading_modifier_kind(op_id) == expected

    @pytest.mark.parametrize(
        "op_id, modifier, expected",
        [
            # Wrapping a bare id with each modifier kind.
            ("RX{0:[f64]}{wires:1}{}", "C", "C(RX){0:[f64]}{wires:1}{}"),
            ("RX{0:[f64]}{wires:1}{}", "2C", "2C(RX){0:[f64]}{wires:1}{}"),
            ("RX{0:[f64]}{wires:1}{}", "Adjoint", "Adjoint(RX){0:[f64]}{wires:1}{}"),
            # Canonical nesting: control is outermost, so C may wrap an already-adjointed id.
            (
                "Adjoint(RX){0:[f64]}{wires:1}{}",
                "C",
                "C(Adjoint(RX)){0:[f64]}{wires:1}{}",
            ),
            # Only the name is wrapped; the `{...}...[uid]` suffix is carried through untouched.
            ("HybridOp{a:[[f64]]}{w:1}{}[42]", "C", "C(HybridOp){a:[[f64]]}{w:1}{}[42]"),
        ],
    )
    def test_wrap_modifier_id_canonical(self, op_id, modifier, expected):
        """A modifier wraps only the operator name, and control may nest outside adjoint."""
        assert wrap_modifier_id(op_id, modifier) == expected

    @pytest.mark.parametrize(
        "op_id",
        ["C(RX){0:[f64]}{wires:1}{}", "2C(RX){0:[f64]}{wires:1}{}"],
    )
    def test_wrap_modifier_id_rejects_non_canonical(self, op_id):
        """Wrapping the canonically-inner ``Adjoint`` around an already-controlled id is rejected."""
        assert _MODIFIER_CANONICAL_ORDER == ("C", "Adjoint")
        with pytest.raises(ValueError, match="Non-canonical modifier order"):
            wrap_modifier_id(op_id, "Adjoint")


class TestSymbolicRules:
    """Tests for the rules registered against a symbolic operator that take the symbolic
    op's args; following the convention in PennyLane."""

    def test_self_adjoint_rule_is_lowered(self):
        """Test ``self_adjoint`` rule on ``Adjoint(Hadamard)``."""

        module = compile_registered_symbolic_rules(
            "Hadamard",
            "Adjoint(Hadamard){}{wires:1}{}",
            {},
            {"wires": 1},
            {},
            op_cls=qp.Hadamard,
            kind="adjoint",
        )
        (rule,) = get_rule_strings_from_module(module)

        assert 'target_gate = "Adjoint(Hadamard){}{wires:1}{}"' in rule
        assert 'resources = {operations = {"Hadamard{}{wires:1}{}" = 1 : i64}}' in rule
        assert "qref.adjoint" not in rule
        assert "(%arg0: !qref.reg<1>, %arg1: tensor<1xi64>)" in rule
        assert rule.count('gate_name = "Hadamard"') == 1

    def test_adjoint_rotation_rule_is_lowered(self):
        """Test ``adjoint_rotation`` reads the angle off the base operator."""

        module = compile_registered_symbolic_rules(
            "RZ",
            "Adjoint(RZ){0:[f64]}{wires:1}{}",
            {"0": ["f64"]},
            {"wires": 1},
            {},
            is_custom_op=True,
            op_cls=qp.RZ,
            kind="adjoint",
        )
        (rule,) = get_rule_strings_from_module(module)

        assert 'target_gate = "Adjoint(RZ){0:[f64]}{wires:1}{}"' in rule
        assert 'resources = {operations = {"RZ{0:[f64]}{wires:1}{}" = 1 : i64}}' in rule
        assert "qref.adjoint" not in rule
        assert "stablehlo.negate" in rule

    @pytest.mark.parametrize(
        "n_ctrl, target_id, signature, resource",
        [
            (
                1,
                "C(Hadamard){}{wires:1}{}",
                "(%arg0: !qref.reg<2>, %arg1: tensor<1xi64>, %arg2: tensor<1xi64>)",
                '"CH{}{wires:2}{}" = 1 : i64',
            ),
            (
                2,
                "2C(Hadamard){}{wires:1}{}",
                "(%arg0: !qref.reg<3>, %arg1: tensor<1xi64>, %arg2: tensor<2xi64>)",
                '"Toffoli{}{wires:3}{}" = 1 : i64',
            ),
        ],
    )
    def test_controlled_rule_is_lowered(self, n_ctrl, target_id, signature, resource):
        """Test a rule registered on ``C(op)`` is lowered for each control count, with the control
        wires *after* the base wires: the compiler reads a register-mode rule as
        ``func(qreg, param*, inWires*, inCtrlWires*)``."""

        module = compile_registered_symbolic_rules(
            "Hadamard",
            target_id,
            {},
            {"wires": 1},
            {},
            op_cls=qp.Hadamard,
            kind="control",
            n_ctrl=n_ctrl,
        )
        (rule,) = get_rule_strings_from_module(module)

        assert f'target_gate = "{target_id}"' in rule
        assert resource in rule
        assert signature in rule

    def test_controlled_rule_is_wired_up_correctly(self):
        """Test a rule registered on ``C(op)`` acts on the wires it was given by executing it."""

        class CtrlWired(qp.core.Operator2):
            """An operator whose controlled form is a CNOT from the control onto its wire."""

            def __init__(self, wires):
                super().__init__(wires=wires)

        @register_resources({qp.PauliX: 1})
        def x_rule(wires):
            qp.X(wires=wires)

        @register_resources(lambda base, control_wires, **_: {qp.CNOT: 1})
        def ctrl_rule(base, control_wires, **_):
            qp.CNOT(wires=list(control_wires) + list(base.wires))

        with local_decomps():
            add_decomps(CtrlWired, x_rule)
            add_decomps("C(CtrlWired)", ctrl_rule)

            @qjit(capture=True)
            @graph_decomposition(gate_set=["CNOT", "PauliX"])
            @qnode(qp.device("lightning.qubit", wires=2))
            def circuit():
                # The X both prepares the control in |1> -- so a swapped control/target leaves
                # wire 0 unflipped -- and is the operation a dropped control qubit would undo.
                qp.X(1)
                qp.ctrl(CtrlWired(0), control=[1])
                return qp.expval(qp.Z(0))

            result = circuit()

        # The controlled op fires and flips wire 0, so <Z_0> = -1; either defect gives +1.
        assert np.allclose(result, -1.0)

    def test_discovered_op_gets_multi_controlled_rules(self):
        """Test an op reached through the walk is given rules for *every* control count in play,
        not just a single control.
        """

        with local_decomps():

            @register_resources({qp.Hadamard: 1})
            def h_rule(reg):
                qp.Hadamard(wires=reg[0])

            add_decomps(NoParams, h_rule)

            module_str = compile_reachable_decomposition_rules_wrapper(
                "NoParams", "3C(NoParams){}{reg:1}{}", {}, {"reg": 1}, {}
            )

        assert 'target_gate = "3C(NoParams){}{reg:1}{}"' in module_str
        # Hadamard is only reached through NoParams' rule, and needs the same control count.
        assert 'target_gate = "3C(Hadamard){}{wires:1}{}"' in module_str

    def test_symbolic_rule_of_multi_wire_argument_op_is_lowered(self):
        """Test the base operator of a symbolic rule is built with each wire argument on its own
        wires.
        """

        class DisjointRegisters(qp.core.Operator2):
            """An operator that rejects overlapping registers, as QROM does."""

            wire_argnames = ("control", "target")

            def __init__(self, control, target):
                if self._labels(control) & self._labels(target):
                    raise ValueError("control and target wires must not overlap")
                super().__init__(control=control, target=target)

            @staticmethod
            def _labels(wires):
                """The concrete wire labels, skipping abstract ones."""
                labels = set()
                for wire in wires:
                    try:
                        labels.add(int(wire))
                    except TypeError:
                        pass
                return labels

        @register_resources(lambda base, control_wires, **_: {qp.CNOT: 1})
        def ctrl_rule(base, control_wires, **_):
            qp.CNOT(wires=[control_wires[0], base.target[0]])

        with local_decomps():
            add_decomps("C(DisjointRegisters)", ctrl_rule)

            module = compile_registered_symbolic_rules(
                "DisjointRegisters",
                "C(DisjointRegisters){}{control:2,target:1}{}",
                {},
                {"control": 2, "target": 1},
                {},
                op_cls=DisjointRegisters,
                kind="control",
                n_ctrl=1,
            )

        (rule,) = get_rule_strings_from_module(module)
        assert 'target_gate = "C(DisjointRegisters){}{control:2,target:1}{}"' in rule
        assert '"CNOT{}{wires:2}{}" = 1 : i64' in rule

    def test_symbolic_resource_id_is_canonical(self):
        """Test a resource that is itself symbolic is spelled the way the compiler spells a
        modified operator: the base op's id with the modifier folded into its name."""

        base = abstractify(qp.S(wires=jnp.array([0])))
        assert GraphOpID(base).getGraphOpId() == "S{}{wires:1}{}"
        assert GraphOpID(qp.adjoint(base)).getGraphOpId() == "Adjoint(S){}{wires:1}{}"
        assert GraphOpID(qp.ctrl(base, control=[1, 2])).getGraphOpId() == "2C(S){}{wires:1}{}"
        # The modifiers a caller applies compose with the operator's own, control outermost.
        assert (
            GraphOpID(qp.adjoint(base)).getGraphOpId(num_controls=1) == "C(Adjoint(S)){}{wires:1}{}"
        )
        # The base alone is what owns the rules, so its id leaves the modifiers off.
        assert GraphOpID(qp.ctrl(base, control=[1, 2])).getBaseGraphOpId() == "S{}{wires:1}{}"
        # A concrete controlled class is *not* a generic wrapper and keeps its own id.
        assert (
            GraphOpID(abstractify(qp.CH(wires=jnp.array([0, 1])))).getGraphOpId()
            == "CH{}{wires:2}{}"
        )

    def test_no_registered_symbolic_rules(self):
        """Test an op with no symbolic rules registered against its adjoint yields no module."""

        with local_decomps():
            assert (
                compile_registered_symbolic_rules(
                    "NoParams",
                    "Adjoint(NoParams){}{reg:2}{}",
                    {},
                    {"reg": 2},
                    {},
                    op_cls=NoParams,
                    kind="adjoint",
                )
                is None
            )

    def test_missing_op_class_raises(self):
        """Test lowering cannot proceed without the base operator's class: these rules take a base
        operator instance, which the operator's name alone cannot produce."""

        with pytest.raises(ValueError, match="operator class of 'Hadamard' is needed"):
            compile_registered_symbolic_rules(
                "Hadamard", "Adjoint(Hadamard){}{wires:1}{}", {}, {"wires": 1}, {}, kind="adjoint"
            )


class TestApplicabilityFilterOrdering:
    """Regression tests that pin the behaviour that a rule's applicability condition is evaluated
    before computing its resources.

    This ensures we don't get cases where an inapplicable rule is generating resource-failure warnings
    as it should never even be considered in the first place.

    """

    OP_ID = "SingleParam{x:[tensor<f64>]}{reg:1}{}"
    OP_ARGS = ({"x": ["tensor<f64>"]}, {"reg": 1}, {})

    # kind, lookup name, ctrl_wires
    SYMBOLIC_KINDS = [
        ("adjoint", "Adjoint(NoParams)", ()),
        ("control", "C(NoParams)", (2,)),
    ]

    @staticmethod
    def _lowering_warnings(recorded):
        return [str(w.message) for w in recorded if issubclass(w.category, RuleLoweringWarning)]

    @staticmethod
    def _rule(name, applicable=True, resources_explode=False, condition_explodes=False):
        def condition(x, reg):
            if condition_explodes:
                raise RuntimeError("some error")

            return applicable

        def resources(x, reg):
            if resources_explode:
                raise ValueError("another error")

            return {NoParams(reg=Wire[1]): 1}

        def impl(x, reg):
            NoParams(reg=reg)

        impl.__name__ = name
        return register_condition(condition)(register_resources(resources)(impl))

    def _compile(self, *rules):
        with local_decomps():
            add_decomps(SingleParam, *rules)
            return compile_decomposition_rules_wrapper("SingleParam", self.OP_ID, *self.OP_ARGS)

    @staticmethod
    def _symbolic_rule(name, applicable=True, condition_explodes=False, resources_explode=False):
        """A rule registered against ``Adjoint(NoParams)``/``C(NoParams)``.

        PennyLane calls such a rule with the symbolic operator's own arguments: ``base`` alone for an
        adjoint, and ``base`` plus the control entries for a controlled operator. The catch-all
        absorbs the latter, so one helper serves both kinds.
        """

        def condition(base, **_):
            if condition_explodes:
                raise RuntimeError("some error")

            return applicable

        def resources(base, **_):
            if resources_explode:
                raise ValueError("another error")

            return {NoParams(reg=Wire[2]): 1}

        def impl(base, **_):
            NoParams(reg=[0, 1])

        impl.__name__ = name
        return register_condition(condition)(register_resources(resources)(impl))

    @staticmethod
    def _collect_symbolic(rule, lookup_name, kind, ctrl_wires):
        with local_decomps():
            add_decomps(lookup_name, rule)
            return collect_symbolic_resources(
                NoParams,
                "NoParams",
                prepare_dynamic_op_kwargs({}, {"reg": 2}),
                False,
                kind=kind,
                ctrl_wires=ctrl_wires,
            )

    # ---- Actual Tests -----

    def test_inapplicable_rule_emits_no_warning(self, recwarn):
        """Test that an inapplicable rule would be skipped. The condition check is first so we
        should never see the resource failure happen."""

        result = self._compile(
            self._rule("inapplicable_rule", applicable=False, resources_explode=True)
        )

        assert self._lowering_warnings(recwarn) == []
        assert "inapplicable_rule" not in result

    def test_inapplicable_rule_never_computes_resources(self):
        """Test that the resource function for an inapplicable rule is never called."""

        calls = []

        def resources(x, reg):
            calls.append("called resources")
            return {NoParams(reg=Wire[1]): 1}

        def impl(x, reg):
            NoParams(reg=reg)

        impl.__name__ = "counted_rule"
        # Make it not applicable
        rule = register_condition(lambda x, reg: False)(register_resources(resources)(impl))

        self._compile(rule)
        assert calls == []

    def test_inapplicable_rule_doesnt_hide_applicable_one(self, recwarn):
        """Test that skipping an inapplicable rule leaves the applicable rules alone."""
        result = self._compile(
            self._rule("inapplicable_rule", applicable=False),
            self._rule("applicable_rule", applicable=True),
        )

        assert self._lowering_warnings(recwarn) == []
        assert "applicable_rule" in result
        assert "inapplicable_rule" not in result
        assert f'target_gate = "{self.OP_ID}"' in result

    def test_raising_condition_drops_only_its_own_rule(self, recwarn):
        """Tests that a condition that raises on the probe arguments is reported and its rule skipped."""

        result = self._compile(
            self._rule("exploding_rule", condition_explodes=True),
            self._rule("applicable_rule"),
        )

        assert self._lowering_warnings(recwarn) == [
            "Excluded the exploding_rule decomposition rule for SingleParam; raised 'some error'"
        ]
        assert "exploding_rule" not in result
        assert "applicable_rule" in result

    @pytest.mark.parametrize("kind, lookup_name, ctrl_wires", SYMBOLIC_KINDS)
    def test_symbolic_skips_inapplicable_rule(self, kind, lookup_name, ctrl_wires, recwarn):
        """Test that an inapplicable rule registered against a symbolic operator is skipped without
        its resource function being called, for both symbolic kinds."""

        rule = self._symbolic_rule(
            "inapplicable_sym_rule", applicable=False, resources_explode=True
        )
        rules, _, name_to_resources, name_to_resource_ids = self._collect_symbolic(
            rule, lookup_name, kind, ctrl_wires
        )

        assert self._lowering_warnings(recwarn) == []
        assert [r.name for r in rules] == []
        assert name_to_resources == {}
        assert name_to_resource_ids == {}

    @pytest.mark.parametrize("kind, lookup_name, ctrl_wires", SYMBOLIC_KINDS)
    def test_symbolic_raising_condition_is_reported(self, kind, lookup_name, ctrl_wires, recwarn):
        """Test that a condition that raises on the probe arguments is reported and its rule
        skipped, for both symbolic kinds."""

        rule = self._symbolic_rule("exploding_sym_rule", condition_explodes=True)
        rules, _, name_to_resources, _ = self._collect_symbolic(rule, lookup_name, kind, ctrl_wires)

        # NOTE: the message names the base operator, which is what `collect_symbolic_resources`
        # hands down; the rule itself is registered against `lookup_name`.
        assert self._lowering_warnings(recwarn) == [
            "Excluded the exploding_sym_rule decomposition rule for NoParams; raised 'some error'"
        ]
        assert [r.name for r in rules] == []
        assert name_to_resources == {}

    def test_control_rule_needing_work_wires_is_skipped(self, recwarn):
        """Test that a ``C(Op)`` rule requiring a work wire is filtered out on this path.

        Note this test is intentionally mimicking _select_decomp_multi_control_work_wire's behaviour.
        """

        def condition(base, work_wires, **_):
            return len(work_wires) >= 1

        def resources(base, work_wires, **_):
            # One work wire is consumed, the rest are passed on: only meaningful once the condition
            # above holds, otherwise this is a register of unknown length.
            passed_on = Wire[len(work_wires) - 1]
            return {NoParams(reg=Wire[2]): 1 + len(passed_on)}

        def impl(base, control_wires, control_values, work_wires, work_wire_type):
            NoParams(reg=[0, 1])

        impl.__name__ = "ctrl_work_wire_rule"
        rule = register_condition(condition)(register_resources(resources)(impl))

        rules, probe_args, name_to_resources, _ = self._collect_symbolic(
            rule, "C(NoParams)", "control", (2,)
        )

        assert probe_args["work_wires"] == Wires([])
        assert self._lowering_warnings(recwarn) == []
        assert [r.name for r in rules] == []
        assert name_to_resources == {}


if __name__ == "__main__":
    pytest.main(["-x", __file__])
