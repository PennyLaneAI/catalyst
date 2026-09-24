# Copyright 2022-2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for UID generation for Operator2 lowering."""

import pennylane as qp
import pytest
from pennylane.pytrees import flatten
from pennylane.typing import AbstractArray
from pennylane.wires import AbstractQubit

from catalyst.from_plxpr.uid import _serialize, generate_uid


class StaticOp(qp.core.Operator2):
    """Test operator with static Python-only data."""

    static_argnames = ("label",)

    # pylint: disable=useless-parent-delegation
    def __init__(self, label, wires):
        super().__init__(label, wires)


class HybridWiresOp(qp.core.Operator2):
    """Test operator with a hybrid wire argument."""

    hybrid_argnames = ("cwires",)
    wire_argnames = ("cwires",)

    # pylint: disable=useless-parent-delegation
    def __init__(self, cwires):
        super().__init__(cwires=cwires)


class DynamicStaticOp(qp.core.Operator2):
    """Test operator with dynamic and static Python-only data."""

    dynamic_argnames = ("angle",)
    static_argnames = ("label",)

    # pylint: disable=useless-parent-delegation
    def __init__(self, angle, label, wires):
        super().__init__(angle, label, wires)


class InnerOp(qp.core.Operator2):
    """Simple test operator to use as an input for hybrid operator testing."""

    dynamic_argnames = ("phi",)

    # pylint: disable=useless-parent-delegation
    def __init__(self, phi, wires):
        super().__init__(phi, wires)


class HybridOp(qp.core.Operator2):
    """Test operator that takes an operator as input."""

    hybrid_argnames = ("op",)
    static_argnames = ("label",)
    wire_argnames = ()

    # pylint: disable=useless-parent-delegation
    def __init__(self, op, label=""):
        super().__init__(op, label=label)


class _Opaque:
    """Opaque type for testing."""


def _static_kwargs(label):
    leaves, tree = flatten(label)
    return {"label": (tuple(leaves), tree)}


class TestGenerateUID:
    """Tests for generate_uid and helpers."""

    def test_same_static_args_same_uid(self):
        """Test that operators with the same static arguments have the same UID."""
        uid_a = generate_uid(
            op_cls=StaticOp,
            hybrid_lens=(),
            hybrid_trees=(),
            static_args=_static_kwargs("hello"),
        )
        uid_b = generate_uid(
            op_cls=StaticOp,
            hybrid_lens=(),
            hybrid_trees=(),
            static_args=_static_kwargs("hello"),
        )
        assert uid_a == uid_b

    def test_different_static_args_different_uid(self):
        """Test that operators with different static arguments have different UIDs."""
        uid_a = generate_uid(
            op_cls=StaticOp,
            hybrid_lens=(),
            hybrid_trees=(),
            static_args=_static_kwargs("hello"),
        )
        uid_b = generate_uid(
            op_cls=StaticOp,
            hybrid_lens=(),
            hybrid_trees=(),
            static_args=_static_kwargs("world"),
        )
        assert uid_a != uid_b

    def test_same_hybrid_wire_count_same_uid(self):
        """Test that operators with the same number of hybrid wires with the same PyTree
        shape have the same UID."""
        _, hybrid_tree = flatten(qp.wires.Wires([0, 1]))
        hybrid_trees = (hybrid_tree,)

        uid_a = generate_uid(
            op_cls=HybridWiresOp,
            hybrid_lens=(2,),
            hybrid_trees=hybrid_trees,
            static_args={},
        )
        uid_b = generate_uid(
            op_cls=HybridWiresOp,
            hybrid_lens=(2,),
            hybrid_trees=hybrid_trees,
            static_args={},
        )
        assert uid_a == uid_b

    def test_different_hybrid_wires_different_uid(self):
        """Test that operators with the same number of hybrid wires but with
        different PyTree structures have different UIDs."""
        _, hybrid_tree1 = flatten(qp.wires.Wires([0, 1]))
        _, hybrid_tree2 = flatten([qp.wires.Wires([0, 1])])

        uid_two = generate_uid(
            op_cls=HybridWiresOp,
            hybrid_lens=(2,),
            hybrid_trees=(hybrid_tree1,),
            static_args={},
        )
        uid_three = generate_uid(
            op_cls=HybridWiresOp,
            hybrid_lens=(2,),
            hybrid_trees=(hybrid_tree2,),
            static_args={},
        )
        assert uid_two != uid_three

    def test_dynamic_avals_do_not_affect_uid(self):
        """Test that dynamic argument avals do not contribute to the UID."""
        kwargs = {
            "op_cls": DynamicStaticOp,
            "hybrid_lens": (),
            "hybrid_trees": (),
            "static_args": _static_kwargs("hello"),
        }

        uid_scalar = generate_uid(AbstractArray((), int), **kwargs)
        uid_matrix = generate_uid(AbstractArray((4, 4), int), **kwargs)
        assert uid_scalar == uid_matrix

    @pytest.mark.parametrize(
        "value",
        [
            None,
            True,
            1,
            1.5,
            1 + 2j,
            "s",
            [1, True],
            (1, "a"),
            {"k": 1},
            {1, 2},
            frozenset([1]),
            _Opaque(),
        ],
    )
    def test_supported_types(self, value):
        """Test that common static Python types are serialized for UID hashing."""
        ser = _serialize(value)
        assert isinstance(hash(ser), int)

    def test_hybrid_operator_avals_in_uid(self):
        """Test that non-wire hybrid operator aval signatures affect UID generation."""
        _, hybrid_tree1 = flatten(InnerOp(0.5, [0, 1, 2]))
        _, hybrid_tree2 = flatten(InnerOp(0.5, [0, 1]))
        kwargs = {
            "op_cls": HybridOp,
            "static_args": _static_kwargs("hello"),
        }

        uid_three_wires = generate_uid(
            AbstractArray((), float),
            AbstractQubit(),
            AbstractQubit(),
            hybrid_trees=(hybrid_tree1,),
            hybrid_lens=(3,),
            **kwargs,
        )
        uid_two_wires = generate_uid(
            AbstractArray((), float),
            AbstractQubit(),
            hybrid_trees=(hybrid_tree2,),
            hybrid_lens=(2,),
            **kwargs,
        )
        assert uid_three_wires != uid_two_wires
