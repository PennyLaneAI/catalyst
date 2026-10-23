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
"""Regression tests for the operand order of ``qref.operator`` and its decomposition rules.

Operand order (declaration order, see ``catalyst.from_plxpr.operand_layout``) is the only positional
order. Argument names are identity only and are sorted only when printed into a GraphOpID. These
tests use argument names whose declaration order differs from their sorted order.
"""

import re

import numpy as np
import pennylane as qp
import pytest
from pennylane.decomposition import add_decomps, register_resources

from catalyst import qjit
from catalyst.decomposition import GraphOpID
from catalyst.decomposition.decomposition_rules import ordered_kwarg_names
from catalyst.passes import graph_decomposition

pytestmark = pytest.mark.usefixtures("use_capture")


class ZetaAlphaOp(qp.core.Operator2):
    """Dynamic args declared in non-alphabetical order."""

    dynamic_argnames = ("zeta", "alpha")
    wire_argnames = ("reg",)

    def __init__(self, zeta, alpha, reg):
        super().__init__(zeta, alpha, reg=reg)


@register_resources(lambda **_: {qp.RX: 1, qp.RY: 1})
def _zeta_alpha_rule(zeta, alpha, reg):
    qp.RX(zeta, reg[0])
    qp.RY(alpha, reg[1])


add_decomps(ZetaAlphaOp, _zeta_alpha_rule)


class XAOp(qp.core.Operator2):
    """Wire args declared in non-alphabetical order, with a static arg (so the GraphOpID has a UID)."""

    wire_argnames = ("x_wires", "a_wires")
    static_argnames = ("label",)

    def __init__(self, x_wires, a_wires, label="hi"):
        super().__init__(x_wires=x_wires, a_wires=a_wires, label=label)


@register_resources(lambda **_: {qp.X: 1})
def _xa_rule(x_wires, a_wires, label):  # pylint: disable=unused-argument
    qp.X(x_wires[0])


add_decomps(XAOp, _xa_rule)


class TestGraphOpIDLayout:
    """GraphOpID exposes the operand order, and prints a sorted identity."""

    def test_dynamic_shape_in_operand_order(self):
        """``dynamic_shape`` follows declaration order, while the id string is sorted."""
        gid = GraphOpID(ZetaAlphaOp(np.float64(0.1), np.float64(0.2), reg=[0, 1]))
        assert list(gid.dynamic_shape) == ["zeta", "alpha"]
        assert gid.getGraphOpId().startswith("ZetaAlphaOp{alpha:")

    def test_wire_lens_in_operand_order(self):
        """``wire_lens`` follows declaration order, while the id string is sorted."""
        gid = GraphOpID(XAOp(x_wires=[0], a_wires=[1, 2]))
        assert list(gid.wire_lens.items()) == [("x_wires", 1), ("a_wires", 2)]
        assert "{a_wires:2,x_wires:1}" in gid.getGraphOpId()

    def test_ordered_kwarg_names_keeps_dynamic_shape_order(self):
        """Rule params are laid out in ``dynamic_shape`` (operand) order, not sorted."""
        call_kwargs = {"reg": None, "zeta": None, "alpha": None}
        dynamic_shape = {"zeta": None, "alpha": None}
        assert ordered_kwarg_names(call_kwargs, dynamic_shape) == ["zeta", "alpha", "reg"]


class TestExecution:
    """Rules receive each argument's own operands."""

    def test_semiadder(self):
        """SemiAdder's wire args are declared x, y, work (issue #3337)."""
        gate_set = {"Toffoli", "CNOT", "X", "Hadamard", "S", "Adjoint(S)", "T", "Adjoint(T)", "CZ"}
        gate_set |= {"MidMeasure", "MidMeasureMP", "MidCircuitMeasure", "Cond"}

        @qjit(capture=True)
        @graph_decomposition(gate_set=gate_set)
        @qp.qnode(qp.device("lightning.qubit", wires=8))
        def circuit():
            qp.X(5)
            qp.X(1)
            qp.SemiAdder(x_wires=[3, 4, 5], y_wires=[0, 1, 2], work_wires=[6, 7])
            return qp.probs(wires=range(8))

        assert format(int(np.argmax(circuit())), "08b") == "01100100"

    def test_unsorted_dynamic_args(self):
        """Each dynamic arg reaches the rule under its own name."""

        @qjit(capture=True)
        @qp.decompose(gate_set={"RX", "RY"})
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            ZetaAlphaOp(0.3, 1.9, reg=[0, 1])
            return qp.probs(wires=[0, 1])

        expected = np.kron(
            [np.cos(0.15) ** 2, np.sin(0.15) ** 2], [np.cos(0.95) ** 2, np.sin(0.95) ** 2]
        )
        assert np.allclose(circuit(), expected)

    @pytest.mark.parametrize("nx, na", [(1, 2), (2, 2)])
    def test_unsorted_wire_args_with_uid(self, nx, na):
        """The rule's UID matches the lowered op's, and each wire arg gets its own qubits."""

        @qjit(capture=True)
        @qp.decompose(gate_set={"X"})
        @qp.qnode(qp.device("lightning.qubit", wires=4))
        def circuit():
            XAOp(x_wires=list(range(nx)), a_wires=list(range(nx, nx + na)))
            return qp.probs(wires=[0])

        assert np.allclose(circuit(), [0.0, 1.0])

        op_uid = re.search(r'qref\.operator "XAOp".*?UID\((\d+)\)', circuit.mlir, re.DOTALL)
        rule_uid = re.search(r'target_gate = "XAOp[^"]*\[(\d+)\]"', circuit.mlir)
        if op_uid and rule_uid:  # the op may already be decomposed in the final IR
            assert op_uid.group(1) == rule_uid.group(1)
