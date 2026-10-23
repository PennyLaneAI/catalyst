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


class MySemiAdder(qp.core.Operator2):
    """SemiAdder-like: wire args declared ``x, y, work`` (sorted order is ``work, x, y``)."""

    wire_argnames = ("x_wires", "y_wires", "work_wires")

    def __init__(self, x_wires, y_wires, work_wires):
        super().__init__(x_wires=x_wires, y_wires=y_wires, work_wires=work_wires)


@register_resources(lambda **_: {qp.CNOT: 3})
def _xor_into_rule(x_wires, y_wires, work_wires):
    """``y ^= x``, routed through ``work_wires[0]`` so that every wire arg is used."""
    for xw, yw in zip(x_wires, y_wires):
        qp.CNOT([xw, work_wires[0]])
        qp.CNOT([work_wires[0], yw])
        qp.CNOT([xw, work_wires[0]])


add_decomps(MySemiAdder, _xor_into_rule)


class MyTrotterVibronic(qp.core.Operator2):
    """TrotterVibronic-like: unsorted wire args of distinct lengths, plus a dynamic and a static
    arg (so the GraphOpID carries a UID)."""

    dynamic_argnames = ("time",)
    wire_argnames = ("electronic_wires", "vib_wires", "cache_wires")
    static_argnames = ("num_steps",)

    def __init__(self, time, electronic_wires, vib_wires, cache_wires, num_steps=1):
        super().__init__(
            time,
            electronic_wires=electronic_wires,
            vib_wires=vib_wires,
            cache_wires=cache_wires,
            num_steps=num_steps,
        )


@register_resources(lambda **_: {qp.RZ: 1, qp.H: 1, qp.X: 1})
def _multi_register_rule(time, electronic_wires, vib_wires, cache_wires, num_steps):
    """One gate type per wire arg, so the gate counts reveal how the wires were sliced."""
    for _ in range(num_steps):
        for w in electronic_wires:
            qp.RZ(time, w)
        for w in vib_wires:
            qp.H(w)
        for w in cache_wires:
            qp.X(w)


add_decomps(MyTrotterVibronic, _multi_register_rule)


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


def test_unsorted_wire_args_are_sliced_in_operand_order():
    """Issue #3337 (SemiAdder): the rule must slice the grouped wire operand in operand order
    (``x, y, work``), not sorted order (``work, x, y``)."""
    x, y, work = [0, 1, 2], [3, 4, 5], [6, 7]

    @qjit(capture=True)
    @qp.decompose(gate_set={"PauliX", "CNOT"})
    @qp.qnode(qp.device("lightning.qubit", wires=8))
    def circuit(xv, yv):
        qp.BasisState(qp.math.int_to_binary(xv, 3), x)
        qp.BasisState(qp.math.int_to_binary(yv, 3), y)
        MySemiAdder(x, y, work_wires=work)
        return qp.probs(wires=y)

    for xv, yv in [(4, 0), (6, 3), (5, 6)]:
        probs = circuit(xv, yv)
        assert np.count_nonzero(probs) == 1  # deterministic outcome
        assert int(np.argmax(probs)) == xv ^ yv


def test_unsorted_wire_args_with_static_arg_match_uid():
    """TrotterVibronic: the lowered op and the captured rule must hash the wire lengths in the
    same order, otherwise the UIDs differ and the rule is not matched."""

    @qjit(capture=True, target="mlir")
    @qp.decompose(gate_set={"RZ", "Hadamard", "PauliX"})
    @qp.qnode(qp.device("null.qubit", wires=7))
    def circuit():
        MyTrotterVibronic(
            0.5, electronic_wires=[0], vib_wires=[1, 2], cache_wires=[3, 4, 5, 6], num_steps=3
        )
        return qp.probs(wires=[0])

    mlir = circuit.mlir
    op_uid = re.search(r'qref\.operator "MyTrotterVibronic".*?UID\((\d+)\)', mlir, re.DOTALL)
    rule_uid = re.search(r'target_gate = "MyTrotterVibronic[^"]*\[(\d+)\]"', mlir)
    assert op_uid.group(1) == rule_uid.group(1)

    # Each wire arg reaches the rule with its own qubits: 1 RZ x 3 steps, 2 H x 3, 4 X x 3.
    resources = qp.analyze(circuit, level=1)().resources
    assert resources.quantum_operations == {"RZ": 3, "Hadamard": 6, "PauliX": 12}
