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

"""Test contextual decomposition-rule capture for plain and adjoint Operator2 gates."""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable = line-too-long,unused-argument,missing-function-docstring

import pennylane as qp
from operator2_dummy_gates import NoParams, SingleParam
from pennylane.typing import Float, Wire


def _base_rule():
    def base_resource_fn(reg):
        return {SingleParam(x=Float, reg=Wire[2]): 1}

    @qp.register_resources(base_resource_fn)
    def base_rule(reg):
        SingleParam(x=0.1, reg=reg[0:2])

    return base_rule


def _adj_rule():
    """A rule for ``Adjoint(NoParams)``."""

    def adj_resource_fn(base):
        return {SingleParam(x=Float, reg=Wire[2]): 2}

    @qp.register_resources(adj_resource_fn)
    def adj_rule(base):
        SingleParam(x=0.2, reg=base.wires[0:2])
        SingleParam(x=0.3, reg=base.wires[0:2])

    return adj_rule


def test_plain_gate_captures_only_base():
    """Lowering a plain gate captures only the rules registered against the plain gate."""
    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, _base_rule())
        qp.add_decomps("Adjoint(NoParams)", _adj_rule())

        @qp.qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=3))
        def c():
            NoParams(reg=[0, 1])
            return qp.state()

        print(c.mlir)

    # CHECK: qref.operator "NoParams"()
    # CHECK-DAG: func.func private @"__builtin_base_rule_NoParams{}{reg:2}{}"{{.*}}"SingleParam{{.*}}target_gate = "NoParams{}{reg:2}{}"


test_plain_gate_captures_only_base()


def test_adjoint_gate_captures_adjoint_rule():
    """Lowering the adjoint of a gate carries the modifier and still captures the Adjoint(Op) rule."""
    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, _base_rule())
        qp.add_decomps("Adjoint(NoParams)", _adj_rule())

        @qp.qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=3))
        def c():
            qp.adjoint(NoParams(reg=[0, 1]))
            return qp.state()

        print(c.mlir)

    # Both alternatives for the requested adjoint target are retained: the directly registered
    # symbolic rule and the rule synthesized by distributing adjoint over the base rule.
    # CHECK: qref.operator "NoParams"() adj
    # CHECK-DAG: func.func private @"__builtin_adj_rule_Adjoint(NoParams){}{reg:2}{}"{{.*}}"SingleParam{{.*}} = 2 : i64{{.*}}target_gate = "Adjoint(NoParams){}{reg:2}{}"
    # CHECK-DAG: func.func private @"__builtin_base_rule_Adjoint(NoParams){}{reg:2}{}"{{.*}}"Adjoint(SingleParam{{.*}}target_gate = "Adjoint(NoParams){}{reg:2}{}"


test_adjoint_gate_captures_adjoint_rule()
