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

"""
Test the lowering of decomposition rules registered against ``Adjoint(Op)`` and written against the
*symbolic* operator's arguments following PennyLane's conventions.
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable = line-too-long,unused-argument,missing-function-docstring

import pennylane as qp
from operator2_dummy_gates import NoParams, SingleParam
from pennylane.typing import Float, Wire

from catalyst.decomposition.decomposition_rules import (
    materialize_reachable_rule_strings,
)


def _base_rule():
    def base_resource_fn(reg):
        return {SingleParam(x=Float, reg=Wire[2]): 1}

    @qp.register_resources(base_resource_fn)
    def base_rule(reg):
        SingleParam(x=0.1, reg=reg[0:2])

    return base_rule


def _self_adjoint_rule():
    """Test the lowering into a rule for ``Adjoint(NoParams)`` that applies the
    bare base op without regions."""

    @qp.register_resources(lambda base: {qp.core.abstractify(base): 1})
    def self_adjoint(base):
        qp.apply(base)

    return self_adjoint


def test_registered_self_adjoint_rule_targets_the_adjoint_op():
    """Test the lowering into a rule for ``Adjoint(NoParams)`` that applies the
    bare base op without regions."""
    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, _base_rule())
        qp.add_decomps("Adjoint(NoParams)", _self_adjoint_rule())

        print(
            "\n".join(
                materialize_reachable_rule_strings(
                    op_name="NoParams",
                    op_id="NoParams{}{reg:2}{}",
                    dynamic_shape={},
                    wire_lens={"reg": 2},
                    static_data={},
                    op_cls=NoParams,
                )
            )
        )

    # CHECK-DAG: func.func private @"__builtin_self_adjoint_Adjoint(NoParams){}{reg:2}{}"(%arg0: !qref.reg<2>, %arg1: tensor<2xi64>){{.*}}"NoParams{}{reg:2}{}" = 1 : i64{{.*}}target_gate = "Adjoint(NoParams){}{reg:2}{}"
    # CHECK-DAG: func.func private @"__builtin_base_rule_NoParams{}{reg:2}{}"{{.*}}target_gate = "NoParams{}{reg:2}{}"


test_registered_self_adjoint_rule_targets_the_adjoint_op()


def test_registered_symbolic_rule_accepts_mcm():
    """Test the lowering into a rule for ``Adjoint(NoParams)`` that contains MCMs."""

    @qp.register_resources({NoParams(Wire[1]): 1, qp.ops.MidMeasure(Wire[1]): 1})
    def rule_with_mcm(base):
        m0 = qp.measure(base.wires[0])
        qp.cond(m0, NoParams)(base.wires[0])

    with qp.decomposition.local_decomps():

        qp.add_decomps("Adjoint(NoParams)", rule_with_mcm)

        print(
            "\n".join(
                materialize_reachable_rule_strings(
                    op_name="NoParams",
                    op_id="NoParams{}{reg:1}{}",
                    dynamic_shape={},
                    wire_lens={"reg": 1},
                    static_data={},
                    op_cls=NoParams,
                )
            )
        )

    # CHECK-LABEL: func.func private @"__builtin_rule_with_mcm_Adjoint(NoParams){}{reg:1}{}"
    # CHECK-SAME: target_gate = "Adjoint(NoParams){}{reg:1}{}"


test_registered_symbolic_rule_accepts_mcm()


def _controlled_rule():
    """A rule registered against ``C(NoParams)`` and written against the controlled operator's own
    arguments: the base operator plus its control wires, values and work wires."""

    @qp.register_resources(lambda base, control_wires, **_: {qp.CNOT: 1})
    def controlled(base, control_wires, **_):
        qp.CNOT(wires=[control_wires[0], base.reg[0]])

    return controlled


def test_registered_control_rule_targets_the_controlled_op():
    """Test a rule registered against ``C(NoParams)`` is lowered for each control count in play
    with the control wires trailing the base wires in the rule's operands."""
    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, _base_rule())
        qp.add_decomps("C(NoParams)", _controlled_rule())

        print(
            "\n".join(
                materialize_reachable_rule_strings(
                    op_name="NoParams",
                    op_id="NoParams{}{reg:2}{}",
                    dynamic_shape={},
                    wire_lens={"reg": 2},
                    static_data={},
                    op_cls=NoParams,
                    n_ctrls=2,
                )
            )
        )

    # CHECK-DAG: func.func private @"__builtin_controlled_C(NoParams){}{reg:2}{}"(%arg0: !qref.reg<3>, %arg1: tensor<2xi64>, %arg2: tensor<1xi64>){{.*}}"CNOT{}{wires:2}{}" = 1 : i64{{.*}}target_gate = "C(NoParams){}{reg:2}{}"
    # CHECK-DAG: func.func private @"__builtin_controlled_2C(NoParams){}{reg:2}{}"(%arg0: !qref.reg<4>, %arg1: tensor<2xi64>, %arg2: tensor<2xi64>){{.*}}target_gate = "2C(NoParams){}{reg:2}{}"
    # CHECK-DAG: func.func private @"__builtin_base_rule_2C(NoParams){}{reg:2}{}"{{.*}}"2C(SingleParam){x:[tensor<f64>]}{reg:2}{}" = 1 : i64{{.*}}target_gate = "2C(NoParams){}{reg:2}{}"


test_registered_control_rule_targets_the_controlled_op()
