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
Test that when lowering a PennyLane Operator2 gate to MLIR, all relevant decomposition rules to that
gate are also lowered to MLIR.
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable = line-too-long,unused-argument

import pennylane as qp
from jax import numpy as jnp
from operator2_dummy_gates import (
    CompilableData,
    MultiParams,
    MultipleFullArgs,
    NoParams,
    SingleParam,
    SingleParamCustomOp,
    TestQubitUnitary,
)
from pennylane.typing import Complex, Float, Int, Wire


def test_one_rule():
    """
    Simple tests for when there is only one rule, i.e. graph looks like
        A ---> B
    """

    def rule_resource_fn(reg):
        return {
            SingleParam(x=Float, reg=Wire[2]): 2,
            SingleParam(x=Float[2], reg=Wire[2]): 1,
        }

    @qp.register_resources(rule_resource_fn)
    def rule(reg):
        SingleParam(x=0.1, reg=reg[0:2])
        SingleParam(x=0.2, reg=reg[0:2])
        SingleParam(x=jnp.array([0.3, 0.4]), reg=reg[0:2])

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule)

        def test_one_gate():
            """
            Test when circuit has just one gate.
            """

            @qp.qjit(capture=True, target="mlir")
            @qp.qnode(qp.device("null.qubit", wires=3))
            def one_gate():
                NoParams(reg=[0, 1])
                return qp.state()

            print(one_gate.mlir)

        # CHECK-LABEL: func.func public @one_gate()
        # CHECK: qref.operator "NoParams"
        # CHECK: func.func private @"__builtin_rule_NoParams{}{reg:2}{}"
        # CHECK-SAME:   resources = {operations = {
        # CHECK-SAME:   "SingleParam{x:[tensor<2xf64>]}{reg:2}{}" = 1 : i64,
        # CHECK-SAME:   "SingleParam{x:[tensor<f64>]}{reg:2}{}" = 2 : i64
        # CHECK-SAME:   target_gate = "NoParams{}{reg:2}{}"
        test_one_gate()

        def test_multiple_gates_same_id():
            """
            Test that when circuit has multiple gates of the same id, the rule is only injected
            once.
            """

            @qp.qjit(capture=True, target="mlir")
            @qp.qnode(qp.device("null.qubit", wires=3))
            def same_id():
                NoParams(reg=[0, 1])
                NoParams(reg=[0, 1])
                return qp.state()

            print(same_id.mlir)

        # CHECK-LABEL: func.func public @same_id()
        # CHECK: qref.operator "NoParams"
        # CHECK: qref.operator "NoParams"
        # CHECK: func.func private @"__builtin_rule_NoParams{}{reg:2}{}"
        # CHECK-SAME:   resources = {operations = {
        # CHECK-SAME:   "SingleParam{x:[tensor<2xf64>]}{reg:2}{}" = 1 : i64,
        # CHECK-SAME:   "SingleParam{x:[tensor<f64>]}{reg:2}{}" = 2 : i64
        # CHECK-SAME:   target_gate = "NoParams{}{reg:2}{}"
        # CHECK-NOT: func.func private @"__builtin_rule_NoParams{}{reg:2}{}"
        test_multiple_gates_same_id()

        def test_multiple_gates_different_ids():
            """
            Test when circuit has multiple gates of the same Operator 2 class but different ids,
            multiple rules are generated.
            """

            @qp.qjit(capture=True, target="mlir")
            @qp.qnode(qp.device("null.qubit", wires=3))
            def different_id():
                NoParams(reg=[0, 1])
                NoParams(reg=[0, 1, 2])
                return qp.state()

            print(different_id.mlir)

        # CHECK-LABEL: func.func public @different_id()
        # CHECK: qref.operator "NoParams"
        # CHECK: qref.operator "NoParams"
        # CHECK: func.func private @"__builtin_rule_NoParams{}{reg:2}{}"
        # CHECK-SAME:   resources = {operations = {
        # CHECK-SAME:   "SingleParam{x:[tensor<2xf64>]}{reg:2}{}" = 1 : i64,
        # CHECK-SAME:   "SingleParam{x:[tensor<f64>]}{reg:2}{}" = 2 : i64
        # CHECK-SAME:   target_gate = "NoParams{}{reg:2}{}"
        # CHECK: func.func private @"__builtin_rule_NoParams{}{reg:3}{}"
        # CHECK-SAME:   resources = {operations = {
        # CHECK-SAME:   "SingleParam{x:[tensor<2xf64>]}{reg:2}{}" = 1 : i64,
        # CHECK-SAME:   "SingleParam{x:[tensor<f64>]}{reg:2}{}" = 2 : i64
        # CHECK-SAME:   target_gate = "NoParams{}{reg:3}{}"
        test_multiple_gates_different_ids()


test_one_rule()


def test_multiple_rules_same_gate():
    """
    Tests for when there are multiple distinct rules on the same gate, i.e. graph looks like
                +---> B
                |
        A ---+
                |
                +---> C
    """

    @qp.register_resources(lambda reg: {SingleParam(x=Float, reg=Wire[2]): 1})
    def rule1(reg):
        SingleParam(x=0.1, reg=reg[0:2])

    @qp.register_resources(lambda reg: {SingleParam(x=Float[2], reg=Wire[1]): 1})
    def rule2(reg):
        SingleParam(x=jnp.array([0.3, 0.4]), reg=reg[0])

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule1)
        qp.add_decomps(NoParams, rule2)

        @qp.qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=3))
        def multi_rule():
            NoParams(reg=[0, 1])
            NoParams(reg=[0, 1])
            NoParams(reg=[0, 1, 2])
            return qp.state()

        print(multi_rule.mlir)


# CHECK-LABEL: func.func public @multi_rule()
# CHECK: qref.operator "NoParams"
# CHECK: qref.operator "NoParams"
# CHECK: qref.operator "NoParams"
# CHECK: func.func private @"__builtin_rule1_NoParams{}{reg:2}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "SingleParam{x:[tensor<f64>]}{reg:2}{}" = 1 : i64
# CHECK-SAME:   target_gate = "NoParams{}{reg:2}{}"
# CHECK: func.func private @"__builtin_rule2_NoParams{}{reg:2}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "SingleParam{x:[tensor<2xf64>]}{reg:1}{}" = 1 : i64
# CHECK-SAME:   target_gate = "NoParams{}{reg:2}{}"
#
# CHECK-NOT: func.func private @"__builtin_rule1_NoParams{}{reg:2}{}"
#
# CHECK: func.func private @"__builtin_rule1_NoParams{}{reg:3}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "SingleParam{x:[tensor<f64>]}{reg:2}{}" = 1 : i64
# CHECK-SAME:   target_gate = "NoParams{}{reg:3}{}"
# CHECK: func.func private @"__builtin_rule2_NoParams{}{reg:3}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "SingleParam{x:[tensor<2xf64>]}{reg:1}{}" = 1 : i64
# CHECK-SAME:   target_gate = "NoParams{}{reg:3}{}"
test_multiple_rules_same_gate()


def test_multiple_rules_chained():
    """
    Tests for when one rule involves another rule, i.e. graph looks like
        A ---> B ---> C
    """

    @qp.register_resources(lambda reg: {SingleParam(x=Float, reg=Wire[1]): 1})
    def rule1(reg):
        SingleParam(x=0.1, reg=reg[0])

    @qp.register_resources(
        lambda x, reg: {CompilableData(a="a", b="b", thing="thing", wires=Wire[1]): 1}
    )
    def rule2(x, reg):
        CompilableData(a="a", b="b", thing="thing", wires=reg[0])

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule1)
        qp.add_decomps(SingleParam, rule2)

        @qp.qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=3))
        def chained_rule():
            NoParams(reg=[0, 1])
            return qp.state()

        print(chained_rule.mlir)


# CHECK-LABEL: func.func public @chained_rule()
# CHECK: qref.operator "NoParams"
# CHECK: func.func private @"__builtin_rule1_NoParams{}{reg:2}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "SingleParam{x:[tensor<f64>]}{reg:1}{}" = 1 : i64
# CHECK-SAME:   target_gate = "NoParams{}{reg:2}{}"
# CHECK: func.func private @"__builtin_rule2_SingleParam{x:[tensor<f64>]}{reg:1}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "CompilableData{}{wires:1}{a = \22a\22, b = \22b\22, thing = \22thing\22}" = 1 : i64
# CHECK-SAME:   target_gate = "SingleParam{x:[tensor<f64>]}{reg:1}{}"
test_multiple_rules_chained()


def test_multiple_rules_chained_and_branch():
    """
    Tests for when one rule involves another rule and then branches out, i.e. graph looks like
                    +---> C
                    |
        A --- B ---+
                    |
                    +---> D
    """

    @qp.register_resources(lambda reg: {SingleParam(x=Float, reg=Wire[1]): 1})
    def ruleAB(reg):
        SingleParam(x=0.1, reg=reg[0])

    @qp.register_resources(
        lambda x, reg: {CompilableData(a="a", b="b", thing="thing", wires=Wire[1]): 1}
    )
    def ruleBC(x, reg):
        CompilableData(a="a", b="b", thing="thing", wires=reg[0])

    @qp.register_resources(
        lambda x, reg: {CompilableData(a="alpha", b="beta", thing="stuff", wires=Wire[1]): 1}
    )
    def ruleBD(x, reg):
        CompilableData(a="alpha", b="beta", thing="stuff", wires=reg[0])

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, ruleAB)
        qp.add_decomps(SingleParam, ruleBC)
        qp.add_decomps(SingleParam, ruleBD)

        @qp.qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=3))
        def chained_branched():
            NoParams(reg=[0, 1])
            return qp.state()

        print(chained_branched.mlir)


# CHECK-LABEL: func.func public @chained_branched()
# CHECK: qref.operator "NoParams"
# CHECK: func.func private @"__builtin_ruleAB_NoParams{}{reg:2}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "SingleParam{x:[tensor<f64>]}{reg:1}{}" = 1 : i64
# CHECK-SAME:   target_gate = "NoParams{}{reg:2}{}"
# CHECK: func.func private @"__builtin_ruleBC_SingleParam{x:[tensor<f64>]}{reg:1}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "CompilableData{}{wires:1}{a = \22a\22, b = \22b\22, thing = \22thing\22}" = 1 : i64
# CHECK-SAME:   target_gate = "SingleParam{x:[tensor<f64>]}{reg:1}{}"
# CHECK: func.func private @"__builtin_ruleBD_SingleParam{x:[tensor<f64>]}{reg:1}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "CompilableData{}{wires:1}{a = \22alpha\22, b = \22beta\22, thing = \22stuff\22}" = 1 : i64
# CHECK-SAME:   target_gate = "SingleParam{x:[tensor<f64>]}{reg:1}{}"
test_multiple_rules_chained_and_branch()


def test_with_cycles():
    """
    Tests for when the decomposition rules have cycles, i.e. graph looks like
            A
            |
        +--+--+
        |     ^
        v     |
        B---> C
    """

    @qp.register_resources(lambda reg: {SingleParam(x=Float, reg=Wire[1]): 1})
    def ruleAB(reg):
        SingleParam(x=0.1, reg=reg[0])

    @qp.register_resources(
        lambda x, reg: {CompilableData(a="a", b="b", thing="thing", wires=Wire[1]): 1}
    )
    def ruleBC(x, reg):
        CompilableData(a="a", b="b", thing="thing", wires=reg[0])

    @qp.register_resources(lambda a, b, thing, wires: {NoParams(reg=Wire[2]): 1})
    def ruleCA(a, b, thing, wires):
        NoParams(reg=[0, 1])

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, ruleAB)
        qp.add_decomps(SingleParam, ruleBC)
        qp.add_decomps(CompilableData, ruleCA)

        @qp.qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=3))
        def cycle():
            NoParams(reg=[0, 1])
            NoParams(reg=[0, 1, 2])
            return qp.state()

        print(cycle.mlir)


# CHECK-LABEL: func.func public @cycle()
# CHECK: qref.operator "NoParams"
# CHECK: func.func private @"__builtin_ruleAB_NoParams{}{reg:2}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "SingleParam{x:[tensor<f64>]}{reg:1}{}" = 1 : i64
# CHECK-SAME:   target_gate = "NoParams{}{reg:2}{}"
# CHECK: func.func private @"__builtin_ruleAB_NoParams{}{reg:3}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "SingleParam{x:[tensor<f64>]}{reg:1}{}" = 1 : i64
# CHECK-SAME:   target_gate = "NoParams{}{reg:3}{}"
# CHECK: func.func private @"__builtin_ruleBC_SingleParam{x:[tensor<f64>]}{reg:1}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "CompilableData{}{wires:1}{a = \22a\22, b = \22b\22, thing = \22thing\22}" = 1 : i64
# CHECK-SAME:   target_gate = "SingleParam{x:[tensor<f64>]}{reg:1}{}"
# CHECK: func.func private @"__builtin_ruleCA_CompilableData{}{wires:1}{a = \22a\22, b = \22b\22, thing = \22thing\22}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "NoParams{}{reg:2}{}" = 1 : i64
# CHECK-SAME:   target_gate = "CompilableData{}{wires:1}{a = \22a\22, b = \22b\22, thing = \22thing\22}"
test_with_cycles()


def test_to_multiple_full_args_op():
    """
    Test that decomposing to an op with multiple names on all arg types works.
    """

    def rule_resource_fn(reg):
        return {
            MultipleFullArgs(
                reg1=Wire[1],
                reg2=Wire[2],
                angles1=Float,
                angles2=Float[2],
                pytree1=[1],
                pytree2=[2],
                op1=SingleParam(x=Float, reg=Wire[1]),
                op2=SingleParam(x=Int, reg=Wire[1]),
                hwires1=[Wire[1], Wire[1]],
                hwires2=[Wire[1]],
            ): 2
        }

    @qp.register_resources(rule_resource_fn)
    def rule(reg):
        MultipleFullArgs(
            reg1=reg[0],
            reg2=reg[1:3],
            angles1=0.1,
            angles2=jnp.array([0.1, 0.2]),
            pytree1=[1],
            pytree2=[2],
            op1=SingleParam(x=0.1, reg=[reg[0]]),
            op2=SingleParam(x=1, reg=[reg[1]]),
            hwires1=[qp.wires.Wires(reg[0]), qp.wires.Wires(reg[1])],
            hwires2=[qp.wires.Wires(reg[2])],
        )
        MultipleFullArgs(
            reg1=reg[2],
            reg2=reg[0:2],
            angles1=1.2,
            angles2=jnp.array([1.1, 1.2]),
            pytree1=[1],
            pytree2=[2],
            op1=SingleParam(x=1.1, reg=[reg[1]]),
            op2=SingleParam(x=2, reg=[reg[2]]),
            hwires1=[qp.wires.Wires(reg[1]), qp.wires.Wires(reg[2])],
            hwires2=[qp.wires.Wires(reg[0])],
        )

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule)

        @qp.qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=3))
        def to_full_args():
            NoParams(reg=[0, 1, 2])
            return qp.state()

        print(to_full_args.mlir)


# CHECK-LABEL: func.func public @to_full_args()
# CHECK: qref.operator "NoParams"
# CHECK: func.func private @"__builtin_rule_NoParams{}{reg:3}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "MultipleFullArgs{angles1:[tensor<f64>],angles2:[tensor<2xf64>]}{hwires1:2,hwires2:1,op1:1,op2:1,reg1:1,reg2:2}{}[[[uid:[0-9]+]]]" = 2 : i64
# CHECK-SAME:   target_gate = "NoParams{}{reg:3}{}"
# CHECK: qref.operator "MultipleFullArgs"
# CHECK-NEXT:  UID([[uid]]
# CHECK-NEXT:  param_map = {angles1 = [0], angles2 = [1]} qubit_map = {hwires1 = [5, 6], hwires2 = [7], op1 = [3], op2 = [4], reg1 = [0], reg2 = [1, 2]}
# CHECK: qref.operator "MultipleFullArgs"
# CHECK-NEXT:  UID([[uid]]
# CHECK-NEXT:  param_map = {angles1 = [0], angles2 = [1]} qubit_map = {hwires1 = [5, 6], hwires2 = [7], op1 = [3], op2 = [4], reg1 = [0], reg2 = [1, 2]}
test_to_multiple_full_args_op()


def test_from_multiple_full_args_op():
    """
    Test that decomposing from an op with multiple names on all arg types works.
    """

    def rule_resource_fn(
        reg1, reg2, angles1, angles2, pytree1, pytree2, op1, op2, hwires1, hwires2
    ):
        return {NoParams(reg=Wire[1]): 2}

    @qp.register_resources(rule_resource_fn)
    def rule(reg1, reg2, angles1, angles2, pytree1, pytree2, op1, op2, hwires1, hwires2):
        NoParams(reg=0)
        NoParams(reg=0)

    with qp.decomposition.local_decomps():
        qp.add_decomps(MultipleFullArgs, rule)

        @qp.qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=3))
        def from_full_args():
            MultipleFullArgs(
                reg1=1,
                reg2=2,
                angles1=0.1,
                angles2=0.2,
                pytree1=[],
                pytree2=[],
                op1=SingleParam(x=1.1, reg=[0]),
                op2=SingleParam(x=2, reg=[1]),
                hwires1=[qp.wires.Wires(1), qp.wires.Wires(2)],
                hwires2=[qp.wires.Wires(0)],
            )
            return qp.state()

        print(from_full_args.mlir)


# CHECK-LABEL: func.func public @from_full_args()
# CHECK: qref.operator "MultipleFullArgs"({{%.+}}: tensor<f64>, {{%.+}}: tensor<f64>)
# CHECK-SAME:   qubits({{%.}}, {{%.}}, {{%.}}, {{%.}}, {{%.}}, {{%.}}, {{%.}})
# CHECK:   UID([[uid:[0-9]+]]) forward({{%.+}}: tensor<f64>, {{%.+}}: tensor<i64>)
# CHECK:   param_map = {angles1 = [0], angles2 = [1]}
# CHECK-SAME:  qubit_map = {hwires1 = [4, 5], hwires2 = [6], op1 = [2], op2 = [3], reg1 = [0], reg2 = [1]}
#
# CHECK: func.func private @"__builtin_rule_MultipleFullArgs{angles1:[tensor<f64>],angles2:[tensor<f64>]}{hwires1:2,hwires2:1,op1:1,op2:1,reg1:1,reg2:1}{}[[[uid]]]"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "NoParams{}{reg:1}{}" = 2 : i64
# CHECK-SAME:   target_gate = "MultipleFullArgs{angles1:[tensor<f64>],angles2:[tensor<f64>]}{hwires1:2,hwires2:1,op1:1,op2:1,reg1:1,reg2:1}{}[[[uid]]]"
test_from_multiple_full_args_op()


def test_to_custom_op():
    """
    Test that decomposing to a custom op works.
    """

    def rule_resource_fn(reg):
        return {SingleParamCustomOp(x=Float, wires=Wire[1]): 1}

    @qp.register_resources(rule_resource_fn)
    def rule(reg):
        SingleParamCustomOp(x=0.1, wires=reg[0])

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule)

        @qp.qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=3))
        def to_custom():
            NoParams(reg=[0, 1, 2])
            return qp.state()

        print(to_custom.mlir)


# CHECK-LABEL: func.func public @to_custom()
# CHECK: qref.operator "NoParams"
# CHECK: func.func private @"__builtin_rule_NoParams{}{reg:3}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "SingleParamCustomOp{0:[f64]}{wires:1}{}" = 1 : i64
# CHECK-SAME:   target_gate = "NoParams{}{reg:3}{}"
# CHECK: qref.custom "SingleParamCustomOp"
test_to_custom_op()


def test_from_custom_op():
    """
    Test that decomposing from a custom op works.
    """

    def rule_resource_fn(x, wires):
        return {NoParams(reg=Wire[1]): 1}

    @qp.register_resources(rule_resource_fn)
    def rule(x, wires):
        NoParams(reg=0)

    with qp.decomposition.local_decomps():
        qp.add_decomps(SingleParamCustomOp, rule)

        @qp.qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=3))
        def from_custom():
            SingleParamCustomOp(x=0.1, wires=[0, 1])
            return qp.state()

        print(from_custom.mlir)


# CHECK-LABEL: func.func public @from_custom()
# CHECK: qref.custom "SingleParamCustomOp"
# CHECK: func.func private @"__builtin_rule_SingleParamCustomOp{0:[f64]}{wires:2}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "NoParams{}{reg:1}{}" = 1 : i64
# CHECK-SAME:   target_gate = "SingleParamCustomOp{0:[f64]}{wires:2}{}"
# CHECK: qref.operator "NoParams"
test_from_custom_op()


def test_phaseshift_to_rz():
    """Test that PhaseShift decomposes to a quantum.custom "RZ" & quantum.gphase correctly."""

    @qp.qjit(target="mlir", capture=True)
    @qp.qnode(qp.device("null.qubit", wires=2))
    def test_phaseshift():
        qp.PhaseShift(0.5, 1)

    print(test_phaseshift.mlir)


# CHECK-LABEL: test_phaseshift
# CHECK: func.func private @"__builtin__phaseshift_to_rz_gp
# CHECK-SAME: resources = {operations = {
# CHECK-SAME: GlobalPhase{phi:[f64]}{}{}
# CHECK-SAME: RZ{0:[f64]}{wires:1}{}
# CHECK: qref.custom "RZ"
# CHECK: qref.gphase
test_phaseshift_to_rz()


def test_basis_rotation_decomposition():
    """Test that qp.BasisRotation decompositions show up properly in MLIR."""

    def test_basis_rotation_complex_valued():
        """Test that a complex-valued qp.BasisRotation compiles its complex decomposition rule."""

        @qp.qjit(target="mlir", capture=True)
        @qp.qnode(qp.device("null.qubit", wires=2))
        def test_basis_rotation_complex_valued():
            U = jnp.array(
                [
                    [-0.77228482 + 0.0j, -0.02959195 + 0.63458685j],
                    [0.63527644 + 0.0j, -0.03597397 + 0.77144651j],
                ],
            )
            qp.BasisRotation(unitary_matrix=U, wires=[0, 1])
            return qp.probs()

        print(test_basis_rotation_complex_valued.mlir)

    # CHECK-LABEL: test_basis_rotation_complex_valued
    # CHECK: func.func private @"__builtin__complex_basis_rotation_decomp_BasisRotation{unitary_matrix:[tensor<2x2xcomplex<f64>>]}{wires:2}{check = false}"
    # CHECK-SAME: resources = {operations = {
    # CHECK-SAME: "PhaseShift{0:[f64]}{wires:1}{}" = 3 : i64
    # CHECK-SAME: "SingleExcitation{0:[f64]}{wires:2}{}" = 1 : i64
    # CHECK-SAME: target_gate = "BasisRotation{unitary_matrix:[tensor<2x2xcomplex<f64>>]}{wires:2}{check = false}"
    test_basis_rotation_complex_valued()

    def test_basis_rotation_real_valued():
        """Test that a real-valued qp.BasisRotation compiles the real decomposition rule instead, keyed
        on a real ``unitary_matrix``. A real and a complex matrix are separate operators with separate
        rules and separate resource counts, so neither may be named with the other's data type."""

        @qp.qjit(target="mlir", capture=True)
        @qp.qnode(qp.device("null.qubit", wires=2))
        def test_basis_rotation_real_valued():
            # A real orthogonal matrix with determinant -1, so the determinant-fixing PhaseShift runs.
            U = jnp.array([[0.76484219, 0.64421769], [0.64421769, -0.76484219]])
            qp.BasisRotation(unitary_matrix=U, wires=[0, 1])
            return qp.probs()

        print(test_basis_rotation_real_valued.mlir)

    # CHECK-LABEL: test_basis_rotation_real_valued
    # CHECK: func.func private @"__builtin__real_basis_rotation_decomp_BasisRotation{unitary_matrix:[tensor<2x2xf64>]}{wires:2}{check = false}"
    # CHECK-SAME: resources = {operations = {
    # CHECK-SAME: "PhaseShift{0:[f64]}{wires:1}{}" = 1 : i64
    # CHECK-SAME: "SingleExcitation{0:[f64]}{wires:2}{}" = 1 : i64
    # CHECK-SAME: target_gate = "BasisRotation{unitary_matrix:[tensor<2x2xf64>]}{wires:2}{check = false}"
    test_basis_rotation_real_valued()

    def _compile_parent(U):
        """Lower a parent operator whose decomposition rule emits a ``BasisRotation`` carrying the
        parent's own matrix, and return the MLIR.
        """

        def rule_resource_fn(matrix, wires):
            spec = Complex if qp.math.get_dtype_name(matrix).startswith("complex") else Float
            return {qp.BasisRotation(spec[2, 2], Wire[2]): 1}

        @qp.register_resources(rule_resource_fn)
        def rule(matrix, wires):
            qp.BasisRotation(matrix, wires)

        with qp.decomposition.local_decomps():
            qp.add_decomps(TestQubitUnitary, rule)

            @qp.qjit(target="mlir", capture=True)
            @qp.qnode(qp.device("null.qubit", wires=2))
            def parent_circuit():
                TestQubitUnitary(U, [0, 1])
                return qp.probs()

            return parent_circuit.mlir

    def test_real_op_that_decomposes_to_basis_rotation():
        """A parent holding a real matrix must declare, emit and find a real ``BasisRotation``."""

        # A real orthogonal matrix with determinant -1, so the determinant-fixing PhaseShift runs.
        print(_compile_parent(jnp.array([[0.76484219, 0.64421769], [0.64421769, -0.76484219]])))

    # CHECK: func.func private @"__builtin_rule_TestQubitUnitary{matrix:[tensor<2x2xf64>]}{wires:2}{}"
    # CHECK-SAME: resources = {operations = {
    # CHECK-SAME: "BasisRotation{unitary_matrix:[tensor<2x2xf64>]}{wires:2}{check = false}" = 1 : i64
    # CHECK-SAME: target_gate = "TestQubitUnitary{matrix:[tensor<2x2xf64>]}{wires:2}{}"
    # CHECK: qref.operator "BasisRotation"(
    # CHECK-SAME: tensor<2x2xf64>
    # CHECK: func.func private @"__builtin__real_basis_rotation_decomp_BasisRotation{unitary_matrix:[tensor<2x2xf64>]}{wires:2}{check = false}"
    # CHECK-SAME: "PhaseShift{0:[f64]}{wires:1}{}" = 1 : i64
    # CHECK-SAME: "SingleExcitation{0:[f64]}{wires:2}{}" = 1 : i64
    # CHECK-SAME: target_gate = "BasisRotation{unitary_matrix:[tensor<2x2xf64>]}{wires:2}{check = false}"
    test_real_op_that_decomposes_to_basis_rotation()

    def test_complex_op_that_decomposes_to_basis_rotation():
        """The same parent holding a complex matrix must route to the complex rule instead."""

        print(
            _compile_parent(
                jnp.array(
                    [
                        [-0.77228482 + 0.0j, -0.02959195 + 0.63458685j],
                        [0.63527644 + 0.0j, -0.03597397 + 0.77144651j],
                    ],
                )
            )
        )

    # CHECK: func.func private @"__builtin_rule_TestQubitUnitary{matrix:[tensor<2x2xcomplex<f64>>]}{wires:2}{}"
    # CHECK-SAME: resources = {operations = {
    # CHECK-SAME: "BasisRotation{unitary_matrix:[tensor<2x2xcomplex<f64>>]}{wires:2}{check = false}" = 1 : i64
    #
    # CHECK: func.func private @"__builtin__complex_basis_rotation_decomp_BasisRotation{unitary_matrix:[tensor<2x2xcomplex<f64>>]}{wires:2}{check = false}"
    # CHECK-SAME: "PhaseShift{0:[f64]}{wires:1}{}" = 3 : i64
    # CHECK-SAME: "SingleExcitation{0:[f64]}{wires:2}{}" = 1 : i64
    test_complex_op_that_decomposes_to_basis_rotation()


test_basis_rotation_decomposition()


def test_ctrl_rule_is_traversed():
    """
    Test that rules from a controlled version of a base gate are compiled and traversed from when
    the circuit just has the base gate.

    In this test, a C(NoParams) decomposes to a CompilableData, and a CompilableData decomposes to
    a SingleParam. We test that just from the base NoParams, both rules are present.
    """

    @qp.register_resources({CompilableData(a="a", b="b", thing="thing", wires=Wire[1]): 1})
    def ctrl_rule(base, control_wires, control_values, work_wires, work_wire_type):
        CompilableData(a="a", b="b", thing="thing", wires=base.wires)

    @qp.register_resources({SingleParam(x=Float, reg=Wire[1]): 1})
    def rule(a, b, thing, wires):
        SingleParam(x=0.1, reg=wires[0])

    with qp.decomposition.local_decomps():
        qp.add_decomps("C(NoParams)", ctrl_rule)
        qp.add_decomps(CompilableData, rule)

        @qp.qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=3))
        def ctrl_is_traversed():
            NoParams(reg=0)
            return qp.probs()

        print(ctrl_is_traversed.mlir)


# CHECK-LABEL: func.func public @ctrl_is_traversed()
# CHECK: qref.operator "NoParams"
#
# CHECK: func.func private @"__builtin_ctrl_rule_C(NoParams){}{reg:1}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "CompilableData{}{wires:1}{a = \22a\22, b = \22b\22, thing = \22thing\22}" = 1 : i64
# CHECK-SAME:   target_gate = "C(NoParams){}{reg:1}{}"
# CHECK: qref.operator "CompilableData"
#
# CHECK: func.func private @"__builtin_rule_CompilableData{}{wires:1}{a = \22a\22, b = \22b\22, thing = \22thing\22}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "SingleParam{x:[tensor<f64>]}{reg:1}{}" = 1 : i64
# CHECK-SAME:   target_gate = "CompilableData{}{wires:1}{a = \22a\22, b = \22b\22, thing = \22thing\22}"
# CHECK: qref.operator "SingleParam"
test_ctrl_rule_is_traversed()


def test_rule_uniqueness():
    """Test that unique rules with equivalent resources are still both lowered."""

    @qp.register_resources({SingleParam(Float, Wire[1]): 1})
    def one_rule(reg):
        SingleParam(0.5, reg[0])

    @qp.register_resources({SingleParam(Float, Wire[1]): 1})
    def two_rule(reg):
        SingleParam(0.5, reg[0])

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, one_rule, two_rule)

        @qp.qjit(target="mlir", capture=True)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            NoParams(0)
            return qp.probs()

        print(circuit.mlir)


# CHECK-LABEL: func.func private @"__builtin_one_rule
# CHECK-LABEL: func.func private @"__builtin_two_rule
test_rule_uniqueness()
