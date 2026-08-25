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
Unit tests for the compile_decomposition_rules_wrapper() entry point function.

This function is the unified entry point for all pathways in Catalyst that generate decomposition
rules, including precompiled rules, lower time rules, and on-demand rules.
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable = missing-class-docstring,line-too-long,unused-argument

import pennylane as qp
from jax import numpy as jnp
from operator2_dummy_gates import (
    CompilableData,
    HybridOpArg,
    HybridWires,
    MultiParams,
    MultipleFullArgs,
    MultipleRegisters,
    NoParams,
    NoParamsCustomOp,
    SingleParam,
    SingleParamCustomOp,
    StaticData,
    StaticDataMultiReg,
)
from pennylane.core import Operator2
from pennylane.typing import Complex, Float, Int, Wire

from catalyst.decomposition.decomposition_rules import (
    compile_decomposition_rules_wrapper,
)


def test_to_dynamic_argnames():
    """
    Test that decomposing to an op with a single dynamic_argname works.
    """

    def rule_resource_fn(reg):
        return {
            SingleParam(x=Float, reg=Wire[2]): 2,
            SingleParam(x=Float[2], reg=Wire[2]): 1,
        }

    @qp.register_resources(rule_resource_fn)
    def rule(reg):
        SingleParam(x=0.1, reg=reg)
        SingleParam(x=0.2, reg=reg)
        SingleParam(x=jnp.array([0.3, 0.4]), reg=reg)

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule)
        result = compile_decomposition_rules_wrapper(
            "NoParams", "NoParams{}{reg:2}{}", {}, {"reg": 2}, {}
        )
        print(result)


# CHECK: func.func private @"rule_NoParams{}{reg:2}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "SingleParam{x:{{\[\[f64,f64\]\]}}}{reg:2}{}" = 1 : i64,
# CHECK-SAME:   "SingleParam{x:{{\[\[f64\]\]}}}{reg:2}{}" = 2 : i64
# CHECK-SAME:   target_gate = "NoParams{}{reg:2}{}"
test_to_dynamic_argnames()


def test_from_dynamic_argnames():
    """
    Test that decomposing from an op with a single dynamic_argname works.
    """

    def rule_resource_fn(x, reg):
        return {NoParams(reg=Wire[1]): 1}

    @qp.register_resources(rule_resource_fn)
    def rule(x, reg):
        NoParams(reg=reg[0])

    with qp.decomposition.local_decomps():
        qp.add_decomps(SingleParam, rule)
        result = compile_decomposition_rules_wrapper(
            "SingleParam",
            "SingleParam{x:[[f64,f64]]}{reg:2}{}",
            {"x": ["f64", "f64"]},
            {"reg": 2},
            {},
        )
        print(result)


# CHECK: func.func private @"rule_SingleParam{x:{{\[\[f64,f64\]\]}}}{reg:2}{}"
# CHECK-SAME:   resources = {operations = {"NoParams{}{reg:1}{}" = 1 : i64}}
# CHECK-SAME:   target_gate = "SingleParam{x:{{\[\[f64,f64\]\]}}}{reg:2}{}"
test_from_dynamic_argnames()


def test_to_multiple_dynamic_argnames():
    """
    Test that decomposing to an op with multiple dynamic_argnames works.
    """

    def rule_resource_fn(reg):
        return {
            MultiParams(reg=Wire[1], a=Float, b=Int[2, 2], c=Complex): 1,
        }

    @qp.register_resources(rule_resource_fn)
    def rule(reg):
        MultiParams(reg=reg, a=0.1, b=jnp.array([[1, 2], [3, 4]]), c=3 + 4j)

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule)
        result = compile_decomposition_rules_wrapper(
            "NoParams", "NoParams{}{reg:1}{}", {}, {"reg": 1}, {}
        )
        print(result)


# CHECK: func.func private @"rule_NoParams{}{reg:1}{}"
# CHECK-SAME:   resources = {operations =
# CHECK-SAME:   "MultiParams{a:{{\[\[f64\]\]}},b:{{\[\[\[i64,i64\],\[i64,i64\]\]\]}},c:{{\[\[complex<f64>\]\]}}}{reg:1}{}" = 1 : i64
# CHECK-SAME:   target_gate = "NoParams{}{reg:1}{}"
test_to_multiple_dynamic_argnames()


def test_from_multiple_dynamic_argnames():
    """
    Test that decomposing from an op with multiple dynamic_argnames works.
    """

    def rule_resource_fn(reg, a, b, c):
        return {NoParamsCustomOp(wires=Wire[2]): 1}

    @qp.register_resources(rule_resource_fn)
    def rule(reg, a, b, c):
        NoParamsCustomOp(wires=reg)

    with qp.decomposition.local_decomps():
        qp.add_decomps(MultiParams, rule)
        result = compile_decomposition_rules_wrapper(
            "MultiParams",
            "MultiParams{a:[[f64]],b:[[i32,f64]],c:[[[i32],[f64]]]}{reg:2}{}",
            {"b": ["i32", "f64"], "c": [["i32"], ["f64"]], "a": ["f64"]},
            {"reg": 2},
            {},
        )
        print(result)


# CHECK: func.func private @"rule_MultiParams{a:{{\[\[f64\]\]}},b:{{\[\[i32,f64\]\]}},c:{{\[\[\[i32\],\[f64\]\]\]}}}{reg:2}{}"
# CHECK-SAME:   resources = {operations = {"NoParamsCustomOp{}{wires:2}{}" = 1 : i64}
# CHECK-SAME:   target_gate = "MultiParams{a:{{\[\[f64\]\]}},b:{{\[\[i32,f64\]\]}},c:{{\[\[\[i32\],\[f64\]\]\]}}}{reg:2}{}"
test_from_multiple_dynamic_argnames()


def test_to_multiple_wire_argnames():
    """
    Test that decomposing to an op with multiple wire_argnames works.
    """

    def rule_resource_fn(reg):
        return {
            MultipleRegisters(reg1=Wire[1], reg2=Wire[2]): 1,
        }

    @qp.register_resources(rule_resource_fn)
    def rule(reg):
        MultipleRegisters(reg1=0, reg2=[0, 1])

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule)
        result = compile_decomposition_rules_wrapper(
            "NoParams", "NoParams{}{reg:1}{}", {}, {"reg": 1}, {}
        )
        print(result)


# CHECK: func.func private @"rule_NoParams{}{reg:1}{}"
# CHECK-SAME:   resources = {operations =
# CHECK-SAME:   "MultipleRegisters{}{reg1:1,reg2:2}{}" = 1 : i64
# CHECK-SAME:   target_gate = "NoParams{}{reg:1}{}"
test_to_multiple_wire_argnames()


def test_from_multiple_wire_argnames():
    """
    Test that decomposing from an op with multiple wire_argnames works.
    """

    def rule_resource_fn(reg1, reg2):
        return {NoParamsCustomOp(wires=Wire[2]): 1, NoParamsCustomOp(wires=Wire[3]): 1}

    @qp.register_resources(rule_resource_fn)
    def rule(reg1, reg2):
        NoParamsCustomOp(wires=reg1)
        NoParamsCustomOp(wires=reg2)

    with qp.decomposition.local_decomps():
        qp.add_decomps(MultipleRegisters, rule)
        result = compile_decomposition_rules_wrapper(
            "MultipleRegisters",
            "MultipleRegisters{}{reg1:2,reg2:3}{}",
            {},
            {"reg1": 2, "reg2": 3},
            {},
        )
        print(result)


# CHECK: func.func private @"rule_MultipleRegisters{}{reg1:2,reg2:3}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "NoParamsCustomOp{}{wires:2}{}" = 1 : i64
# CHECK-SAME:   "NoParamsCustomOp{}{wires:3}{}" = 1 : i64
# CHECK-SAME:   target_gate = "MultipleRegisters{}{reg1:2,reg2:3}{}"
test_from_multiple_wire_argnames()


def test_to_compilable_data():
    """
    Test that decomposing to an op with compilable_argnames works.
    """

    def rule_resource_fn(reg):
        return {
            CompilableData("a", "b", "thing", Wire[1]): 1,
            CompilableData("aa", "bb", "stuff", Wire[1]): 2,
        }

    @qp.register_resources(rule_resource_fn)
    def rule(reg):
        CompilableData(a="a", b="b", thing="thing", wires=reg[0])
        CompilableData(a="aa", b="bb", thing="stuff", wires=reg[1])
        CompilableData(a="aa", b="bb", thing="stuff", wires=reg[1])

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule)
        result = compile_decomposition_rules_wrapper(
            "NoParams", "NoParams{}{reg:2}{}", {}, {"reg": 2}, {}
        )
        print(result)


# CHECK: func.func private @"rule_NoParams{}{reg:2}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:     "CompilableData{}{wires:1}{a:a,b:b,thing:thing}" = 1 : i64,
# CHECK-SAME:     "CompilableData{}{wires:1}{a:aa,b:bb,thing:stuff}" = 2 : i64
# CHECK-SAME:   target_gate = "NoParams{}{reg:2}{}"
test_to_compilable_data()


def test_from_compilable_data():
    """
    Test that decomposing from an op with compilable_argnames works.
    """

    def rule_resource_fn(a, b, thing, wires):
        return {SingleParam(x=Float, reg=Wire[1]): 1}

    @qp.register_resources(rule_resource_fn)
    def rule(a, b, thing, wires):
        if a == 1:
            SingleParam(x=0.1, reg=wires)
        else:
            SingleParam(x=1.1, reg=wires)

    with qp.decomposition.local_decomps():
        qp.add_decomps(CompilableData, rule)
        result_a1 = compile_decomposition_rules_wrapper(
            "CompilableData",
            "CompilableData{}{wires:1}{a:1,b:2,thing:3}",
            {},
            {"wires": 1},
            {"a": 1, "b": 2, "thing": 3},
        )
        print(result_a1)

        result_a10 = compile_decomposition_rules_wrapper(
            "CompilableData",
            "CompilableData{}{wires:1}{a:10,b:2,thing:3}",
            {},
            {"wires": 1},
            {"a": 10, "b": 2, "thing": 3},
        )
        print(result_a10)


# CHECK: func.func private @"rule_CompilableData{}{wires:1}{a:1,b:2,thing:3}"
# CHECK-SAME:   resources = {operations = {"SingleParam{x:{{\[\[f64\]\]}}}{reg:1}{}" = 1 : i64}
# CHECK-SAME:   target_gate = "CompilableData{}{wires:1}{a:1,b:2,thing:3}"
# CHECK: stablehlo.constant dense<1.000000e-01> : tensor<f64>
# CHECK-NOT: stablehlo.constant dense<1.100000e+00> : tensor<f64>
#
# CHECK: func.func private @"rule_CompilableData{}{wires:1}{a:10,b:2,thing:3}"
# CHECK-SAME:   resources = {operations = {"SingleParam{x:{{\[\[f64\]\]}}}{reg:1}{}" = 1 : i64}
# CHECK-SAME:   target_gate = "CompilableData{}{wires:1}{a:10,b:2,thing:3}"
# CHECK: stablehlo.constant dense<1.100000e+00> : tensor<f64>
# CHECK-NOT: stablehlo.constant dense<1.000000e-01> : tensor<f64>
test_from_compilable_data()


def test_to_static_data():
    """
    Test that decomposing to an op with static_argnames works.
    """

    def rule_resource_fn(reg):
        return {
            StaticData(label="hello", reg=Wire[1]): 1,
            StaticData(label="world", reg=Wire[1]): 2,
        }

    @qp.register_resources(rule_resource_fn)
    def rule(reg):
        StaticData(label="hello", reg=reg[0])
        StaticData(label="world", reg=reg[1])
        StaticData(label="world", reg=reg[0])

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule)
        result = compile_decomposition_rules_wrapper(
            "NoParams", "NoParams{}{reg:2}{}", {}, {"reg": 2}, {}
        )
        print(result)


# CHECK: func.func private @"rule_NoParams{}{reg:2}{}"
# CHECK-DAG: "StaticData{}{reg:1}{}[[[uid_1:[0-9]+]]]" = 1
# CHECK-DAG: "StaticData{}{reg:1}{}[[[uid_2:[0-9]+]]]" = 2
# CHECK-DAG:   target_gate = "NoParams{}{reg:2}{}"
# CHECK: "qref.operator"
# CHECK-SAME: UID = [[uid_1]]
# CHECK: "qref.operator"
# CHECK-SAME: UID = [[uid_2]]
# CHECK: "qref.operator"
# CHECK-SAME: UID = [[uid_2]]
test_to_static_data()


def test_from_static_data():
    """
    Test that decomposing from an op with static_argnames works.
    """

    def rule_resource_fn(label, reg):
        if label == 1234:
            return {SingleParam(x=Float, reg=Wire[1]): 1}
        else:
            return {SingleParam(x=Float, reg=Wire[1]): 2}

    @qp.register_resources(rule_resource_fn)
    def rule(label, reg):
        if label == 1234:
            SingleParam(x=0.1, reg=reg)
        else:
            SingleParam(x=1.1, reg=reg)
            SingleParam(x=2.2, reg=reg)

    with qp.decomposition.local_decomps():
        qp.add_decomps(StaticData, rule)
        result_1234 = compile_decomposition_rules_wrapper(
            "StaticData", "StaticData{}{reg:1}{}[1234]", {}, {"reg": 1}, {}, {"label": 1234}
        )
        print(result_1234)

        result_4321 = compile_decomposition_rules_wrapper(
            "StaticData", "StaticData{}{reg:1}{}[4321]", {}, {"reg": 1}, {}, {"label": 4321}
        )
        print(result_4321)


# CHECK: func.func private @"rule_StaticData{}{reg:1}{}[1234]"
# CHECK-SAME:   resources = {operations = {"SingleParam{x:{{\[\[f64\]\]}}}{reg:1}{}" = 1 : i64}
# CHECK-SAME:   target_gate = "StaticData{}{reg:1}{}[1234]"
# CHECK: stablehlo.constant dense<1.000000e-01> : tensor<f64>
#
# CHECK: func.func private @"rule_StaticData{}{reg:1}{}[4321]"
# CHECK-SAME:   resources = {operations = {"SingleParam{x:{{\[\[f64\]\]}}}{reg:1}{}" = 2 : i64}
# CHECK-SAME:   target_gate = "StaticData{}{reg:1}{}[4321]"
# CHECK-DAG: stablehlo.constant dense<1.100000e+00> : tensor<f64>
# CHECK-DAG: stablehlo.constant dense<2.200000e+00> : tensor<f64>
test_from_static_data()


def test_to_hybrid_wires():
    """
    Test that decomposing to an op with a wire-like hybrid_argnames works.
    """

    def rule_resource_fn(reg):
        return {
            HybridWires(cwires=[Wire[1]]): 2,
            HybridWires(cwires=[Wire[1], [Wire[1], Wire[1]]]): 1,
        }

    @qp.register_resources(rule_resource_fn)
    def rule(reg):
        HybridWires(cwires=[qp.wires.Wires(0)])
        HybridWires(cwires=[qp.wires.Wires(1)])
        HybridWires(
            cwires=[qp.wires.Wires(reg[0]), [qp.wires.Wires(reg[1]), qp.wires.Wires(reg[2])]]
        )

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule)
        result = compile_decomposition_rules_wrapper(
            "NoParams", "NoParams{}{reg:3}{}", {}, {"reg": 3}, {}
        )
        print(result)


# CHECK: func.func private @"rule_NoParams{}{reg:3}{}"
# CHECK-DAG: "HybridWires{}{}{}[[[uid_1:[0-9]+]]]" = 1
# CHECK-DAG: "HybridWires{}{}{}[[[uid_2:[0-9]+]]]" = 2
# CHECK-DAG:   target_gate = "NoParams{}{reg:3}{}"
# CHECK: "qref.operator"
# CHECK-SAME: UID = [[uid_2]]
# CHECK: "qref.operator"
# CHECK-SAME: UID = [[uid_2]]
# CHECK: "qref.operator"
# CHECK-SAME: UID = [[uid_1]]
test_to_hybrid_wires()


def test_from_hybrid_wires():
    """
    Test that decomposing from an op with a wire-like hybrid_argnames works.
    """

    def rule_resource_fn(cwires):
        return {NoParams(reg=Wire[1]): 3}

    @qp.register_resources(rule_resource_fn)
    def rule(cwires):
        NoParams(reg=cwires[0])
        NoParams(reg=cwires[1][0])
        NoParams(reg=cwires[1][1])

    with qp.decomposition.local_decomps():
        qp.add_decomps(HybridWires, rule)
        result = compile_decomposition_rules_wrapper(
            "HybridWires",
            "HybridWires{}{}{}[3742]",
            {},
            {},
            {},
            extra_data={"cwires": [1, [2, qp.wires.Wires(3)]]},
        )
        print(result)


# CHECK: func.func private @"rule_HybridWires{}{}{}[3742]"
# CHECK-SAME:   resources = {operations = {"NoParams{}{reg:1}{}" = 3 : i64}}
# CHECK-SAME:   target_gate = "HybridWires{}{}{}[3742]"
# CHECK: idx_attr = 1
# CHECK: idx_attr = 2
# CHECK: idx_attr = 3
test_from_hybrid_wires()


def test_to_hybrid_op():
    """
    Test that decomposing to an op with a non-wire-like hybrid_argnames works.
    """

    def rule_resource_fn(reg):
        return {
            HybridOpArg(
                angle=Float,
                op=StaticDataMultiReg(label="hello", reg=Wire[1], reg2=Wire[2], theta=Float),
                cwires=Wire[1],
                n_iters=100,
            ): 2,
            HybridOpArg(
                angle=Float,
                op=StaticDataMultiReg(label="hello", reg=Wire[1], reg2=Wire[2], theta=Float),
                cwires=Wire[1],
                n_iters=101,
            ): 1,
        }

    @qp.register_resources(rule_resource_fn)
    def rule(reg):
        HybridOpArg(
            angle=0.1,
            op=StaticDataMultiReg("hello", reg=[42], reg2=[11, 12], theta=0.4),
            cwires=37,
            n_iters=100,
        )
        HybridOpArg(
            angle=0.2,
            op=StaticDataMultiReg("hello", reg=[1], reg2=[2, 3], theta=0.2),
            cwires=4,
            n_iters=100,
        )
        HybridOpArg(
            angle=0.3,
            op=StaticDataMultiReg("hello", reg=[42], reg2=[11, 12], theta=0.4),
            cwires=37,
            n_iters=101,
        )

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule)
        result = compile_decomposition_rules_wrapper(
            "NoParams", "NoParams{}{reg:3}{}", {}, {"reg": 3}, {}
        )
        print(result)


# CHECK: func.func private @"rule_NoParams{}{reg:3}{}"
# CHECK-DAG: "HybridOpArg{angle:{{\[\[f64\]\]}}}{cwires:1}{}[[[uid_1:[0-9]+]]]" = 1
# CHECK-DAG: "HybridOpArg{angle:{{\[\[f64\]\]}}}{cwires:1}{}[[[uid_2:[0-9]+]]]" = 2
# CHECK-DAG:   target_gate = "NoParams{}{reg:3}{}"
# CHECK: "qref.operator"
# CHECK-SAME: UID = [[uid_2]]
# CHECK: "qref.operator"
# CHECK-SAME: UID = [[uid_2]]
# CHECK: "qref.operator"
# CHECK-SAME: UID = [[uid_1]]
test_to_hybrid_op()


def test_from_hybrid_op():
    """
    Test that decomposing from an op with a non-wire-like hybrid_argnames works.
    """

    def rule_resource_fn(angle, op, cwires, n_iters):
        return {NoParams(reg=Wire[1]): 1, op: 1}

    @qp.register_resources(rule_resource_fn)
    def rule(angle, op, cwires, n_iters):
        qp.apply(op)
        NoParams(reg=cwires[0])

    with qp.decomposition.local_decomps():
        qp.add_decomps(HybridOpArg, rule)
        result = compile_decomposition_rules_wrapper(
            "HybridOpArg",
            "HybridOpArg{angle:[[f64]]}{cwires:1}{}[5678]",
            {"angle": ["f64"]},
            {"cwires": 1},
            {},
            extra_data={
                "op": StaticDataMultiReg("hello", reg=[1], reg2=[2, 3], theta=0.2),
                "n_iters": 100,
            },
        )
        print(result)


# CHECK: func.func private @"rule_HybridOpArg{angle:{{\[\[f64\]\]}}}{cwires:1}{}[5678]"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "NoParams{}{reg:1}{}" = 1 : i64,
# CHECK-SAME:   "StaticDataMultiReg{theta:{{\[\[f64\]\]}}}{reg:1,reg2:2}{}[[[uid:[0-9]+]]]" = 1 : i64
# CHECK-SAME:   target_gate = "HybridOpArg{angle:{{\[\[f64\]\]}}}{cwires:1}{}[5678]"
# CHECK: "qref.operator"
# CHECK-SAME: UID = [[uid]] : i64, op_name = "StaticDataMultiReg"
test_from_hybrid_op()


def test_to_hybrid_op_nested():
    """
    Test that decomposing to an op with a hybrid_argnames that is a nested op works.
    """

    def rule_resource_fn(reg):
        return {
            HybridOpArg(
                angle=Float,
                op=HybridOpArg(
                    angle=Float,
                    op=StaticDataMultiReg(label="hello", reg=Wire[1], reg2=Wire[2], theta=Float),
                    cwires=Wire[3],
                    n_iters=200,
                ),
                cwires=Wire[1],
                n_iters=100,
            ): 1,
        }

    @qp.register_resources(rule_resource_fn)
    def rule(reg):
        HybridOpArg(
            angle=0.1,
            op=HybridOpArg(
                angle=0.2,
                op=StaticDataMultiReg("hello", reg=[42], reg2=[11, 12], theta=0.4),
                cwires=[100, 200, 300],
                n_iters=200,
            ),
            cwires=37,
            n_iters=100,
        )

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule)
        result = compile_decomposition_rules_wrapper(
            "NoParams", "NoParams{}{reg:3}{}", {}, {"reg": 3}, {}
        )
        print(result)


# CHECK: func.func private @"rule_NoParams{}{reg:3}{}"
# CHECK-SAME: "HybridOpArg{angle:{{\[\[f64\]\]}}}{cwires:1}{}[[[uid:[0-9]+]]]" = 1
# CHECK-SAME:   target_gate = "NoParams{}{reg:3}{}"
# CHECK: "qref.operator"
# CHECK-SAME: UID = [[uid]]
test_to_hybrid_op_nested()


def test_from_hybrid_op_nested():
    """
    Test that decomposing from an op with a hybrid_argnames that is a nested op works.
    """

    def rule_resource_fn(angle, op, cwires, n_iters):
        return {NoParams(reg=Wire[1]): 1, op: 1, op.op: 1}

    @qp.register_resources(rule_resource_fn)
    def rule(angle, op, cwires, n_iters):
        qp.apply(op)
        qp.apply(op.op)
        NoParams(reg=cwires[0])

    with qp.decomposition.local_decomps():
        qp.add_decomps(HybridOpArg, rule)
        result = compile_decomposition_rules_wrapper(
            "HybridOpArg",
            "HybridOpArg{angle:[[f64]]}{cwires:1}{}[7654]",
            {"angle": ["f64"]},
            {"cwires": 1},
            {},
            extra_data={
                "op": HybridOpArg(
                    angle=0.2,
                    op=StaticDataMultiReg("hello", reg=[1], reg2=[2, 3], theta=0.2),
                    cwires=2,
                    n_iters=200,
                ),
                "n_iters": 100,
            },
        )
        print(result)


# CHECK: func.func private @"rule_HybridOpArg{angle:{{\[\[f64\]\]}}}{cwires:1}{}[7654]"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "HybridOpArg{angle:{{\[\[f64\]\]}}}{cwires:1}{}[[[uid_outer:[0-9]+]]]" = 1 : i64,
# CHECK-SAME:   "NoParams{}{reg:1}{}" = 1 : i64,
# CHECK-SAME:   "StaticDataMultiReg{theta:{{\[\[f64\]\]}}}{reg:1,reg2:2}{}[[[uid_inner:[0-9]+]]]" = 1 : i64
# CHECK-SAME:   target_gate = "HybridOpArg{angle:{{\[\[f64\]\]}}}{cwires:1}{}[7654]"
# CHECK: "qref.operator"
# CHECK-SAME: UID = [[uid_outer]] : i64, op_name = "HybridOpArg"
# CHECK: "qref.operator"
# CHECK-SAME: UID = [[uid_inner]] : i64, op_name = "StaticDataMultiReg"
test_from_hybrid_op_nested()


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
        result = compile_decomposition_rules_wrapper(
            "NoParams", "NoParams{}{reg:3}{}", {}, {"reg": 3}, {}
        )
        print(result)


# CHECK: func.func private @"rule_NoParams{}{reg:3}{}"
# CHECK-DAG: "MultipleFullArgs{angles1:{{\[\[f64\]\]}},angles2:{{\[\[f64,f64\]\]}}}{reg1:1,reg2:2}{}[[[uid:[0-9]+]]]" = 2
# CHECK-DAG:   target_gate = "NoParams{}{reg:3}{}"
# CHECK: "qref.operator"
# CHECK-SAME: UID = [[uid]]
# CHECK: "qref.operator"
# CHECK-SAME: UID = [[uid]]
test_to_multiple_full_args_op()


def test_from_multiple_full_args_op():
    """
    Test that decomposing from an op with multiple names on all arg types works.
    """

    def rule_resource_fn(
        reg1, reg2, angles1, angles2, pytree1, pytree2, op1, op2, hwires1, hwires2
    ):
        return {NoParams(reg=Wire[1]): 1, op1: 1, op2: 1}

    @qp.register_resources(rule_resource_fn)
    def rule(reg1, reg2, angles1, angles2, pytree1, pytree2, op1, op2, hwires1, hwires2):
        qp.apply(op1)
        qp.apply(op2)
        NoParams(reg=hwires1[0])

    with qp.decomposition.local_decomps():
        qp.add_decomps(MultipleFullArgs, rule)
        result = compile_decomposition_rules_wrapper(
            "MultipleFullArgs",
            "MultipleFullArgs{angles1:[[f64]],angles2:[[f64,f64]]}{reg1:1,reg2:2}{}[4444]",
            {"angles1": ["f64"], "angles2": ["f64", "f64"]},
            {"reg1": 1, "reg2": 2},
            {},
            extra_data={
                "pytree1": [1],
                "pytree2": [2],
                "op1": SingleParam(x=1.1, reg=[1]),
                "op2": SingleParam(x=2, reg=[2]),
                "hwires1": [qp.wires.Wires(3), qp.wires.Wires(2)],
                "hwires2": [qp.wires.Wires(1)],
            },
        )
        print(result)


# CHECK: func.func private @"rule_MultipleFullArgs{angles1:{{\[\[f64\]\]}},angles2:{{\[\[f64,f64\]\]}}}{reg1:1,reg2:2}{}[4444]"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "NoParams{}{reg:1}{}" = 1 : i64
# CHECK-SAME:   "SingleParam{x:{{\[\[f64\]\]}}}{reg:1}{}" = 1 : i64
# CHECK-SAME:   "SingleParam{x:{{\[\[i64\]\]}}}{reg:1}{}" = 1 : i64
# CHECK-SAME:   target_gate = "MultipleFullArgs{angles1:{{\[\[f64\]\]}},angles2:{{\[\[f64,f64\]\]}}}{reg1:1,reg2:2}{}[4444]"
# CHECK: "qref.operator"
# CHECK-SAME: op_name = "SingleParam"
# CHECK: "qref.operator"
# CHECK-SAME: op_name = "SingleParam"
# CHECK: "qref.operator"
# CHECK-SAME: op_name = "NoParams"
test_from_multiple_full_args_op()


def test_multiple_rules():
    """
    Test when multiple rules are registered on an op.
    """

    def rule1_resource_fn(reg):
        return {
            SingleParam(x=Float, reg=Wire[1]): 1,
        }

    @qp.register_resources(rule1_resource_fn)
    def rule1(reg):
        SingleParam(x=0.1, reg=[reg])

    def rule2_resource_fn(reg):
        return {
            CompilableData("a", "b", "thing", Wire[3]): 1,
        }

    @qp.register_resources(rule2_resource_fn)
    def rule2(reg):
        CompilableData(b="b", thing="thing", a="a", wires=[reg, reg + 1, reg + 2])

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule1)
        qp.add_decomps(NoParams, rule2)
        result = compile_decomposition_rules_wrapper(
            "NoParams", "NoParams{}{reg:1}{}", {}, {"reg": 1}, {}
        )
        print(result)


# CHECK: func.func private @"rule1_NoParams{}{reg:1}{}"
# CHECK-SAME:   resources = {operations = {"SingleParam{x:{{\[\[f64\]\]}}}{reg:1}{}" = 1 : i64}}
# CHECK-SAME:   target_gate = "NoParams{}{reg:1}{}"
# CHECK: func.func private @"rule2_NoParams{}{reg:1}{}"
# CHECK-SAME:   resources = {operations = {"CompilableData{}{wires:3}{a:a,b:b,thing:thing}" = 1 : i64}}
# CHECK-SAME:   target_gate = "NoParams{}{reg:1}{}"
test_multiple_rules()


def test_for_loop():
    """Test when the rule body has a for loop."""

    class LayerRX(qp.core.Operator2):
        dynamic_argnames = ("angle",)

        def __init__(self, angle, wires):
            super().__init__(angle, wires)

    class TestRX(qp.core.Operator2):
        dynamic_argnames = ("theta",)
        wires_argnames = ("wires",)
        arg_specs = {"theta": Float, "wires": Wire[1]}

        def __init__(self, theta, wires):
            super().__init__(theta, wires)

    @qp.register_resources(lambda angle, wires: {TestRX(Float, Wire[1]): len(wires)})
    def test_rule(angle, wires):
        @qp.for_loop(len(wires))
        def l(i):
            TestRX(angle, wires[i])

        l()  # pylint: disable=no-value-for-parameter

    with qp.decomposition.local_decomps():
        qp.add_decomps(LayerRX, test_rule)

        result = compile_decomposition_rules_wrapper(
            "LayerRX", "TestID", {"angle": ["f64"]}, {"wires": 3}, {}
        )
        print(result)


# CHECK: func.func private @test_rule_TestID
# CHECK-SAME:   resources = {operations = {"TestRX{0:[f64]}{wires:1}{}" = 3 : i64}}
# CHECK-SAME:   target_gate = "TestID"
# CHECK-DAG: arith.constant 0 : index
# CHECK-DAG: arith.constant 3 : index
# CHECK-DAG: arith.constant 1 : index
# CHECK: scf.for
test_for_loop()


def test_if():
    """Test when the rule body has an if op."""

    class TestOp(Operator2):
        dynamic_argnames = ("flag",)
        wire_argnames = ("wires",)

        def __init__(self, flag, wires):
            super().__init__(flag, wires)

    @qp.register_resources(lambda flag, wires: {NoParams(Wire[1]): 1})
    def if_decomp(flag, wires):
        qp.cond(flag, NoParams)(wires)

    qp.add_decomps(TestOp, if_decomp)

    print(compile_decomposition_rules_wrapper("TestOp", "IfID", {"flag": ["i1"]}, {"wires": 1}, {}))


# CHECK: if_decomp
# CHECK: scf.if
test_if()


def test_while_loop():
    """Test when the rule body has a while loop."""

    class WhileOp(Operator2):
        dynamic_argnames = ("angle",)
        wire_argnames = ("wires",)

        def __init__(self, angle, wires):
            super().__init__(angle, wires)

    @qp.register_resources(lambda angle, wires: {SingleParamCustomOp(Float[1], Wire[1]): 1})
    def while_decomp(angle, wires):
        @qp.while_loop(lambda angle: angle < jnp.pi)
        def while_body(angle):
            return angle + 1.5

        angle = while_body(angle)

        SingleParamCustomOp(angle, wires)

    qp.add_decomps(WhileOp, while_decomp)

    print(
        compile_decomposition_rules_wrapper(
            "WhileOp", "whileID", {"angle": ["f64"]}, {"wires": 1}, {}
        )
    )


# CHECK: while_decomp
# CHECK: scf.while
test_while_loop()


def test_custom_op_numbered_args():
    """
    Check that custom op parameter names don't interfere with decomposition rule lower.

    When calling from the compiler, custom op dynamic args are assigned numbers since they are
    unnamed. Since these numbers don't correspond to names in the frontend, we need to ensure
    that they are parsed into args and used correctly to call the decomposition rules.
    """
    print(
        compile_decomposition_rules_wrapper(
            "RZ",
            "RZ{0:[f64]}{wires:2}{}",
            {"0": ["f64"]},
            {"wires": 1},
            {},
            is_custom_op=True,
        )
    )


test_custom_op_numbered_args()


def test_rule_with_helper_functions():
    """
    Check that decomp rules that call helper functions are correctly inlined.
    """

    @qp.capture.subroutine
    def my_helper():
        return 4.2

    @qp.register_resources({MultiParams(reg=Wire[1], a=Float, b=Float, c=Float): 1})
    def rule(reg):
        MultiParams(reg=reg, a=my_helper(), b=0.2, c=0.3)

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule)
        result = compile_decomposition_rules_wrapper(
            "NoParams", "NoParams{}{reg:1}{}", {}, {"reg": 1}, {}
        )
        print(result)


# CHECK: func.func private @"rule_NoParams{}{reg:1}{}"
# CHECK-SAME:   resources = {operations = {
# CHECK-SAME:   "MultiParams{a:{{\[\[f64\]\]}},b:{{\[\[f64\]\]}},c:{{\[\[f64\]\]}}}{reg:1}{}" = 1 : i64
# CHECK-SAME:   target_gate = "NoParams{}{reg:1}{}"
# CHECK: stablehlo.constant dense<4.200000e+00> : tensor<f64>
# CHECK-NOT: call
# CHECK-NOT: my_helper
test_rule_with_helper_functions()
