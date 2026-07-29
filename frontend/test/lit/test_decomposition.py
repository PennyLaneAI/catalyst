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
"""Tests for graph based decomposition system."""

# pylint: disable = missing-function-docstring, line-too-long

import pennylane as qp
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
    StaticDataMultiReg,
)
from pennylane.typing import Float, Wire

from catalyst.decomposition.decomposition_rules import (
    compile_decomposition_rules_wrapper,
)


def test_compile_decomposition_rules_wrapper_entry_point():
    """
    Unit tests for the compile_decomposition_rules_wrapper() entry point function.
    """

    def test_single_rule():
        def rule_resource_fn(reg):
            return {
                SingleParam(x=Float, reg=Wire[2]): 1,
                CompilableData("a", "b", "thing", Wire[1]): 1,
            }

        @qp.register_resources(rule_resource_fn)
        def rule(reg):
            SingleParam(x=0.1, reg=reg)
            CompilableData(a="a", b="b", thing="thing", wires=reg[0])

        with qp.decomposition.local_decomps():
            qp.add_decomps(NoParams, rule)
            result = compile_decomposition_rules_wrapper(
                "NoParams", "NoParams{}{reg:2}{}", {}, {"reg": 2}, {}
            )
            print(result)

    # CHECK: func.func private @"rule_NoParams{}{reg:2}{}"
    # CHECK-SAME:   resources = {operations = {"CompilableData{}{wires:1}{a:a,b:b,thing:thing}" = 1 : i64,
    # CHECK-SAME:     "SingleParam{x:[f64]}{reg:2}{}" = 1 : i64}}
    # CHECK-SAME:   target_gate = "NoParams{}{reg:2}{}"
    test_single_rule()

    def test_multiple_rules():
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
    # CHECK-SAME:   resources = {operations = {"SingleParam{x:[f64]}{reg:1}{}" = 1 : i64}}
    # CHECK-SAME:   target_gate = "NoParams{}{reg:1}{}"
    # CHECK: func.func private @"rule2_NoParams{}{reg:1}{}"
    # CHECK-SAME:   resources = {operations = {"CompilableData{}{wires:3}{a:a,b:b,thing:thing}" = 1 : i64}}
    # CHECK-SAME:   target_gate = "NoParams{}{reg:1}{}"
    test_multiple_rules()

    def test_single_rule_custom_op():
        def rule_resource_fn(reg):
            return {NoParamsCustomOp(wires=Wire[2]): 1}

        @qp.register_resources(rule_resource_fn)
        def rule(reg):
            NoParamsCustomOp(wires=reg)

        with qp.decomposition.local_decomps():
            qp.add_decomps(NoParams, rule)
            result = compile_decomposition_rules_wrapper(
                "NoParams", "NoParams{}{reg:2}{}", {}, {"reg": 2}, {}
            )
            print(result)

    # CHECK: func.func private @"rule_NoParams{}{reg:2}{}"
    # CHECK-SAME:   resources = {operations = {"NoParamsCustomOp{}{wires:2}{}" = 1 : i64}
    # CHECK-SAME:   target_gate = "NoParams{}{reg:2}{}"
    test_single_rule_custom_op()

    def test_multi_params():
        def rule_resource_fn(reg, a, b, c):
            return {NoParamsCustomOp(wires=Wire[2]): 1}

        @qp.register_resources(rule_resource_fn)
        def rule(reg, a, b, c):
            NoParamsCustomOp(wires=reg)

        with qp.decomposition.local_decomps():
            qp.add_decomps(MultiParams, rule)
            result = compile_decomposition_rules_wrapper(
                "MultiParams",
                "MultiParams{a:[f64],b:[i32,f64],c:[[i32],[f64]]}{reg:2}{}",
                {"b": ["i32", "f64"], "c": [["i32"], ["f64"]], "a": ["f64"]},
                {"reg": 2},
                {},
            )
            print(result)

    # CHECK: func.func private @"rule_MultiParams{a:[f64],b:[i32,f64],c:{{\[\[}}i32],[f64{{\]\]}}}{reg:2}{}"
    # CHECK-SAME:   resources = {operations = {"NoParamsCustomOp{}{wires:2}{}" = 1 : i64}
    # CHECK-SAME:   target_gate = "MultiParams{a:[f64],b:[i32,f64],c:{{\[\[}}i32],[f64{{\]\]}}}{reg:2}{}"
    test_multi_params()

    def test_multi_wires():
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
    # CHECK-SAME:   resources = {operations = {"NoParamsCustomOp{}{wires:2}{}" = 1 : i64, "NoParamsCustomOp{}{wires:3}{}" = 1 : i64}
    # CHECK-SAME:   target_gate = "MultipleRegisters{}{reg1:2,reg2:3}{}"
    test_multi_wires()

    def test_compilable_data():
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
    # CHECK-SAME:   resources = {operations = {"SingleParam{x:[f64]}{reg:1}{}" = 1 : i64}
    # CHECK-SAME:   target_gate = "CompilableData{}{wires:1}{a:1,b:2,thing:3}"
    # CHECK: stablehlo.constant dense<1.000000e-01> : tensor<f64>
    #
    # CHECK: func.func private @"rule_CompilableData{}{wires:1}{a:10,b:2,thing:3}"
    # CHECK-SAME:   resources = {operations = {"SingleParam{x:[f64]}{reg:1}{}" = 1 : i64}
    # CHECK-SAME:   target_gate = "CompilableData{}{wires:1}{a:10,b:2,thing:3}"
    # CHECK: stablehlo.constant dense<1.100000e+00> : tensor<f64>
    test_compilable_data()

    def test_static_data():
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
    # CHECK-SAME:   resources = {operations = {"SingleParam{x:[f64]}{reg:1}{}" = 1 : i64}
    # CHECK-SAME:   target_gate = "StaticData{}{reg:1}{}[1234]"
    # CHECK: stablehlo.constant dense<1.000000e-01> : tensor<f64>
    #
    # CHECK: func.func private @"rule_StaticData{}{reg:1}{}[4321]"
    # CHECK-SAME:   resources = {operations = {"SingleParam{x:[f64]}{reg:1}{}" = 2 : i64}
    # CHECK-SAME:   target_gate = "StaticData{}{reg:1}{}[4321]"
    # CHECK: stablehlo.constant dense<1.100000e+00> : tensor<f64>
    # CHECK: stablehlo.constant dense<2.200000e+00> : tensor<f64>
    test_static_data()

    def test_decompose_to_compilable_data():
        def rule_resource_fn(reg):
            return {
                CompilableData("a", "b", "thing", Wire[1]): 1,
                CompilableData("aa", "bb", "stuff", Wire[1]): 1,
            }

        @qp.register_resources(rule_resource_fn)
        def rule(reg):
            CompilableData(a="a", b="b", thing="thing", wires=reg[0])
            CompilableData(a="aa", b="bb", thing="stuff", wires=reg[0])

        with qp.decomposition.local_decomps():
            qp.add_decomps(NoParams, rule)
            result = compile_decomposition_rules_wrapper(
                "NoParams", "NoParams{}{reg:2}{}", {}, {"reg": 2}, {}
            )
            print(result)

    # CHECK: func.func private @"rule_NoParams{}{reg:2}{}"
    # CHECK-SAME:   resources = {operations = {"CompilableData{}{wires:1}{a:a,b:b,thing:thing}" = 1 : i64,
    # CHECK-SAME:     "CompilableData{}{wires:1}{a:aa,b:bb,thing:stuff}" = 1 : i64}}
    # CHECK-SAME:   target_gate = "NoParams{}{reg:2}{}"
    test_decompose_to_compilable_data()

    def test_decompose_to_static_data():
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
    # CHECK-DAG: "StaticData{}{reg:1}{}[[[uid_1:[-0-9]+]]]" = 1
    # CHECK-DAG: "StaticData{}{reg:1}{}[[[uid_2:[-0-9]+]]]" = 2
    # CHECK-DAG:   target_gate = "NoParams{}{reg:2}{}"
    # CHECK: "qref.operator"({{%.+}}) {UID = [[uid_1]]
    # CHECK: "qref.operator"({{%.+}}) {UID = [[uid_2]]
    # CHECK: "qref.operator"({{%.+}}) {UID = [[uid_2]]
    test_decompose_to_static_data()

    def test_decompose_to_hybrid_wires():
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
    # CHECK-DAG: "HybridWires{}{}{}[[[uid_1:[-0-9]+]]]" = 1
    # CHECK-DAG: "HybridWires{}{}{}[[[uid_2:[-0-9]+]]]" = 2
    # CHECK-DAG:   target_gate = "NoParams{}{reg:3}{}"
    # CHECK: "qref.operator"({{%.+}}) {UID = [[uid_2]]
    # CHECK: "qref.operator"({{%.+}}) {UID = [[uid_2]]
    # CHECK: "qref.operator"({{%.+}}) {UID = [[uid_1]]
    test_decompose_to_hybrid_wires()

    def test_decompose_to_hybrid_op():
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
    # CHECK-DAG: "HybridOpArg{angle:[f64]}{cwires:1}{}[[[uid_1:[-0-9]+]]]" = 1
    # CHECK-DAG: "HybridOpArg{angle:[f64]}{cwires:1}{}[[[uid_2:[-0-9]+]]]" = 2
    # CHECK-DAG:   target_gate = "NoParams{}{reg:3}{}"
    # CHECK: "qref.operator"({{%.+}}) {UID = [[uid_2]]
    # CHECK: "qref.operator"({{%.+}}) {UID = [[uid_2]]
    # CHECK: "qref.operator"({{%.+}}) {UID = [[uid_1]]
    test_decompose_to_hybrid_op()

    def test_decompose_to_hybrid_op_nested():
        def rule_resource_fn(reg):
            return {
                HybridOpArg(
                    angle=Float,
                    op=HybridOpArg(
                        angle=Float,
                        op=StaticDataMultiReg(
                            label="hello", reg=Wire[1], reg2=Wire[2], theta=Float
                        ),
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
    # CHECK-DAG: "HybridOpArg{angle:[f64]}{cwires:1}{}[[[uid_1:[-0-9]+]]]" = 1
    # CHECK-DAG:   target_gate = "NoParams{}{reg:3}{}"
    # CHECK: "qref.operator"({{%.+}}) {UID = [[uid_1]]
    test_decompose_to_hybrid_op_nested()


test_compile_decomposition_rules_wrapper_entry_point()
