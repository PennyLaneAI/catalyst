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
    HybridNoOpArg,
    HybridOpArg,
    HybridWires,
    MultiParams,
    MultipleRegisters,
    NoParams,
    NoParamsCustomOp,
    SingleParam,
    StaticData,
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
            result = compile_decomposition_rules_wrapper(
                "CompilableData",
                "CompilableData{}{wires:1}{a:1,b:2,thing:3}",
                {},
                {"wires": 1},
                {"a": 1, "b": 2, "thing": 3},
            )
            print(result)

    # CHECK: func.func private @"rule_CompilableData{}{wires:1}{a:1,b:2,thing:3}"
    # CHECK-SAME:   resources = {operations = {"SingleParam{x:[f64]}{reg:1}{}" = 1 : i64}
    # CHECK-SAME:   target_gate = "CompilableData{}{wires:1}{a:1,b:2,thing:3}"
    # CHECK: stablehlo.constant dense<1.000000e-01> : tensor<f64>
    test_compilable_data()


test_compile_decomposition_rules_wrapper_entry_point()
