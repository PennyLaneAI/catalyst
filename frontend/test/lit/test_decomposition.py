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

import pennylane as qp
from operator2_dummy_gates import (
    CompilableData,
    GlobalPhase,
    HybridNoOpArg,
    HybridOpArg,
    HybridWires,
    MultiParams,
    MultiParamsCustom,
    MultipleRegisters,
    MultiRZ,
    NoParams,
    NoParamsCustomOp,
    PauliRot,
    PCPhase,
    QubitUnitary,
    SingleParam,
    SingleParamCustomOp,
    StaticData,
)
from pennylane.typing import Float, Wire

from catalyst.decomposition.decomposition_rules import (
    compile_decomposition_rules_wrapper,
)


def test_single_rule():
    def rule_resource_fn(reg):
        return {SingleParam(x=Float, reg=Wire[2]): 1, CompilableData("a", "b", "thing", Wire[1]): 1}

    @qp.register_resources(rule_resource_fn)
    def rule(reg):
        SingleParam(x=0.1, reg=reg)
        CompilableData(a="a", b="b", thing="thing", wires=reg[0])

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule)
        result = compile_decomposition_rules_wrapper(
            "NoParams", "NoParams[][2]{}", {}, {"reg": 2}, {}
        )
        print(result)


# CHECK: func.func private @"rule_NoParams[][2]{}"
# CHECK-SAME:   resources = {operations = {"CompilableData[][1]{a:a,b:b,thing:thing}" = 1 : i64, "SingleParam[f64][2]{}" = 1 : i64}}
# CHECK-SAME:   target_gate = "NoParams[][2]{}"
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
        CompilableData(a="a", b="b", thing="thing", wires=[reg, reg + 1, reg + 2])

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, rule1)
        qp.add_decomps(NoParams, rule2)
        result = compile_decomposition_rules_wrapper(
            "NoParams", "NoParams[][1]{}", {}, {"reg": 1}, {}
        )
        print(result)


# CHECK: func.func private @"rule1_NoParams[][1]{}"
# CHECK-SAME:   resources = {operations = {"SingleParam[f64][1]{}" = 1 : i64}}
# CHECK-SAME:   target_gate = "NoParams[][1]{}"
# CHECK: func.func private @"rule2_NoParams[][1]{}"
# CHECK-SAME:   resources = {operations = {"CompilableData[][3]{a:a,b:b,thing:thing}" = 1 : i64}}
# CHECK-SAME:   target_gate = "NoParams[][1]{}"
test_multiple_rules()
