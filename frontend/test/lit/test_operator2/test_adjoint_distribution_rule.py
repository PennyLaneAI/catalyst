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
Test the "distribution" pathway for CQRs.
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable = line-too-long,unused-argument,missing-function-docstring

import pennylane as qp
from operator2_dummy_gates import NoParams, SingleParam
from pennylane.typing import Float, Wire


def test_distribution_rule_synthesized_from_base():
    """``NoParams`` has only a base rule; lowering it synthesizes an Adjoint(NoParams)
    distribution rule with an adjoint region body and adjointed resources."""

    def base_resource_fn(reg):
        return {SingleParam(x=Float, reg=Wire[2]): 2}

    @qp.register_resources(base_resource_fn)
    def base_rule(reg):
        SingleParam(x=0.1, reg=reg[0:2])
        SingleParam(x=0.2, reg=reg[0:2])

    with qp.decomposition.local_decomps():
        qp.add_decomps(NoParams, base_rule)

        @qp.qjit(capture=True, target="mlir")
        @qp.qnode(qp.device("null.qubit", wires=3))
        def c():
            NoParams(reg=[0, 1])
            return qp.state()

        print(c.mlir)
    # CHECK-DAG: func.func private @"__builtin_base_rule_NoParams{}{reg:2}{}"
    # CHECK-DAG:   target_gate = "NoParams{}{reg:2}{}"
    # CHECK-DAG: func.func private @"__builtin_base_rule_Adjoint(NoParams{}{reg:2}{})"
    # CHECK-DAG:   resources = {operations = {"Adjoint(SingleParam{x:{{\[\[f64\]\]}}}{reg:2}{})" = 2 : i64}
    # CHECK-DAG:   target_gate = "Adjoint(NoParams{}{reg:2}{})"
    # CHECK: qref.adjoint {


test_distribution_rule_synthesized_from_base()
