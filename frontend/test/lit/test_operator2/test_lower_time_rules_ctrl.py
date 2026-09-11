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
Test the ``@graph_decomposition`` decorator on controlled and controlled-adjoint Operator2 ops.
"""

# RUN: %PYTHON %s | FileCheck %s

# pylint: disable = line-too-long,unused-argument,missing-function-docstring

import numpy as np
import pennylane as qp

from catalyst import qjit
from catalyst.passes import graph_decomposition


def _s_to_phaseshift():
    """A custom rule for the migrated ``S`` gate: ``S -> PhaseShift(pi/2)``."""

    def resources(wires):
        return {qp.PhaseShift: 1}

    @qp.register_resources(resources)
    def s_to_ps(wires):
        qp.PhaseShift(np.pi / 2, wires=wires)

    return s_to_ps


_GATE_SET = {qp.PhaseShift, qp.GlobalPhase, qp.RZ, qp.CNOT}


# CHECK-LABEL: func.func public @controlled_op
@qjit(capture=True, target="mlir")
@graph_decomposition(gate_set=_GATE_SET, alt_decomps={qp.S: [_s_to_phaseshift()]})
@qp.qnode(qp.device("null.qubit", wires=2))
def controlled_op():
    # CHECK: qref.custom "S"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
    qp.ctrl(qp.S(0), control=[1])
    return qp.state()


# CHECK-DAG: target_gate = "S{}{wires:1}{}"
# CHECK-DAG: target_gate = "C(S){}{wires:1}{}"
print(controlled_op.mlir)


# CHECK-LABEL: func.func public @controlled_adjoint_op
@qjit(capture=True, target="mlir")
@graph_decomposition(gate_set=_GATE_SET, alt_decomps={qp.S: [_s_to_phaseshift()]})
@qp.qnode(qp.device("null.qubit", wires=2))
def controlled_adjoint_op():
    # CHECK: qref.custom "S"() %{{.*}} adj ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
    qp.ctrl(qp.adjoint(qp.S(0)), control=[1])
    return qp.state()


# CHECK-DAG: target_gate = "Adjoint(S){}{wires:1}{}"
# CHECK-DAG: target_gate = "C(Adjoint(S)){}{wires:1}{}"
print(controlled_adjoint_op.mlir)


# CHECK-LABEL: func.func public @controlled_op_fixed
@qjit(capture=True, target="mlir")
@graph_decomposition(gate_set=_GATE_SET, fixed_decomps={qp.S: _s_to_phaseshift()})
@qp.qnode(qp.device("null.qubit", wires=2))
def controlled_op_fixed():
    # CHECK: qref.custom "S"() %{{.*}} ctrls(%{{.*}}) ctrlvals(%{{.*}}) : !qref.bit ctrls !qref.bit
    qp.ctrl(qp.S(0), control=[1])
    return qp.state()


# CHECK-DAG: target_gate = "C(S){}{wires:1}{}"
print(controlled_op_fixed.mlir)
