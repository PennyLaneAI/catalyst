# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RUN: %PYTHON %s | FileCheck --implicit-check-not convert_element_type %s

# pylint: disable=missing-function-docstring

import pennylane as qp

from catalyst import cond, qjit


# CHECK-NOT: Verification failed
# CHECK-LABEL: public @jit_circuit
@qjit(target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=1))
def circuit(n: int):
    # CHECK-LABEL: public @circuit(
    # CHECK: [[five:%.+]] = stablehlo.constant dense<5> : tensor<i64>
    # CHECK: [[qreg:%.+]] = qref.alloc( 1)
    # CHECK: [[condition:%.+]] = stablehlo.compare LE, %arg0, [[five]], SIGNED
    # CHECK: [[flag:%.+]] = tensor.extract [[condition]]
    # CHECK: [[result:%.+]] = scf.if [[flag]] -> (tensor<i64>)
    # CHECK: [[qubit:%.+]] = qref.get [[qreg]][ 0]
    # CHECK: qref.custom "PauliX"() [[qubit]]
    # CHECK: scf.yield %arg0
    # CHECK: } else {
    # CHECK: [[square:%.+]] = stablehlo.multiply %arg0, %arg0
    # CHECK: [[cube:%.+]] = stablehlo.multiply [[square]], %arg0
    # CHECK: scf.yield [[cube]]
    # CHECK: [[angle_tensor:%.+]] = stablehlo.convert [[result]]
    # CHECK: [[angle:%.+]] = tensor.extract [[angle_tensor]]
    # CHECK: qref.custom "RY"([[angle]])
    # CHECK: qref.dealloc [[qreg]]
    # CHECK: return
    @qp.cond(n <= 5)
    def cond_fn():
        # pylint: disable=line-too-long
        qp.PauliX(wires=0)
        return n

    @cond_fn.otherwise
    def otherwise():
        return n**3

    out = cond_fn()
    qp.RY(out, wires=0)
    return qp.expval(qp.Z(0))


print(circuit.mlir)

# -----


# CHECK-LABEL: public @jit_circuit_single_gate
@qjit(target="mlir")
@qp.qnode(qp.device("lightning.qubit", wires=1))
def circuit_single_gate(n: int):
    # CHECK-LABEL: public @circuit_single_gate(
    # CHECK-DAG: [[five:%.+]] = stablehlo.constant dense<5> : tensor<i64>
    # CHECK-DAG: [[six:%.+]] = stablehlo.constant dense<6> : tensor<i64>
    # CHECK-DAG: [[seven:%.+]] = stablehlo.constant dense<7> : tensor<i64>
    # CHECK-DAG: [[eight:%.+]] = stablehlo.constant dense<8> : tensor<i64>
    # CHECK-DAG: [[nine:%.+]] = stablehlo.constant dense<9> : tensor<i64>
    # CHECK: [[qreg:%.+]] = qref.alloc( 1)
    # CHECK: stablehlo.compare LE, %arg0, [[five]], SIGNED
    # CHECK: scf.if
    # CHECK: qref.custom "PauliX"
    # CHECK: } else {
    # CHECK: qref.custom "Hadamard"
    # CHECK: stablehlo.compare LE, %arg0, [[six]], SIGNED
    # CHECK: scf.if
    # CHECK: qref.custom "RX"
    # CHECK: stablehlo.compare LE, %arg0, [[seven]], SIGNED
    # CHECK: stablehlo.compare LE, %arg0, [[eight]], SIGNED
    # CHECK: stablehlo.compare LE, %arg0, [[nine]], SIGNED
    # CHECK: scf.if
    # CHECK: qref.custom "Hadamard"
    # CHECK: } else {
    # CHECK: scf.if
    # CHECK: qref.custom "PauliY"
    # CHECK: } else {
    # CHECK: scf.if
    # CHECK: qref.custom "PauliZ"
    # CHECK: } else {
    # CHECK: qref.custom "PauliX"
    # CHECK: [[obs:%.+]] = qref.compbasis(qreg [[qreg]] : !qref.reg<1>)
    # CHECK: [[result:%.+]] = quantum.probs [[obs]]
    # CHECK: return [[result]]
    # pylint: disable=line-too-long
    qp.cond(n <= 5, qp.PauliX, qp.Hadamard)(wires=0)

    qp.cond(n <= 6, qp.RX)(3.14, wires=0)
    # pylint: disable=line-too-long
    qp.cond(
        n <= 7,
        qp.Hadamard,
        qp.PauliX,
        (
            (n <= 8, qp.PauliY),
            (n <= 9, qp.PauliZ),
        ),
    )(wires=0)
    return qp.probs()


print(circuit_single_gate.mlir)


# -----


# CHECK-LABEL: test_convert_element_type
# old_frontend: checks Catalyst conditional branch type promotion.
@qjit(capture=False)
def test_convert_element_type(i: int, f: float):
    """Test the presence of convert_element_type JAX primitive when the type conversion is
    required."""

    # CHECK: cond[
    @cond(i <= 3)
    def cond_fn():
        # CHECK: convert_element_type
        return i

    @cond_fn.otherwise
    def otherwise():
        # CHECK: add {{[a-z]+}} 2
        return f + 2

    # CHECK: ]

    return cond_fn()


print("test_convert_element_type")
print(test_convert_element_type.jaxpr)


# -----


# CHECK-LABEL: test_no_convert_element_type
# old_frontend: checks Catalyst conditional branch type promotion.
@qjit(capture=False)
def test_no_convert_element_type(i: int):
    """Test the absense of convert_element_type JAX primitive when no type conversion is required"""

    # CHECK: cond[
    @cond(i <= 3)
    def cond_fn():
        # CHECK: add {{[a-z]+}} 1
        return i + 1

    @cond_fn.otherwise
    def otherwise():
        # CHECK: add {{[a-z]+}} 2
        return i + 2

    # CHECK: ]

    return cond_fn()


print("test_no_convert_element_type")
print(test_no_convert_element_type.jaxpr)


# -----


# A single `estimated_probability` hint is attached to the emitted `scf.if`.
# CHECK-LABEL: public @jit_cond_estimated_probability
# old_frontend: control-flow hints are specific to the Catalyst API.
@qjit(capture=False, target="mlir")
def cond_estimated_probability(n: int):
    # CHECK:       scf.if
    # CHECK:       catalyst.estimated_probability = 5.000000e-01
    @cond(n <= 5, estimated_probability=0.5)
    def branch():
        return n

    @branch.otherwise
    def otherwise():
        return n * 2

    return branch()


print(cond_estimated_probability.mlir)


# -----


# The if/elif chain lowers to nested `scf.if`s, each carrying the *conditional* probability
# that its branch is taken given no earlier branch was. The user-facing unconditional hints
# (0.5, 0.125) therefore become (0.5, 0.125 / (1 - 0.5) = 0.25) on the two `scf.if`s.
# CHECK-LABEL: public @jit_cond_estimated_probability_chain
# old_frontend: control-flow hints are specific to the Catalyst API.
@qjit(capture=False, target="mlir")
def cond_estimated_probability_chain(n: int):
    # CHECK-DAG:   catalyst.estimated_probability = 5.000000e-01
    # CHECK-DAG:   catalyst.estimated_probability = 2.500000e-01
    @cond(n <= 5, estimated_probability=0.5)
    def branch():
        return n

    @branch.else_if(n <= 8, estimated_probability=0.125)
    def elif_branch():
        return n + 1

    @branch.otherwise
    def otherwise():
        return n * 2

    return branch()


print(cond_estimated_probability_chain.mlir)
