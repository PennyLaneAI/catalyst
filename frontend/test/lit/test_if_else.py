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

import pennylane as qml

from catalyst import cond, measure, qjit


# CHECK-NOT: Verification failed
# CHECK-LABEL: public @jit_circuit
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit(n: int):
    # CHECK-DAG:   [[c5:%[a-zA-Z0-9_]+]] = stablehlo.constant dense<5> : tensor<i64>
    # CHECK:       [[b_t:%[a-zA-Z0-9_]+]] = stablehlo.compare  LE, %arg0, [[c5]], SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    # CHECK-DAG:   [[qreg_0:%[a-zA-Z0-9_]+]] = quantum.alloc
    # CHECK:       [[b:%[a-zA-Z0-9_]+]] = tensor.extract [[b_t]]
    @cond(n <= 5)
    # CHECK:       [[qreg_2:%.+]]:2 = scf.if [[b]]
    def cond_fn():
        # CHECK-DAG:   [[q0:%[a-zA-Z0-9_]+]] = quantum.extract
        # CHECK-DAG:   [[q1:%[a-zA-Z0-9_]+]] = quantum.custom "PauliX"() [[q0]]
        # pylint: disable=line-too-long
        # CHECK-DAG:   [[qreg_1:%[a-zA-Z0-9_]+]] = quantum.insert [[qreg_0]][ {{[%a-zA-Z0-9_]+}}], [[q1]]
        # CHECK:       scf.yield %arg0, [[qreg_1]]
        qml.PauliX(wires=0)
        return n

    @cond_fn.otherwise
    def otherwise():
        # CHECK:       [[r0:%[a-zA-Z0-9_a-z]+]] = stablehlo.multiply %arg0, %arg0
        # CHECK:       [[r1:%[a-zA-Z0-9_a-z]+]] = stablehlo.multiply [[r0]], %arg0
        # CHECK:       scf.yield [[r1]], [[qreg_0]]
        return n**3

    out = cond_fn()
    # CHECK:       [[qreg_3:%.+]] = quantum.insert [[qreg_2]]#1[ 0], %out_qubit : !quantum.reg, !quantum.bit
    # CHECK:       quantum.dealloc [[qreg_3]]
    # CHECK:       return
    return out, measure(wires=0)


print(circuit.mlir)

# -----


# CHECK-LABEL: public @jit_circuit_single_gate
@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit_single_gate(n: int):
    # pylint: disable=line-too-long
    # CHECK-DAG:   [[c5:%[a-zA-Z0-9_]+]] = stablehlo.constant dense<5> : tensor<i64>
    # CHECK-DAG:   [[c6:%[a-zA-Z0-9_]+]] = stablehlo.constant dense<6> : tensor<i64>
    # CHECK-DAG:   [[c7:%[a-zA-Z0-9_]+]] = stablehlo.constant dense<7> : tensor<i64>
    # CHECK-DAG:   [[c8:%[a-zA-Z0-9_]+]] = stablehlo.constant dense<8> : tensor<i64>
    # CHECK-DAG:   [[c9:%[a-zA-Z0-9_]+]] = stablehlo.constant dense<9> : tensor<i64>
    # CHECK-DAG:       [[b_t5:%[a-zA-Z0-9_]+]] = stablehlo.compare  LE, %arg0, [[c5]], SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    # CHECK-DAG:       [[b_t6:%[a-zA-Z0-9_]+]] = stablehlo.compare  LE, %arg0, [[c6]], SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    # CHECK-DAG:       [[b_t7:%[a-zA-Z0-9_]+]] = stablehlo.compare  LE, %arg0, [[c7]], SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    # CHECK-DAG:       [[b_t8:%[a-zA-Z0-9_]+]] = stablehlo.compare  LE, %arg0, [[c8]], SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    # CHECK-DAG:       [[b_t9:%[a-zA-Z0-9_]+]] = stablehlo.compare  LE, %arg0, [[c9]], SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    # CHECK-DAG:   [[qreg_0:%[a-zA-Z0-9_]+]] = quantum.alloc

    # CHECK:       [[b5:%[a-zA-Z0-9_]+]] = tensor.extract [[b_t5]]
    # CHECK:       [[qreg_out:%.+]] = scf.if [[b5]]
    # CHECK-DAG:   [[q0:%[a-zA-Z0-9_]+]] = quantum.extract [[qreg_0]]
    # CHECK-DAG:   [[q1:%[a-zA-Z0-9_]+]] = quantum.custom "PauliX"() [[q0]]
    # CHECK-DAG:   [[qreg_1:%[a-zA-Z0-9_]+]] = quantum.insert [[qreg_0]][ {{[%a-zA-Z0-9_]+}}], [[q1]]
    # CHECK:       scf.yield [[qreg_1]]

    # CHECK:       else
    # CHECK-DAG:   [[q2:%[a-zA-Z0-9_]+]] = quantum.extract [[qreg_0]]
    # CHECK-DAG:   [[q3:%[a-zA-Z0-9_]+]] = quantum.custom "Hadamard"() [[q2]]
    # CHECK-DAG:   [[qreg_2:%[a-zA-Z0-9_]+]] = quantum.insert [[qreg_0]][ {{[%a-zA-Z0-9_]+}}], [[q3]]
    # CHECK:       scf.yield [[qreg_2]]
    qml.cond(n <= 5, qml.PauliX, qml.Hadamard)(wires=0)

    # CHECK:       [[b6:%[a-zA-Z0-9_]+]] = tensor.extract [[b_t6]]
    # CHECK:       [[qreg_out1:%.+]] = scf.if [[b6]]
    # CHECK-DAG:   [[q4:%[a-zA-Z0-9_]+]] = quantum.extract [[qreg_out]]
    # CHECK-DAG:   [[q5:%[a-zA-Z0-9_]+]] = quantum.custom "RX"( [3.140000e+00]) [[q4]]
    # CHECK-DAG:   [[qreg_3:%[a-zA-Z0-9_]+]] = quantum.insert [[qreg_out]][ {{[%a-zA-Z0-9_]+}}], [[q5]]
    # CHECK:       scf.yield [[qreg_3]]
    # CHECK:       else
    # CHECK:       scf.yield [[qreg_out]]

    qml.cond(n <= 6, qml.RX)(3.14, wires=0)

    # CHECK:       [[b7:%[a-zA-Z0-9_]+]] = tensor.extract [[b_t7]]
    # CHECK:       [[qreg_out2:%.+]] = scf.if [[b7]]
    # CHECK-DAG:      [[q7:%[a-zA-Z0-9_]+]] = quantum.extract [[qreg_out1]]
    # CHECK-DAG:      [[q8:%[a-zA-Z0-9_]+]] = quantum.custom "Hadamard"() [[q7]]
    # pylint: disable=line-too-long
    # CHECK-DAG:      [[qreg_4:%[a-zA-Z0-9_]+]] = quantum.insert [[qreg_out1]][ {{[%a-zA-Z0-9_]+}}], [[q8]]
    # CHECK:          scf.yield [[qreg_4]]
    # CHECK:       else {
    # CHECK:          [[b8:%[a-zA-Z0-9_]+]] = tensor.extract [[b_t8]]
    # CHECK:          [[qreg_out3:%.+]] = scf.if [[b8]]
    # CHECK-DAG:         [[q9:%[a-zA-Z0-9_]+]] = quantum.extract [[qreg_out1]]
    # CHECK-DAG:         [[q10:%[a-zA-Z0-9_]+]] = quantum.custom "PauliY"() [[q9]]
    # CHECK-DAG:         [[qreg_5:%[a-zA-Z0-9_]+]] = quantum.insert [[qreg_out1]][ {{[%a-zA-Z0-9_]+}}], [[q10]]
    # CHECK:             scf.yield [[qreg_5]]
    # CHECK:          else {
    # CHECK:             [[b9:%[a-zA-Z0-9_]+]] = tensor.extract [[b_t9]]
    # CHECK:             [[qreg_out4:%.+]] = scf.if [[b9]]
    # CHECK-DAG:            [[q11:%[a-zA-Z0-9_]+]] = quantum.extract [[qreg_out1]]
    # CHECK-DAG:            [[q12:%[a-zA-Z0-9_]+]] = quantum.custom "PauliZ"() [[q11]]
    # CHECK-DAG:            [[qreg_6:%[a-zA-Z0-9_]+]] = quantum.insert [[qreg_out1]][ {{[%a-zA-Z0-9_]+}}], [[q12]]
    # CHECK:                scf.yield [[qreg_6]]
    # CHECK:                else {
    # CHECK-DAG:            [[q13:%[a-zA-Z0-9_]+]] = quantum.extract [[qreg_out1]]
    # CHECK-DAG:            [[q14:%[a-zA-Z0-9_]+]] = quantum.custom "PauliX"() [[q13]]
    # CHECK-DAG:            [[qreg_7:%[a-zA-Z0-9_]+]] = quantum.insert [[qreg_out1]][ {{[%a-zA-Z0-9_]+}}], [[q14]]
    # CHECK:                scf.yield [[qreg_7]]
    # CHECK:             scf.yield [[qreg_out4]]
    # CHECK:          scf.yield [[qreg_out3]]
    qml.cond(
        n <= 7,
        qml.Hadamard,
        qml.PauliX,
        (
            (n <= 8, qml.PauliY),
            (n <= 9, qml.PauliZ),
        ),
    )(wires=0)

    # CHECK:       [[qreg_ret:%.+]] = quantum.extract [[qreg_out2]][ 0]
    # CHECK:       [[qobs:%.+]] = quantum.compbasis [[qreg_ret]] : !quantum.obs
    # CHECK:       [[ret:%.+]] = quantum.probs [[qobs]]
    # CHECK:       return [[ret]]
    return qml.probs()


print(circuit_single_gate.mlir)


# -----


# CHECK-LABEL: test_convert_element_type
@qjit
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
@qjit
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
