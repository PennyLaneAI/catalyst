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
    # CHECK:       scf.if [[b]]
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
        # CHECK:       [[r1:%[a-zA-Z0-9_a-z]+]] = stablehlo.multiply %arg0, [[r0]]
        # CHECK:       scf.yield [[r1]], [[qreg_0]]
        return n**3

    out = cond_fn()
    # CHECK:       quantum.dealloc [[qreg_0]]
    # CHECK:       return
    return out, measure(wires=0)


print(circuit.mlir)

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
