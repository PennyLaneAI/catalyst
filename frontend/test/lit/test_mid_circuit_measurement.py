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

# RUN: %PYTHON %s | FileCheck %s

import numpy as np
import pennylane as qml

from catalyst import measure, qjit
from catalyst.debug import get_compilation_stage
from catalyst.passes import merge_rotations


@qjit(target="mlir")
@qml.qnode(qml.device("lightning.qubit", wires=1))
def circuit(x: float):
    qml.RX(x, wires=0)
    # CHECK: {{%.+}}, {{%.+}} = quantum.measure {{%.+}}
    m = measure(wires=0)
    return m


print(circuit.mlir)


# -----


@qjit(static_argnums=0)
def test_one_shot_with_static_argnums(N):
    """
    Test static argnums is passed correctly to the one shot qnodes.
    """
    # CHECK: func.func public @jit_test_one_shot_with_static_argnums() -> tensor<1024xf64>
    # CHECK: {{%.+}} = call @one_shot_wrapper() : () -> tensor<1024xf64>

    # CHECK: func.func private @one_shot_wrapper() -> tensor<1024xf64>
    # CHECK-DAG: [[one:%.+]] = arith.constant 1 : index
    # CHECK-DAG: [[ten:%.+]] = arith.constant 10 : index
    # CHECK-DAG: [[zero:%.+]] = arith.constant 0 : index
    # CHECK: scf.for %arg0 = [[zero]] to [[ten]] step [[one]]
    # CHECK: {{%.+}} = catalyst.launch_kernel @module_circ::@circ() : () -> tensor<1024xf64>

    # CHECK: func.func public @circ() -> tensor<1024xf64>
    # CHECK: [[one:%.+]] = arith.constant 1 : i64
    # CHECK: quantum.device shots([[one]])
    # CHECK: quantum.alloc( 10)

    dev = qml.device("lightning.qubit", wires=N)

    @qml.set_shots(N)
    @qml.qnode(dev, mcm_method="one-shot")
    def circ():
        return qml.probs()

    return circ()


test_one_shot_with_static_argnums(10)
print(test_one_shot_with_static_argnums.mlir)


# -----


@qjit(target="mlir")
def test_one_shot_with_passes():
    """
    Test pass pipeline is passed correctly to the one shot qnodes.
    """
    # CHECK: func.func public @jit_test_one_shot_with_passes() -> tensor<10x1xi64>
    # CHECK: {{%.+}} = call @one_shot_wrapper() : () -> tensor<10x1xi64>

    # CHECK: func.func private @one_shot_wrapper() -> tensor<10x1xi64>
    # CHECK: scf.for
    # CHECK: {{%.+}} = catalyst.launch_kernel @module_circ::@circ() : () -> tensor<1x1xi64>

    # CHECK: module @module_circ
    # CHECK: transform.named_sequence @__transform_main
    # CHECK: transform.apply_registered_pass "merge-rotations"
    # CHECK: func.func public @circ() -> tensor<1x1xi64>

    dev = qml.device("lightning.qubit", wires=1)

    @merge_rotations
    @qml.set_shots(10)
    @qml.qnode(dev, mcm_method="one-shot")
    def circ():
        return qml.sample()

    return circ()


print(test_one_shot_with_passes.mlir)


# -----


qml.capture.enable()


@qjit(keep_intermediate=True)
@qml.transform(pass_name="one-shot-mcm")
@qml.qnode(qml.device("lightning.qubit", wires=2), shots=1000)
def test_mlir_one_shot_pass_probs():
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with probs
    """

    # CHECK: transform.apply_registered_pass "one-shot-mcm"
    qml.Hadamard(wires=0)
    return qml.probs()  # only has probablilities in |00> and |10>


print(test_mlir_one_shot_pass_probs.mlir)

# CHECK: func.func public @test_mlir_one_shot_pass_probs.quantum_kernel
# CHECK:    [[one:%.+]] = arith.constant 1 : i64
# CHECK:    quantum.device shots([[one]])
# CHECK:    Hadamard
# CHECK:    probs
# CHECK: func.func public @test_mlir_one_shot_pass_probs
# CHECK:    index.constant 1000
# CHECK:    scf.for
# CHECK:    func.call @test_mlir_one_shot_pass_probs.quantum_kernel
# CHECK:    stablehlo.add
# CHECK:    stablehlo.divide
print(get_compilation_stage(test_mlir_one_shot_pass_probs, "QuantumCompilationStage"))

res = test_mlir_one_shot_pass_probs()
assert res[1] == 0
assert res[3] == 0
assert np.allclose(sum(res), 1.0)

qml.capture.disable()


# -----


qml.capture.enable()


@qjit(keep_intermediate=True, seed=38)
@qml.transform(pass_name="one-shot-mcm")
@qml.qnode(qml.device("lightning.qubit", wires=2), shots=1000)
def test_mlir_one_shot_pass_probs_mcm():
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with probs
    on a mid circuit measurement
    """

    # CHECK: transform.apply_registered_pass "one-shot-mcm"
    qml.Hadamard(wires=0)
    m = qml.measure(0)
    return qml.probs(op=m)


print(test_mlir_one_shot_pass_probs_mcm.mlir)

# CHECK: func.func public @test_mlir_one_shot_pass_probs_mcm.quantum_kernel
# CHECK:    [[one:%.+]] = arith.constant 1 : i64
# CHECK:    quantum.device shots([[one]])
# CHECK:    Hadamard
# CHECK:    measure
# CHECK-NOT:   probs
# CHECK: func.func public @test_mlir_one_shot_pass_probs_mcm
# CHECK:    index.constant 1000
# CHECK:    scf.for
# CHECK:    func.call @test_mlir_one_shot_pass_probs_mcm.quantum_kernel
# CHECK:    stablehlo.add
# CHECK:    stablehlo.divide
print(get_compilation_stage(test_mlir_one_shot_pass_probs_mcm, "QuantumCompilationStage"))

res = test_mlir_one_shot_pass_probs_mcm()
assert res.dtype == "float64"
assert res.shape == (2,)
assert np.allclose(res, [0.5, 0.5], atol=0.01, rtol=0.01)

qml.capture.disable()


# -----


qml.capture.enable()


@qjit(keep_intermediate=True, seed=38)
@qml.transform(pass_name="one-shot-mcm")
@qml.qnode(qml.device("lightning.qubit", wires=2), shots=1000)
def test_mlir_one_shot_pass_expval_mcm():
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with expval
    on a mid circuit measurement
    """

    # CHECK: transform.apply_registered_pass "one-shot-mcm"
    qml.Hadamard(wires=0)
    m = qml.measure(0)
    return qml.expval(m)


print(test_mlir_one_shot_pass_expval_mcm.mlir)

# CHECK: func.func public @test_mlir_one_shot_pass_expval_mcm.quantum_kernel
# CHECK:    [[one:%.+]] = arith.constant 1 : i64
# CHECK:    quantum.device shots([[one]])
# CHECK:    Hadamard
# CHECK:    measure
# CHECK-NOT:   expval
# CHECK: func.func public @test_mlir_one_shot_pass_expval_mcm
# CHECK:    index.constant 1000
# CHECK:    scf.for
# CHECK:    func.call @test_mlir_one_shot_pass_expval_mcm.quantum_kernel
# CHECK:    stablehlo.add
# CHECK:    stablehlo.divide
print(get_compilation_stage(test_mlir_one_shot_pass_expval_mcm, "QuantumCompilationStage"))

res = test_mlir_one_shot_pass_expval_mcm()
assert res.dtype == "float64"
assert res.shape == ()
assert np.allclose(res, 0.5, atol=0.01, rtol=0.01)

qml.capture.disable()


# -----


qml.capture.enable()


@qjit(keep_intermediate=True)
@qml.transform(pass_name="one-shot-mcm")
@qml.qnode(qml.device("lightning.qubit", wires=2), shots=1000)
def test_mlir_one_shot_pass_sample():
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with sample
    """

    # CHECK: transform.apply_registered_pass "one-shot-mcm"
    qml.Hadamard(wires=0)
    return qml.sample()  # only has samples in |00> and |10>


print(test_mlir_one_shot_pass_sample.mlir)

# CHECK: func.func public @test_mlir_one_shot_pass_sample.quantum_kernel
# CHECK:    [[one:%.+]] = arith.constant 1 : i64
# CHECK:    quantum.device shots([[one]])
# CHECK:    Hadamard
# CHECK:    sample
# CHECK: func.func public @test_mlir_one_shot_pass_sample
# CHECK:    index.constant 1000
# CHECK:    scf.for
# CHECK:    func.call @test_mlir_one_shot_pass_sample.quantum_kernel
# CHECK:    tensor.insert_slice
print(get_compilation_stage(test_mlir_one_shot_pass_sample, "QuantumCompilationStage"))

res = test_mlir_one_shot_pass_sample()
assert res.shape == (1000, 2)
for sample in res:
    assert sample[1] == 0

qml.capture.disable()


# -----


qml.capture.enable()


@qjit(keep_intermediate=True, seed=38)
@qml.transform(pass_name="one-shot-mcm")
@qml.qnode(qml.device("lightning.qubit", wires=2), shots=1000)
def test_mlir_one_shot_pass_sample_mcm():
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with sample
    on a mid circuit measurement
    """

    # CHECK: transform.apply_registered_pass "one-shot-mcm"
    qml.Hadamard(wires=0)
    m = qml.measure(0)
    return qml.sample(m)


print(test_mlir_one_shot_pass_sample_mcm.mlir)

# CHECK: func.func public @test_mlir_one_shot_pass_sample_mcm.quantum_kernel
# CHECK:    [[one:%.+]] = arith.constant 1 : i64
# CHECK:    quantum.device shots([[one]])
# CHECK:    Hadamard
# CHECK:    measure
# CHECK-NOT:   sample
# CHECK: func.func public @test_mlir_one_shot_pass_sample_mcm
# CHECK:    index.constant 1000
# CHECK:    scf.for
# CHECK:    func.call @test_mlir_one_shot_pass_sample_mcm.quantum_kernel
# CHECK:    tensor.insert_slice
print(get_compilation_stage(test_mlir_one_shot_pass_sample_mcm, "QuantumCompilationStage"))

res = test_mlir_one_shot_pass_sample_mcm()
assert res.dtype == "int64"
assert res.shape == (1000, 1)
assert np.allclose(sum(res) / 1000, 0.5, atol=0.01, rtol=0.01)

qml.capture.disable()


# -----


qml.capture.enable()


@qjit(keep_intermediate=True)
@qml.transform(pass_name="one-shot-mcm")
@qml.qnode(qml.device("lightning.qubit", wires=2), shots=1000)
def test_mlir_one_shot_pass_counts():
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with counts
    """

    # CHECK: transform.apply_registered_pass "one-shot-mcm"
    qml.Hadamard(wires=0)
    return qml.counts()  # only has samples in |00> and |10>


print(test_mlir_one_shot_pass_counts.mlir)

# CHECK: func.func public @test_mlir_one_shot_pass_counts.quantum_kernel
# CHECK:    [[one:%.+]] = arith.constant 1 : i64
# CHECK:    quantum.device shots([[one]])
# CHECK:    Hadamard
# CHECK:    counts
# CHECK: func.func public @test_mlir_one_shot_pass_counts
# CHECK:    index.constant 1000
# CHECK:    scf.for
# CHECK:    func.call @test_mlir_one_shot_pass_counts.quantum_kernel
# CHECK:    stablehlo.add
print(get_compilation_stage(test_mlir_one_shot_pass_counts, "QuantumCompilationStage"))

res = test_mlir_one_shot_pass_counts()
eigs, counts = res
assert eigs.shape == (4,)
assert np.allclose(eigs, [0, 1, 2, 3])
assert counts.shape == (4,)
assert sum(counts) == 1000
assert counts[1] == 0
assert counts[3] == 0

qml.capture.disable()
