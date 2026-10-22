# Copyright 2025 Haiqu, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IR tests for the REM feature: assert that mitigate_with_rem emits a
mitigation.rem op with the expected attribute and result types, plus the
JAX-traced calibrate / apply post-processing helpers with the right shape
on the call sites and on the private func.func definitions.
"""

# RUN: %PYTHON %s | FileCheck %s

import jax.numpy as jnp
import pennylane as qp

from catalyst import mitigate_with_rem, qjit

# pylint: disable=line-too-long


@qjit(target="mlir")
def rem_with_probs():
    """ProbsMP emits one f64 callee tensor plus two f64 calibration tensors."""
    dev = qp.device("lightning.qubit", wires=2)

    def circuit():
        qp.Hadamard(wires=0)
        qp.CNOT(wires=[0, 1])
        return qp.probs(wires=[0, 1])

    qnode = qp.set_shots(qp.QNode(circuit, dev), shots=200)
    return mitigate_with_rem(qnode)()


# CHECK-LABEL: func.func public @jit_rem_with_probs
# CHECK: mitigation.rem @{{[^(]+}}() runCalibration(true)
# CHECK-SAME: -> (tensor<4xf64>, tensor<4xf64>, tensor<4xf64>)
# Post-processing:
#   calibration consumes the two f64 calibration tensors and yields the (n_qubits, 2, 2) confusion stack.
#   apply consumes the f64 mitigatee probs and yields the mitigated 2^n probs vector.
# CHECK: call @rem_calibrate_probs({{.*}}) : (tensor<4xf64>, tensor<4xf64>) -> tensor<2x2x2xf64>
# CHECK: call @rem_apply_to_probs({{.*}}) : (tensor<4xf64>, tensor<2x2x2xf64>, tensor<2xi64>) -> tensor<4xf64>
# CHECK: func.func private @rem_calibrate_probs(%{{.*}}: tensor<4xf64>, %{{.*}}: tensor<4xf64>) -> tensor<2x2x2xf64>
# CHECK: func.func private @rem_apply_to_probs(%{{.*}}: tensor<4xf64>, %{{.*}}: tensor<2x2x2xf64>, %{{.*}}: tensor<2xi64>) -> tensor<4xf64>
print(rem_with_probs.mlir)


@qjit(target="mlir")
def rem_with_counts():
    """CountsMP emits a (i64, i64) callee pair plus two i64 calibration tensors."""
    dev = qp.device("lightning.qubit", wires=2)

    def circuit():
        qp.Hadamard(wires=0)
        qp.CNOT(wires=[0, 1])
        return qp.counts()

    qnode = qp.set_shots(qp.QNode(circuit, dev), shots=200)
    return mitigate_with_rem(qnode)()


# CHECK-LABEL: func.func public @jit_rem_with_counts
# CHECK: mitigation.rem @{{[^(]+}}() runCalibration(true)
# CHECK-SAME: -> (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>)
# Post-processing: counts path reuses the probs calibration after normalization,
# and apply returns f64 because the linear solve is float-valued.
# CHECK: call @rem_calibrate_counts({{.*}}) : (tensor<4xi64>, tensor<4xi64>) -> tensor<2x2x2xf64>
# CHECK: call @rem_apply_to_counts({{.*}}) : (tensor<4xi64>, tensor<2x2x2xf64>, tensor<2xi64>) -> tensor<4xf64>
# CHECK: func.func private @rem_calibrate_counts(%{{.*}}: tensor<4xi64>, %{{.*}}: tensor<4xi64>) -> tensor<2x2x2xf64>
# CHECK: func.func private @rem_apply_to_counts(%{{.*}}: tensor<4xi64>, %{{.*}}: tensor<2x2x2xf64>, %{{.*}}: tensor<2xi64>) -> tensor<4xf64>
print(rem_with_counts.mlir)


@qjit(target="mlir")
def rem_with_sample_histogram():
    """SampleMP, 2**n_qubits <= n_shots -> rem_apply_to_samples picks the full-histogram path (K = 2**n)."""
    dev = qp.device("lightning.qubit", wires=2)

    def circuit():
        qp.Hadamard(wires=0)
        qp.CNOT(wires=[0, 1])
        return qp.sample()

    qnode = qp.set_shots(qp.QNode(circuit, dev), shots=200)
    return mitigate_with_rem(qnode)()


# CHECK-LABEL: func.func public @jit_rem_with_sample_histogram
# CHECK: mitigation.rem @{{[^(]+}}() runCalibration(true)
# CHECK-SAME: -> (tensor<200x2xi64>, tensor<200x2xf64>, tensor<200x2xf64>)
# Post-processing, histogram path: calibrate sees (shots, qubits) i32 samples,
# apply returns ((K, n_qubits) bitstrings, (K,) counts) with K = 2**n_qubits = 4.
# CHECK: call @rem_calibrate_samples({{.*}}) : (tensor<200x2xi32>, tensor<200x2xi32>) -> tensor<2x2x2xf64>
# CHECK: call @rem_apply_to_samples({{.*}}) : (tensor<200x2xi32>, tensor<2x2x2xf64>, tensor<2xi64>) -> (tensor<4x2xi32>, tensor<4xf64>)
# CHECK: func.func private @rem_calibrate_samples(%{{.*}}: tensor<200x2xi32>, %{{.*}}: tensor<200x2xi32>) -> tensor<2x2x2xf64>
# CHECK: func.func private @rem_apply_to_samples(%{{.*}}: tensor<200x2xi32>, %{{.*}}: tensor<2x2x2xf64>, %{{.*}}: tensor<2xi64>) -> (tensor<4x2xi32>, tensor<4xf64>)
print(rem_with_sample_histogram.mlir)


@qjit(target="mlir")
def rem_with_sample_sort_rle():
    """SampleMP, 2**n_qubits > n_shots -> rem_apply_to_samples picks the sort-RLE path (K = n_shots)."""
    dev = qp.device("lightning.qubit", wires=4)

    def circuit():
        qp.Hadamard(wires=0)
        for k in range(1, 4):
            qp.CNOT(wires=[0, k])
        return qp.sample()

    qnode = qp.set_shots(qp.QNode(circuit, dev), shots=4)
    return mitigate_with_rem(qnode)()


# CHECK-LABEL: func.func public @jit_rem_with_sample_sort_rle
# CHECK: mitigation.rem @{{[^(]+}}() runCalibration(true)
# CHECK-SAME: -> (tensor<4x4xi64>, tensor<4x4xf64>, tensor<4x4xf64>)
# Post-processing, sort-RLE path: apply output is (K = n_shots = 4, n_qubits) and (K,).
# CHECK: call @rem_calibrate_samples({{.*}}) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x2x2xf64>
# CHECK: call @rem_apply_to_samples({{.*}}) : (tensor<4x4xi32>, tensor<4x2x2xf64>, tensor<4xi64>) -> (tensor<4x4xi32>, tensor<4xf64>)
# CHECK: func.func private @rem_calibrate_samples(%{{.*}}: tensor<4x4xi32>, %{{.*}}: tensor<4x4xi32>) -> tensor<4x2x2xf64>
# CHECK: func.func private @rem_apply_to_samples(%{{.*}}: tensor<4x4xi32>, %{{.*}}: tensor<4x2x2xf64>, %{{.*}}: tensor<4xi64>) -> (tensor<4x4xi32>, tensor<4xf64>)
print(rem_with_sample_sort_rle.mlir)


@qjit(target="mlir")
def rem_cached():
    """Passing precomputed calibration matrices runs the op with runCalibration(false)
    and emits rem_apply_to_probs but not rem_calibrate_probs."""
    dev = qp.device("lightning.qubit", wires=2)

    def circuit():
        qp.Hadamard(wires=0)
        qp.CNOT(wires=[0, 1])
        return qp.probs(wires=[0, 1])

    qnode = qp.set_shots(qp.QNode(circuit, dev), shots=200)
    cm = jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (2, 2, 2))
    return mitigate_with_rem(qnode, calibration_matrices=cm)()


# CHECK-LABEL: func.func public @jit_rem_cached
# The op's result tuple must shrink to only the callee's return, no zeros/ones slots.
# CHECK: mitigation.rem @{{[^(]+}}() runCalibration(false)
# CHECK-SAME: -> tensor<4xf64>
# CHECK: call @rem_apply_to_probs({{.*}}) : (tensor<4xf64>, tensor<2x2x2xf64>, tensor<2xi64>) -> tensor<4xf64>
# CHECK-NOT: call @rem_calibrate
# CHECK-NOT: func.func private @rem_calibrate
print(rem_cached.mlir)
