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

"""End-to-end tests for one-shot mcm transform in MLIR"""

import numpy as np
import pennylane as qml
import pytest

from catalyst import qjit


def test_mlir_one_shot_pass_expval(backend):
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with expval
    """

    @qjit(capture=True, seed=38)
    @qml.transform(pass_name="one-shot-mcm")
    @qml.qnode(qml.device(backend, wires=2), shots=1000)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.expval(qml.X(0))

    res = circuit()
    assert res.dtype == "float64"
    assert res.shape == ()
    assert np.allclose(res, 1.0, atol=0.01, rtol=0.01)


def test_mlir_one_shot_pass_expval_mcm(backend):
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with expval
    on a mid circuit measurement
    """

    @qjit(capture=True, seed=38)
    @qml.transform(pass_name="one-shot-mcm")
    @qml.qnode(qml.device(backend, wires=2), shots=1000)
    def circuit():
        qml.Hadamard(wires=0)
        m = qml.measure(0)
        return qml.expval(m)

    res = circuit()
    assert res.dtype == "float64"
    assert res.shape == ()
    assert np.allclose(res, 0.5, atol=0.01, rtol=0.01)


def test_mlir_one_shot_pass_probs(backend):
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with probs
    """

    @qjit(capture=True, seed=12345)
    @qml.transform(pass_name="one-shot-mcm")
    @qml.qnode(qml.device(backend, wires=2), shots=1000)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.probs()  # only has probabilities in |00> and |10>

    res = circuit()
    assert res.dtype == "float64"
    assert res.shape == (4,)
    assert np.allclose(res, [0.5, 0, 0.5, 0], atol=0.01, rtol=0.01)


def test_mlir_one_shot_pass_probs_mcm(backend):
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with probs
    on a mid circuit measurement
    """

    @qjit(capture=True, seed=12345)
    @qml.transform(pass_name="one-shot-mcm")
    @qml.qnode(qml.device(backend, wires=2), shots=1000)
    def circuit():
        qml.Hadamard(wires=0)
        m0 = qml.measure(0)
        m1 = qml.measure(1)
        return qml.probs(op=[m0, m1])

    res = circuit()
    assert res.dtype == "float64"
    assert res.shape == (4,)
    assert np.allclose(res, [0.5, 0, 0.5, 0], atol=0.01, rtol=0.01)


def test_mlir_one_shot_pass_var_mcm(backend):
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with variance
    on a mid circuit measurement
    """

    @qjit(capture=True, seed=38)
    @qml.transform(pass_name="one-shot-mcm")
    @qml.qnode(qml.device(backend, wires=2), shots=1000)
    def circuit():
        qml.Hadamard(wires=0)
        m_0 = qml.measure(0)
        m_1 = qml.measure(1)
        return qml.var(m_0), qml.var(m_1)

    res = circuit()
    assert np.allclose(res[0], 0.25, atol=0.01, rtol=0.01)
    assert np.allclose(res[1], 0, atol=0.01, rtol=0.01)


def test_mlir_one_shot_pass_sample(backend):
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with sample
    """

    @qjit(capture=True, seed=12345)
    @qml.transform(pass_name="one-shot-mcm")
    @qml.qnode(qml.device(backend, wires=2), shots=1000)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.sample()

    res = circuit()
    assert res.dtype == "int64"
    assert res.shape == (1000, 2)
    for sample in res:
        assert sample[1] == 0
    wire0_sum = res[:, 0].sum()
    assert np.allclose(wire0_sum / 1000, 0.5, atol=0.01, rtol=0.01)


def test_mlir_one_shot_pass_sample_mcm(backend):
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with sample
    on a mid circuit measurement
    """

    @qjit(capture=True, seed=12345)
    @qml.transform(pass_name="one-shot-mcm")
    @qml.qnode(qml.device(backend, wires=2), shots=1000)
    def circuit():
        qml.Hadamard(wires=0)
        m0 = qml.measure(0)
        m1 = qml.measure(1)
        return qml.sample([m0, m1])

    res = circuit()
    assert res.dtype == "int64"
    assert res.shape == (1000, 2)
    for sample in res:
        assert sample[1] == 0
    wire0_sum = res[:, 0].sum()
    assert np.allclose(wire0_sum / 1000, 0.5, atol=0.01, rtol=0.01)


def test_mlir_one_shot_pass_counts(backend):
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with counts
    """

    @qjit(capture=True, seed=12345)
    @qml.transform(pass_name="one-shot-mcm")
    @qml.qnode(qml.device(backend, wires=2), shots=1000)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.counts()

    res = circuit()
    eigs, counts = res
    assert eigs.shape == (4,)
    assert np.allclose(eigs, [0, 1, 2, 3])
    assert counts.shape == (4,)
    assert np.allclose(counts, [500, 0, 500, 0], rtol=0.01)


def test_mlir_one_shot_pass_counts_mcm(backend):
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with counts
    on MCMs.
    """

    @qjit(capture=True, seed=12345)
    @qml.transform(pass_name="one-shot-mcm")
    @qml.qnode(qml.device(backend, wires=2), shots=1000)
    def circuit():
        qml.Hadamard(wires=0)
        m_0 = qml.measure(0)
        m_1 = qml.measure(1)
        return qml.counts([m_0, m_1])

    res = circuit()
    eigs, counts = res
    assert eigs.shape == (4,)
    assert np.allclose(eigs, [0, 1, 2, 3])
    assert counts.shape == (4,)
    assert np.allclose(counts, [500, 0, 500, 0], atol=10)


def test_mlir_one_shot_pass_multiple_MPs(backend):
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with
    multiple MPs
    """

    @qjit(capture=True, seed=123456)
    @qml.transform(pass_name="one-shot-mcm")
    @qml.qnode(qml.device(backend, wires=2), shots=1000)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.sample(), qml.counts(), qml.expval(qml.X(0)), qml.probs()

    res = circuit()
    samples, eigs_and_counts, expval, probs = res
    eigens, counts = eigs_and_counts

    assert samples.shape == (1000, 2)
    for sample in samples:
        assert sample[1] == 0

    assert eigens.shape == (4,)
    assert np.allclose(eigens, [0, 1, 2, 3])
    assert counts.shape == (4,)
    assert np.allclose(counts, [500, 0, 500, 0], atol=10)

    assert np.allclose(expval, 1.0)

    assert np.allclose(probs, [0.5, 0, 0.5, 0], atol=0.01, rtol=0.01)


def test_mlir_one_shot_pass_multiple_MPs_mcms(backend):
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with
    multiple MPs on MCMs
    """

    @qjit(capture=True, seed=12345)
    @qml.transform(pass_name="one-shot-mcm")
    @qml.qnode(qml.device(backend, wires=2), shots=1000)
    def circuit():
        qml.Hadamard(wires=0)
        m_0 = qml.measure(0)
        m_1 = qml.measure(1)
        return (
            qml.sample([m_0, m_1]),
            qml.expval(m_0),
            qml.probs(op=[m_0, m_1]),
            qml.counts([m_0]),
        )

    res = circuit()
    samples, expval, probs, eigs_and_counts = res
    eigens, counts = eigs_and_counts

    assert samples.shape == (1000, 2)
    for sample in samples:
        assert sample[1] == 0

    assert np.allclose(expval, 0.5, atol=0.01, rtol=0.01)

    assert np.allclose(probs, [0.5, 0, 0.5, 0], atol=0.01, rtol=0.01)

    assert eigens.shape == (2,)
    assert np.allclose(eigens, [0, 1])
    assert counts.shape == (2,)
    assert np.allclose(counts, [500, 500], atol=10)


def test_mlir_one_shot_pass_dynamic_shots(backend):
    """
    Test that the mlir implementation of --one-shot-mcm pass can be used from frontend with a
    dynamic number of shots
    """

    @qjit(capture=True, seed=12345)
    def workflow(shots):
        @qml.transform(pass_name="one-shot-mcm")
        @qml.qnode(qml.device(backend, wires=2), shots=shots)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.counts()

        return circuit()

    res = workflow(1000)
    eigs, counts = res
    assert eigs.shape == (4,)
    assert np.allclose(eigs, [0, 1, 2, 3])
    assert counts.shape == (4,)
    assert sum(counts) == 1000

    res = workflow(500)
    eigs, counts = res
    assert eigs.shape == (4,)
    assert np.allclose(eigs, [0, 1, 2, 3])
    assert counts.shape == (4,)
    assert sum(counts) == 500


if __name__ == "__main__":
    pytest.main(["-x", __file__])
