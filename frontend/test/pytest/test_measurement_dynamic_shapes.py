# Copyright 2025 Xanadu Quantum Technologies Inc.

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
This file contains tests for measurement primitives when the return shape is dynamic.
"""

# pylint: disable=line-too-long

from functools import partial

import numpy as np
import pennylane as qml
import pytest

import catalyst


def test_dynamic_sample():
    """Test that a `sample` program with dynamic shots can be executed correctly."""

    @catalyst.qjit
    def workflow_dyn_sample(shots):
        device = qml.device("lightning.qubit", wires=1)

        @partial(qml.set_shots, shots=shots)
        @qml.qnode(device)
        def circuit():
            qml.RX(1.5, 0)
            return qml.sample()

        return circuit()

    res = workflow_dyn_sample(10)
    assert len(res) == 10
    res = workflow_dyn_sample(37)
    assert len(res) == 37

    workflow_dyn_sample.workspace.cleanup()


def test_dynamic_counts():
    """Test that a `counts` program with dynamic shots can be executed correctly."""

    @catalyst.qjit
    def workflow_dyn_counts(shots):
        device = qml.device("lightning.qubit", wires=1)

        @partial(qml.set_shots, shots=shots)
        @qml.qnode(device)
        def circuit():
            qml.RX(1.5, 0)
            return qml.counts()

        return circuit()

    res = workflow_dyn_counts(10)
    assert res[1][0] + res[1][1] == 10

    res = workflow_dyn_counts(4000)
    assert res[1][0] + res[1][1] == 4000

    workflow_dyn_counts.workspace.cleanup()


@pytest.mark.parametrize("readout", [qml.expval, qml.var])
def test_dynamic_wires_scalar_readouts(readout, backend, capfd):
    """
    Test that a circuit with dynamic number of wires can be executed correctly.

    As a unit test for allocating a dynamic number of wires, we use measurements
    whose shape do not depend on the number of wires, i.e. expval and var
    """

    def ref(num_qubits):
        print("compiling...")
        dev = qml.device(backend, wires=num_qubits)

        @qml.qnode(dev)
        def circ():
            @catalyst.for_loop(0, num_qubits, 1)
            def loop_0(i):
                qml.RY(2.2, wires=i)

            loop_0()
            qml.RX(1.23, wires=num_qubits - 1)
            return readout(qml.Z(wires=num_qubits - 1))

        return circ()

    cat = catalyst.qjit(ref)

    assert np.allclose(ref(10), cat(10))
    assert np.allclose(ref(4), cat(4))
    out, _ = capfd.readouterr()
    assert out.count("compiling...") == 3


@pytest.mark.parametrize("readout", [qml.probs])
def test_dynamic_wires_statebased_with_wires(readout, backend, capfd):
    """
    Test that a circuit with dynamic number of wires can be executed correctly
    with state based measurements with wires specified.

    Note that qml.state() cannot have wires.
    """

    def ref(num_qubits):
        print("compiling...")
        dev = qml.device(backend, wires=num_qubits)

        @qml.qnode(dev)
        def circ():
            @catalyst.for_loop(0, num_qubits, 1)
            def loop_0(i):
                qml.RY(2.2, wires=i)

            loop_0()
            qml.RX(1.23, wires=num_qubits - 1)
            qml.RZ(3.45, wires=0)
            qml.CNOT(wires=[num_qubits - 2, 1])
            return readout(wires=[0, num_qubits - 2])

        return circ()

    cat = catalyst.qjit(ref)

    assert np.allclose(ref(10), cat(10))
    assert np.allclose(ref(4), cat(4))
    out, _ = capfd.readouterr()
    assert out.count("compiling...") == 3


@pytest.mark.parametrize("readout", [qml.probs, qml.state])
def test_dynamic_wires_statebased_without_wires(readout, backend, capfd):
    """
    Test that a circuit with dynamic number of wires can be executed correctly
    with state based measurements without wires specified.
    """

    def ref(num_qubits):
        print("compiling...")
        dev = qml.device(backend, wires=num_qubits)

        @qml.qnode(dev)
        def circ(x):
            @catalyst.for_loop(0, num_qubits, 1)
            def loop_0(i):
                qml.RY(2.2, wires=i)

            loop_0()
            qml.cond(x == 42, qml.RZ)(3.45, wires=0)
            return readout()

        return circ(42)

    cat = catalyst.qjit(ref)

    assert np.allclose(ref(10), cat(10))
    assert np.allclose(ref(4), cat(4))
    out, _ = capfd.readouterr()
    assert out.count("compiling...") == 3


@pytest.mark.parametrize("shots", [3, (3, 4, 5)])
def test_dynamic_wires_sample_with_wires(shots, backend, capfd):
    """
    Test that a circuit with dynamic number of wires can be executed correctly
    with sample measurements with wires specified.
    """

    def ref(num_qubits):
        print("compiling...")
        dev = qml.device(backend, wires=num_qubits, shots=shots)

        @qml.qnode(dev)
        def circ():
            @catalyst.for_loop(0, num_qubits, 1)
            def loop_0(i):
                qml.RY(0.0, wires=i)

            loop_0()
            qml.RX(0.0, wires=num_qubits - 1)
            return qml.sample(wires=[0, num_qubits - 1])

        return circ()

    cat = catalyst.qjit(ref)
    num_shots = 1 if isinstance(shots, int) else len(shots)
    for test_nqubits in (10, 4):
        expected = ref(test_nqubits)
        observed = cat(test_nqubits)
        assert all(np.allclose(expected[i], observed[i]) for i in range(num_shots))

    out, _ = capfd.readouterr()
    assert out.count("compiling...") == 3


@pytest.mark.parametrize("shots", [3, (3, 4, 5), (7,) * 3])
def test_dynamic_wires_sample_without_wires(shots, backend, capfd):
    """
    Test that a circuit with dynamic number of wires can be executed correctly
    with sample measurements without wires specified.
    """

    def ref(num_qubits):
        print("compiling...")
        dev = qml.device(backend, wires=num_qubits, shots=shots)

        @qml.qnode(dev)
        def circ():
            @catalyst.for_loop(0, num_qubits, 1)
            def loop_0(i):
                qml.RY(0.0, wires=i)

            loop_0()
            qml.RX(0.0, wires=num_qubits - 1)
            return qml.sample()

        return circ()

    cat = catalyst.qjit(ref)
    num_shots = 1 if isinstance(shots, int) else len(shots)
    for test_nqubits in (10, 4):
        expected = ref(test_nqubits)
        observed = cat(test_nqubits)
        assert all(np.allclose(expected[i], observed[i]) for i in range(num_shots))

    out, _ = capfd.readouterr()
    assert out.count("compiling...") == 3


def test_dynamic_wires_counts_with_wires(backend, capfd):
    """
    Test that a circuit with dynamic number of wires can be executed correctly
    with counts measurements with wires specified.

    Note that Catalyst does not support shot vectors with counts.
    """

    @catalyst.qjit
    def func(num_qubits):
        print("compiling...")
        dev = qml.device(backend, wires=num_qubits, shots=1000)

        @qml.qnode(dev)
        def circ():
            qml.RX(0.0, wires=num_qubits - 1)
            return qml.counts(wires=[0, num_qubits - 1])

        return circ()

    expected = [np.array([0, 1, 2, 3]), np.array([1000, 0, 0, 0])]
    for test_nqubits in (10, 4, 7):
        observed = func(test_nqubits)
        assert np.allclose(expected, observed)

    out, _ = capfd.readouterr()
    assert out.count("compiling...") == 1


def test_dynamic_wires_counts_without_wires(backend, capfd):
    """
    Test that a circuit with dynamic number of wires can be executed correctly
    with counts measurements without wires specified.

    Note that Catalyst does not support shot vectors with counts.
    """

    @catalyst.qjit
    def func(num_qubits):
        print("compiling...")
        dev = qml.device(backend, wires=num_qubits, shots=1000)

        @qml.qnode(dev)
        def circ():
            qml.RX(0.0, wires=num_qubits - 1)
            return qml.counts()

        return circ()

    for test_nqubits in (1, 2, 3):
        size = 2**test_nqubits
        expected_counts = np.zeros(size)
        expected_counts[0] = 1000
        expected = [np.arange(size), expected_counts]
        observed = func(test_nqubits)
        assert np.allclose(expected, observed)

    out, _ = capfd.readouterr()
    assert out.count("compiling...") == 1


@pytest.mark.parametrize("wires", [1.1, (1.1)])
def test_wrong_wires_argument(backend, wires):
    """
    Test that a circuit with a wrongly typed and shaped dynamic wire argument
    is correctly caught.
    """

    @catalyst.qjit
    def func(num_qubits):
        dev = qml.device(backend, wires=num_qubits)

        @qml.qnode(dev)
        def circ():
            return qml.expval(qml.Z(wires=num_qubits - 1))

        return circ()

    with pytest.raises(
        AttributeError, match="Number of wires on the device should be a scalar integer."
    ):
        func(wires)


def test_dynamic_shots_no_recompilation(capfd):
    """Test that dynamic shots work correctly and don't trigger recompilation for different shot values."""

    device = qml.device("lightning.qubit", wires=2)

    @catalyst.qjit
    def workflow_dynamic_shots(num_shots):
        print("compiling...")

        @partial(qml.set_shots, shots=num_shots)
        @qml.qnode(device)
        def circuit():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            return qml.sample()

        return circuit()

    # Test with different shot numbers - should only compile once
    result_3 = workflow_dynamic_shots(3)
    result_5 = workflow_dynamic_shots(5)
    result_10 = workflow_dynamic_shots(10)
    result_37 = workflow_dynamic_shots(37)

    # Verify results have correct shapes
    assert len(result_3) == 3
    assert result_3.shape == (3, 2)

    assert len(result_5) == 5
    assert result_5.shape == (5, 2)

    assert len(result_10) == 10
    assert result_10.shape == (10, 2)

    assert len(result_37) == 37
    assert result_37.shape == (37, 2)

    # Check that compilation only happened once
    out, _ = capfd.readouterr()
    assert out.count("compiling...") == 1

    workflow_dynamic_shots.workspace.cleanup()


def test_dynamic_shots_counts_no_recompilation(capfd):
    """Test that dynamic shots with counts work correctly and don't trigger recompilation."""

    device = qml.device("lightning.qubit", wires=1)

    @catalyst.qjit
    def workflow_dynamic_counts(num_shots):
        print("compiling...")

        @partial(qml.set_shots, shots=num_shots)
        @qml.qnode(device)
        def circuit():
            qml.RX(1.5, 0)
            return qml.counts()

        return circuit()

    # Test with different shot numbers
    result_10 = workflow_dynamic_counts(10)
    result_100 = workflow_dynamic_counts(100)
    result_1000 = workflow_dynamic_counts(1000)

    # Verify results have correct total counts
    assert result_10[1][0] + result_10[1][1] == 10
    assert result_100[1][0] + result_100[1][1] == 100
    assert result_1000[1][0] + result_1000[1][1] == 1000

    # Check that compilation only happened once
    out, _ = capfd.readouterr()
    assert out.count("compiling...") == 1

    workflow_dynamic_counts.workspace.cleanup()


if __name__ == "__main__":
    pytest.main(["-x", __file__])
