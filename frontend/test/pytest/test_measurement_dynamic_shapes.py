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
import pennylane as qp
import pytest

import catalyst


@pytest.mark.usefixtures("use_both_frontend")
def test_dynamic_sample(capfd):
    """Test that a `sample` program with dynamic shots can be executed correctly and doesn't recompile."""

    @catalyst.qjit
    def workflow_dyn_sample(shots):
        print("compiling...")
        device = qp.device("lightning.qubit", wires=1)

        @partial(qp.set_shots, shots=shots)
        @qp.qnode(device)
        def circuit():
            qp.RX(1.5, 0)
            return qp.sample()

        return circuit()

    # Test with different shot numbers - should only compile once
    res = workflow_dyn_sample(10)
    assert len(res) == 10
    res = workflow_dyn_sample(37)
    assert len(res) == 37
    res = workflow_dyn_sample(5)
    assert len(res) == 5

    # Check that compilation only happened once
    out, _ = capfd.readouterr()
    assert out.count("compiling...") == 1


@pytest.mark.usefixtures("use_both_frontend")
def test_dynamic_counts(capfd):
    """Test that a `counts` program with dynamic shots can be executed correctly and doesn't recompile."""

    @catalyst.qjit
    def workflow_dyn_counts(shots):
        print("compiling...")
        device = qp.device("lightning.qubit", wires=1)

        @partial(qp.set_shots, shots=shots)
        @qp.qnode(device)
        def circuit():
            qp.RX(1.5, 0)
            return qp.counts(all_outcomes=True)

        return circuit()

    # Test with different shot numbers - should only compile once
    res = workflow_dyn_counts(10)
    assert res[1][0] + res[1][1] == 10

    res = workflow_dyn_counts(4000)
    assert res[1][0] + res[1][1] == 4000

    res = workflow_dyn_counts(100)
    assert res[1][0] + res[1][1] == 100

    # Check that compilation only happened once
    out, _ = capfd.readouterr()
    assert out.count("compiling...") == 1


@pytest.mark.parametrize("readout", [qp.expval, qp.var])
def test_dynamic_wires_scalar_readouts(readout, backend, capfd):
    """
    Test that a circuit with dynamic number of wires can be executed correctly.

    As a unit test for allocating a dynamic number of wires, we use measurements
    whose shape do not depend on the number of wires, i.e. expval and var
    """

    def ref(num_qubits):
        print("compiling...")
        dev = qp.device(backend, wires=num_qubits)

        @qp.qnode(dev)
        def circ():
            @catalyst.for_loop(0, num_qubits, 1)
            def loop_0(i):
                qp.RY(2.2, wires=i)

            loop_0()
            qp.RX(1.23, wires=num_qubits - 1)
            return readout(qp.Z(wires=num_qubits - 1))

        return circ()

    cat = catalyst.qjit(ref)

    assert np.allclose(ref(10), cat(10))
    assert np.allclose(ref(4), cat(4))
    out, _ = capfd.readouterr()
    assert out.count("compiling...") == 3


@pytest.mark.parametrize("readout", [qp.probs])
def test_dynamic_wires_statebased_with_wires(readout, backend, capfd):
    """
    Test that a circuit with dynamic number of wires can be executed correctly
    with state based measurements with wires specified.

    Note that qp.state() cannot have wires.
    """

    def ref(num_qubits):
        print("compiling...")
        dev = qp.device(backend, wires=num_qubits)

        @qp.qnode(dev)
        def circ():
            @catalyst.for_loop(0, num_qubits, 1)
            def loop_0(i):
                qp.RY(2.2, wires=i)

            loop_0()
            qp.RX(1.23, wires=num_qubits - 1)
            qp.RZ(3.45, wires=0)
            qp.CNOT(wires=[num_qubits - 2, 1])
            return readout(wires=[0, num_qubits - 2])

        return circ()

    cat = catalyst.qjit(ref)

    assert np.allclose(ref(10), cat(10))
    assert np.allclose(ref(4), cat(4))
    out, _ = capfd.readouterr()
    assert out.count("compiling...") == 3


@pytest.mark.parametrize("readout", [qp.probs, qp.state])
def test_dynamic_wires_statebased_without_wires(readout, backend, capfd):
    """
    Test that a circuit with dynamic number of wires can be executed correctly
    with state based measurements without wires specified.
    """

    def ref(num_qubits):
        print("compiling...")
        dev = qp.device(backend, wires=num_qubits)

        @qp.qnode(dev)
        def circ(x):
            @catalyst.for_loop(0, num_qubits, 1)
            def loop_0(i):
                qp.RY(2.2, wires=i)

            loop_0()
            qp.cond(x == 42, qp.RZ)(3.45, wires=0)
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
        dev = qp.device(backend, wires=num_qubits)

        @qp.set_shots(shots)
        @qp.qnode(dev)
        def circ():
            @catalyst.for_loop(0, num_qubits, 1)
            def loop_0(i):
                qp.RY(0.0, wires=i)

            loop_0()
            qp.RX(0.0, wires=num_qubits - 1)
            return qp.sample(wires=[0, num_qubits - 1])

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
        dev = qp.device(backend, wires=num_qubits)

        @qp.set_shots(shots)
        @qp.qnode(dev)
        def circ():
            @catalyst.for_loop(0, num_qubits, 1)
            def loop_0(i):
                qp.RY(0.0, wires=i)

            loop_0()
            qp.RX(0.0, wires=num_qubits - 1)
            return qp.sample()

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
        dev = qp.device(backend, wires=num_qubits)

        @qp.set_shots(1000)
        @qp.qnode(dev)
        def circ():
            qp.RX(0.0, wires=num_qubits - 1)
            return qp.counts(wires=[0, num_qubits - 1])

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
        dev = qp.device(backend, wires=num_qubits)

        @qp.set_shots(1000)
        @qp.qnode(dev)
        def circ():
            qp.RX(0.0, wires=num_qubits - 1)
            return qp.counts()

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
        dev = qp.device(backend, wires=num_qubits)

        @qp.qnode(dev)
        def circ():
            return qp.expval(qp.Z(wires=num_qubits - 1))

        return circ()

    with pytest.raises(
        AttributeError, match="Number of wires on the device should be a scalar integer."
    ):
        func(wires)


def test_dynamic_shots_and_wires(capfd):
    """Test that a circuit with both dynamic shots and dynamic wires works correctly with qp.sample."""

    @catalyst.qjit
    def workflow_dynamic_shots_and_wires(num_shots, num_wires):
        print("compiling...")
        device = qp.device("lightning.qubit", wires=num_wires)

        @partial(qp.set_shots, shots=num_shots)
        @qp.qnode(device)
        def circuit():
            # Apply Hadamard to all wires
            @catalyst.for_loop(0, num_wires, 1)
            def apply_hadamards(i):
                qp.Hadamard(i)

            apply_hadamards()

            # Apply some entangling gates if we have multiple wires
            @catalyst.cond(num_wires > 1)
            def add_entanglement():
                @catalyst.for_loop(0, num_wires - 1, 1)
                def apply_cnots(i):
                    qp.CNOT([i, i + 1])

                apply_cnots()

            add_entanglement()

            return qp.sample()

        return circuit()

    # Test with different combinations of shots and wires - should only compile once
    result_1 = workflow_dynamic_shots_and_wires(10, 2)
    assert result_1.shape == (10, 2)

    result_2 = workflow_dynamic_shots_and_wires(20, 3)
    assert result_2.shape == (20, 3)

    result_3 = workflow_dynamic_shots_and_wires(5, 1)
    assert result_3.shape == (5, 1)

    result_4 = workflow_dynamic_shots_and_wires(15, 4)
    assert result_4.shape == (15, 4)

    # Check that compilation only happened once
    out, _ = capfd.readouterr()
    assert out.count("compiling...") == 1


if __name__ == "__main__":
    pytest.main(["-x", __file__])
