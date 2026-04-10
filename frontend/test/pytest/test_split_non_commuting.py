# Copyright 2026 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Integration test for the split non-commuting pass.
"""

import numpy as np
import pennylane as qp
import pytest

from catalyst import qjit
from catalyst.utils.exceptions import CompileError


@pytest.mark.parametrize(
    "hamiltonian",
    [
        [qp.Z(0) + qp.X(1) + 2 * qp.Y(2), lambda term1, term2, term3: term1 + term2 + 2 * term3],
        [
            3 * qp.Z(0) + qp.X(1) + 2 * qp.Y(2),
            lambda term1, term2, term3: 3 * term1 + term2 + 2 * term3,
        ],
    ],
)
def test_split_non_commuting_integration(hamiltonian, capture_mode):
    """
    Test that split non-commuting pass produces the same results as

    """
    dev = qp.device("lightning.qubit", wires=3)
    hamiltonian_obs, post_process_fn = hamiltonian

    # Circuit with Hamiltonian observable
    # Expected: split into individual terms with coefficients
    @qjit(capture=capture_mode)
    @qp.transform(pass_name="split-non-commuting")
    @qp.qnode(dev)
    def circ1():
        qp.Rot(0.3, 0.5, 0.7, wires=0)
        qp.Rot(0.2, 0.4, 0.6, wires=1)
        qp.Rot(0.1, 0.8, 0.9, wires=2)
        return qp.expval(hamiltonian_obs), qp.expval(qp.Z(1))

    # Manual implementation: split into individual execution and compute weighted sum
    @qjit(capture=capture_mode)
    @qp.qnode(dev)
    def group0():
        qp.Rot(0.3, 0.5, 0.7, wires=0)
        qp.Rot(0.2, 0.4, 0.6, wires=1)
        qp.Rot(0.1, 0.8, 0.9, wires=2)
        return qp.expval(qp.Z(0))

    @qjit(capture=capture_mode)
    @qp.qnode(dev)
    def group1():
        qp.Rot(0.3, 0.5, 0.7, wires=0)
        qp.Rot(0.2, 0.4, 0.6, wires=1)
        qp.Rot(0.1, 0.8, 0.9, wires=2)
        return qp.expval(qp.X(1))

    @qjit(capture=capture_mode)
    @qp.qnode(dev)
    def group2():
        qp.Rot(0.3, 0.5, 0.7, wires=0)
        qp.Rot(0.2, 0.4, 0.6, wires=1)
        qp.Rot(0.1, 0.8, 0.9, wires=2)
        return qp.expval(qp.Y(2))

    @qjit(capture=capture_mode)
    @qp.qnode(dev)
    def group3():
        qp.Rot(0.3, 0.5, 0.7, wires=0)
        qp.Rot(0.2, 0.4, 0.6, wires=1)
        qp.Rot(0.1, 0.8, 0.9, wires=2)
        return qp.expval(qp.Z(1))

    def circ2():
        return group0(), group1(), group2(), group3()

    def post_processing():
        term1, term2, term3, term4 = circ2()
        # Compute weighted sum using the post-processing function
        return post_process_fn(term1, term2, term3), term4

    # Validate that the pass was applied
    assert "hamiltonian" in circ1.mlir
    assert "Hamiltonian" not in circ1.mlir_opt

    # # Compare results
    result1 = circ1()
    result2 = post_processing()

    assert np.allclose(result1, result2), f"Results don't match: {result1} vs {result2}"


def test_split_non_commuting_with_tensor_product(capture_mode):
    """
    Test split-to-single-terms with tensor product observables.
    """
    dev = qp.device("lightning.qubit", wires=3)

    @qjit(capture=capture_mode)
    @qp.transform(pass_name="split-non-commuting")
    @qp.qnode(dev)
    def circ1():
        qp.Rot(0.4, 0.3, 0.2, wires=0)
        qp.Rot(0.6, 0.5, 0.4, wires=1)
        qp.Rot(0.8, 0.7, 0.6, wires=2)
        return qp.expval(2 * (qp.Z(0) @ qp.X(1)) + 3 * qp.Y(2)), qp.expval(qp.Z(1))

    @qjit(capture=capture_mode)
    @qp.qnode(dev)
    def group0():
        qp.Rot(0.4, 0.3, 0.2, wires=0)
        qp.Rot(0.6, 0.5, 0.4, wires=1)
        qp.Rot(0.8, 0.7, 0.6, wires=2)
        return qp.expval(qp.Z(0) @ qp.X(1))

    @qjit(capture=capture_mode)
    @qp.qnode(dev)
    def group1():
        qp.Rot(0.4, 0.3, 0.2, wires=0)
        qp.Rot(0.6, 0.5, 0.4, wires=1)
        qp.Rot(0.8, 0.7, 0.6, wires=2)
        return qp.expval(qp.Y(2))

    @qjit(capture=capture_mode)
    @qp.qnode(dev)
    def group2():
        qp.Rot(0.4, 0.3, 0.2, wires=0)
        qp.Rot(0.6, 0.5, 0.4, wires=1)
        qp.Rot(0.8, 0.7, 0.6, wires=2)
        return qp.expval(qp.Z(1))

    def circ2():
        return group0(), group1(), group2()

    def post_processing():
        term1, term2, term3 = circ2()
        return 2 * term1 + 3 * term2, term3

    assert "hamiltonian" in circ1.mlir
    assert "Hamiltonian" not in circ1.mlir_opt

    result1 = circ1()
    result2 = post_processing()

    assert np.allclose(result1, result2), f"Results don't match: {result1} vs {result2}"


@pytest.mark.capture_todo
def test_split_non_commuting_with_Identity(capture_mode):
    """
    Test split-non-commuting with Identity observables.
    Identity observables are removed from the quantum circuit since their
    expectation value is always 1, and their coefficient is added in post-processing.
    """
    dev = qp.device("lightning.qubit", wires=3)

    @qjit(capture=capture_mode)
    @qp.transform(pass_name="split-non-commuting")
    @qp.qnode(dev)
    def circ1():
        qp.Rot(0.5, 0.3, 0.2, wires=0)
        qp.Rot(0.4, 0.6, 0.1, wires=1)
        return qp.expval(qp.Z(0) + 2 * qp.X(1) + 0.7 * qp.Identity(2))

    @qjit(capture=capture_mode)
    @qp.qnode(dev)
    def group0():
        qp.Rot(0.5, 0.3, 0.2, wires=0)
        qp.Rot(0.4, 0.6, 0.1, wires=1)
        return qp.expval(qp.Z(0))

    @qjit(capture=capture_mode)
    @qp.qnode(dev)
    def group1():
        qp.Rot(0.5, 0.3, 0.2, wires=0)
        qp.Rot(0.4, 0.6, 0.1, wires=1)
        return qp.expval(qp.X(1))

    def circ2():
        return group0(), group1()

    def post_processing():
        term1, term2 = circ2()
        return term1 + 2 * term2 + 0.7

    assert "hamiltonian" in circ1.mlir
    assert "Hamiltonian" not in circ1.mlir_opt

    result1 = circ1()
    result2 = post_processing()

    assert np.allclose(result1, result2), f"Results don't match: {result1} vs {result2}"


def test_lightning_execution_with_structure(capture_mode):
    """Test that the split non-commuting pass on lightning.qubit for a circuit with program
    structure is executable and returns results as expected."""
    dev = qp.device("lightning.qubit", wires=10)

    @qp.for_loop(0, 10, 1)
    def for_fn(i):
        qp.H(i)
        qp.S(i)
        qp.RZ(phi=0.1, wires=[i])

    @qp.while_loop(lambda i: i < 10)
    def while_fn(i):
        qp.H(i)
        qp.S(i)
        qp.RZ(phi=0.1, wires=[i])
        i = i + 1
        return i

    @qjit(capture=capture_mode)
    @qp.transform(pass_name="split-non-commuting")
    @qp.qnode(dev)
    def circuit():
        for_fn()  # pylint: disable=no-value-for-parameter
        while_fn(0)
        qp.CNOT(wires=[0, 1])
        return (
            qp.expval(qp.Z(wires=0)),
            qp.expval(qp.Y(wires=1)),
            qp.expval(qp.X(wires=0)),
        )

    res = circuit()

    @qjit(capture=capture_mode)
    @qp.qnode(dev)
    def circuit_ref():
        for_fn()  # pylint: disable=no-value-for-parameter
        while_fn(0)
        qp.CNOT(wires=[0, 1])
        return (
            qp.expval(qp.Z(wires=0)),
            qp.expval(qp.Y(wires=1)),
            qp.expval(qp.X(wires=0)),
        )

    res_ref = circuit_ref()
    assert res == res_ref


@pytest.mark.parametrize(
    "measurement",
    [
        lambda: qp.probs(wires=[0, 1]),
        lambda: qp.counts(wires=[0, 1]),
        lambda: qp.sample(wires=[0, 1]),
        lambda: qp.var(qp.Z(0)),
        qp.state,
    ],
)
def test_split_non_commuting_error_non_expval(measurement, capture_mode):
    """Test that an error is raised when a non-expval measurement is included in the return.

    The split-non-commuting pass internally runs split-to-single-terms, which only supports
    quantum.expval measurements. All other MeasurementProcess ops cause a compile error.
    """
    dev = qp.device("lightning.qubit", wires=2)

    with pytest.raises(CompileError, match="unsupported measurement operation"):

        @qjit(capture=capture_mode)
        @qp.transform(pass_name="split-non-commuting")
        @qp.set_shots(100 if measurement is not qp.state else None)
        @qp.qnode(dev)
        def circ():
            return measurement()

        circ()


class TestSplitNonCommutingWires:
    """Tests for the split-non-commuting pass with grouping_strategy='wires'."""

    snc_pass = qp.transform(pass_name="split-non-commuting")(grouping_strategy="wires")

    def test_non_overlapping_single_group(self, capture_mode):
        """Z(0), X(1), Y(2) on separate wires -> 1 group"""
        dev = qp.device("lightning.qubit", wires=3)

        @qp.qnode(dev)
        def circ():
            qp.RX(0.3, wires=0)
            qp.RY(0.5, wires=1)
            qp.RX(0.7, wires=2)
            return qp.expval(qp.Z(0)), qp.expval(qp.X(1)), qp.expval(qp.Y(2))

        circ_split = qjit(self.snc_pass(circ), capture=capture_mode)
        circ_ref = qjit(circ, capture=capture_mode)
        assert np.allclose(circ_split(), circ_ref())

    def test_overlapping_two_groups(self, capture_mode):
        """Z(0), X(1), Y(1): Z(0), X(1) in group 0, Y(1) in group 1."""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circ():
            qp.RX(0.4, wires=0)
            qp.RY(0.6, wires=1)
            qp.RZ(0.25, wires=1)
            return qp.expval(qp.Z(0)), qp.expval(qp.X(1)), qp.expval(qp.Y(1))

        circ_split = qjit(self.snc_pass(circ), capture=capture_mode)
        circ_ref = qjit(circ, capture=capture_mode)
        assert np.allclose(circ_split(), circ_ref())

    def test_tensor_overlap(self, capture_mode):
        """Z(0)@Z(1) overlaps X(0) on wire 0 -> 2 groups."""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circ():
            qp.RY(0.3, wires=0)
            qp.RY(0.5, wires=1)
            return qp.expval(qp.Z(0) @ qp.Z(1)), qp.expval(qp.X(0))

        circ_split = qjit(self.snc_pass(circ), capture=capture_mode)
        circ_ref = qjit(circ, capture=capture_mode)
        assert np.allclose(circ_split(), circ_ref())

    def test_hamiltonian_non_overlapping(self, capture_mode):
        """Hamiltonian Z(0)+X(1)+2*Y(2): Z(0), X(1), Y(2) on separate wires -> 1 group."""
        dev = qp.device("lightning.qubit", wires=3)

        @qp.qnode(dev)
        def circ():
            qp.RX(0.3, wires=0)
            qp.RY(0.5, wires=1)
            qp.RX(0.7, wires=2)
            return qp.expval(qp.Z(0) + qp.X(1) + 2 * qp.Y(2))

        circ_split = qjit(self.snc_pass(circ), capture=capture_mode)
        circ_ref = qjit(circ, capture=capture_mode)
        assert np.allclose(circ_split(), circ_ref())

    def test_hamiltonian_overlapping(self, capture_mode):
        """Hamiltonian 0.5*Z(0)+3*X(1)+Y(1): Z(0), X(1) in group 0, Y(1) in group 1."""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circ():
            qp.RX(0.4, wires=0)
            qp.RY(0.6, wires=1)
            qp.RZ(0.25, wires=1)
            return qp.expval(0.5 * qp.Z(0) + 3 * qp.X(1) + qp.Y(1))

        circ_split = qjit(self.snc_pass(circ), capture=capture_mode)
        circ_ref = qjit(circ, capture=capture_mode)
        assert np.allclose(circ_split(), circ_ref())

    def test_duplicate_observables(self, capture_mode):
        """Z(0) appears twice -> deduplicated, single group."""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circ():
            qp.RX(0.3, wires=0)
            return qp.expval(2 * qp.Z(0) + qp.Z(0))

        circ_split = qjit(self.snc_pass(circ), capture=capture_mode)
        circ_ref = qjit(circ, capture=capture_mode)
        assert np.allclose(circ_split(), circ_ref())

    def test_multiple_hamiltonians_with_shared_observable(self, capture_mode):
        """expval(2*Y(0)+X(1)), expval(3*X(1)): X(1) shared across MPs, deduplicated, 1 group"""
        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circ():
            qp.RX(0.4, wires=0)
            qp.RY(0.6, wires=1)
            return qp.expval(2 * qp.Y(0) + qp.X(1)), qp.expval(3 * qp.X(1))

        circ_split = qjit(self.snc_pass(circ), capture=capture_mode)
        circ_ref = qjit(circ, capture=capture_mode)
        assert np.allclose(circ_split(), circ_ref())


if __name__ == "__main__":
    pytest.main(["-x", __file__])
