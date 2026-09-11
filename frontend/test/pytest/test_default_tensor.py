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
"""Unit tests for the `default.tensor` runtime backend (exact tensor network).

The differential tests against ``lightning.qubit`` are the real safety net here:
they run identical circuits through the same qjit pipeline, which catches
wire-ordering and convention bugs that self-consistent checks cannot see.
"""

import numpy as np
import pennylane as qp
import pytest

from catalyst import measure, qjit
from catalyst.device import extract_backend_info
from catalyst.utils.exceptions import CompileError

# default.tensor requires quimb, which is an optional PennyLane dependency.
quimb = pytest.importorskip("quimb")

TOL = 1e-10
PARAMS = np.array([0.63, -1.17, 0.42])


def tensor_device(wires, max_intermediate_log2=None, **kwargs):
    """An exact-method default.tensor device.

    `max_intermediate_log2` is set as an attribute rather than passed to the
    constructor: PennyLane's DefaultTensor validates **kwargs against a fixed
    allowlist and raises TypeError on anything outside it.
    """
    dev = qp.device("default.tensor", wires=wires, method="tn", **kwargs)
    if max_intermediate_log2 is not None:
        dev.max_intermediate_log2 = max_intermediate_log2
    return dev


def run(dev, body, meas, *args):
    """qjit a QNode assembled from `body` (gates) and `meas` (return value)."""

    @qjit
    @qp.qnode(dev)
    def circuit(*a):
        body(*a)
        return meas()

    return np.asarray(circuit(*args))


class TestBackendResolution:
    """The device label must resolve to the runtime backend with no plugin code."""

    def test_backend_info(self):
        """default.tensor maps to the DefaultTensor factory symbol."""
        info = extract_backend_info(tensor_device(2))
        assert info.c_interface_name == "DefaultTensor"
        assert "librtd_default_tensor" in info.lpath

    def test_mps_method_is_rejected(self):
        """The runtime performs exact contraction, so the truncating MPS method
        must be refused rather than silently producing different numbers."""
        with pytest.raises(CompileError, match="method 'mps' is not supported"):
            extract_backend_info(qp.device("default.tensor", wires=2, method="mps"))

    def test_default_method_is_rejected(self):
        """PennyLane defaults to method='mps', so an unqualified device errors."""
        with pytest.raises(CompileError, match="not supported"):
            extract_backend_info(qp.device("default.tensor", wires=2))


class TestCorrectness:
    """Numerical correctness of the bare gate set."""

    def test_bell_state(self):
        """A Bell state has equal weight on |00> and |11>."""

        def body():
            qp.Hadamard(0)
            qp.CNOT([0, 1])

        assert np.allclose(run(tensor_device(2), body, qp.probs), [0.5, 0, 0, 0.5], atol=TOL)

    @pytest.mark.parametrize("theta", [0.0, 0.3, np.pi / 2, np.pi, 2.7])
    def test_ry_expval(self, theta):
        """RY(theta) gives <Z> = cos(theta)."""
        out = run(
            tensor_device(1), lambda t: qp.RY(t, wires=0), lambda: qp.expval(qp.PauliZ(0)), theta
        )
        assert np.isclose(out, np.cos(theta), atol=TOL)

    @pytest.mark.parametrize("theta", [0.2, 1.1, 2.5])
    def test_rx_expval_y(self, theta):
        """RX(theta) gives <Y> = -sin(theta)."""
        out = run(
            tensor_device(1), lambda t: qp.RX(t, wires=0), lambda: qp.expval(qp.PauliY(0)), theta
        )
        assert np.isclose(out, -np.sin(theta), atol=TOL)

    @pytest.mark.parametrize("theta", [0.2, 1.1, 2.5])
    def test_variance(self, theta):
        """RY(theta) gives var(Z) = 1 - cos^2(theta)."""
        out = run(
            tensor_device(1), lambda t: qp.RY(t, wires=0), lambda: qp.var(qp.PauliZ(0)), theta
        )
        assert np.isclose(out, 1 - np.cos(theta) ** 2, atol=TOL)

    def test_wire_ordering_is_not_symmetric(self):
        """Wire 0 is the most significant bit; X(0) and X(1) must differ."""
        p0 = run(tensor_device(2), lambda: qp.PauliX(0), qp.probs)
        p1 = run(tensor_device(2), lambda: qp.PauliX(1), qp.probs)
        assert np.allclose(p0, [0, 0, 1, 0], atol=TOL)
        assert np.allclose(p1, [0, 1, 0, 0], atol=TOL)

    def test_adjoint_round_trip(self):
        """A gate followed by its adjoint restores the initial state."""

        def body(t):
            qp.RX(t, wires=0)
            qp.adjoint(qp.RX(t, wires=0))

        assert np.isclose(
            run(tensor_device(1), body, lambda: qp.expval(qp.PauliZ(0)), 0.9), 1.0, atol=TOL
        )

    def test_multi_controlled_gate(self):
        """An X with two control wires acts as a Toffoli."""

        def body():
            qp.PauliX(0)
            qp.PauliX(1)
            qp.ctrl(qp.PauliX(2), control=[0, 1])

        assert np.isclose(run(tensor_device(3), body, qp.probs)[7], 1.0, atol=1e-9)

    def test_gate_outside_gateset_is_decomposed(self):
        """Toffoli is absent from the TOML, so Catalyst must decompose it."""

        def body():
            qp.PauliX(0)
            qp.PauliX(1)
            qp.Toffoli(wires=[0, 1, 2])

        assert np.isclose(run(tensor_device(3), body, qp.probs)[7], 1.0, atol=1e-9)


class TestObservables:
    """Observables are evaluated by closing a <psi|O|psi> network."""

    def test_hamiltonian(self):
        """A linear combination of observables is summed correctly."""

        def body():
            qp.Hadamard(0)
            qp.CNOT([0, 1])

        def meas():
            return qp.expval(0.5 * qp.PauliZ(0) @ qp.PauliZ(1) + 0.25 * qp.PauliX(0))

        assert np.isclose(run(tensor_device(2), body, meas), 0.5, atol=TOL)

    def test_hamiltonian_over_disjoint_wires(self):
        """Summands acting on different wires must not interfere."""

        def body():
            qp.Hadamard(0)
            qp.CNOT([0, 1])

        def meas():
            return qp.expval(0.5 * qp.PauliZ(0) + 0.25 * qp.PauliZ(2))

        assert np.isclose(run(tensor_device(3), body, meas), 0.25, atol=TOL)

    def test_identity_is_normalisation(self):
        """<Identity> is 1 for any normalised state."""
        assert np.isclose(
            run(
                tensor_device(2),
                lambda: (qp.Hadamard(0), qp.CNOT([0, 1])),
                lambda: qp.expval(qp.Identity(0)),
            ),
            1.0,
            atol=TOL,
        )


class TestDynamicQubitManagement:
    """Catalyst exposes two independent dynamic mechanisms; both are covered."""

    def test_automatic_qubit_management(self):
        """wires=None lets the runtime allocate qubits on first use."""

        @qjit
        @qp.qnode(tensor_device(None))
        def circuit():
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            return qp.probs(wires=[0, 1])

        assert np.allclose(np.asarray(circuit()), [0.5, 0, 0, 0.5], atol=TOL)

    def test_automatic_qubit_management_ghz(self):
        """Automatic management extends to more than two lazily-created wires."""

        @qjit
        @qp.qnode(tensor_device(None))
        def circuit():
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            qp.CNOT([1, 2])
            return qp.probs(wires=[0, 1, 2])

        p = np.asarray(circuit())
        assert np.isclose(p[0], 0.5, atol=TOL)
        assert np.isclose(p[7], 0.5, atol=TOL)

    def test_scratch_register(self):
        """qp.allocate requires program capture; a scratch qubit returned to |0>
        must leave the rest of the state untouched."""

        @qjit(capture=True)
        @qp.qnode(tensor_device(2))
        def circuit():
            qp.Hadamard(0)
            with qp.allocate(1) as aux:
                qp.CNOT([0, aux[0]])
                qp.CNOT([0, aux[0]])  # undo
            qp.CNOT([0, 1])
            return qp.probs(wires=[0, 1])

        assert np.allclose(np.asarray(circuit()), [0.5, 0, 0, 0.5], atol=TOL)

    def test_nested_scratch_registers(self):
        """Nested qp.allocate blocks allocate and free in the right order."""

        @qjit(capture=True)
        @qp.qnode(tensor_device(2))
        def circuit():
            qp.PauliX(0)
            with qp.allocate(1) as a:
                qp.CNOT([0, a[0]])
                with qp.allocate(1) as b:
                    qp.CNOT([a[0], b[0]])
                    qp.CNOT([b[0], 1])
                    qp.CNOT([a[0], b[0]])
                qp.CNOT([0, a[0]])
            return qp.probs(wires=[0, 1])

        assert np.allclose(np.asarray(circuit()), [0, 0, 0, 1], atol=1e-9)

    def test_releasing_entangled_scratch_stays_normalised(self):
        """Freeing a still-entangled qubit acts like an unread measurement: the
        survivors collapse to one branch but stay normalised."""

        @qjit(capture=True)
        @qp.qnode(tensor_device(1))
        def circuit():
            qp.Hadamard(0)
            with qp.allocate(1) as aux:
                qp.CNOT([0, aux[0]])  # deliberately left entangled
            return qp.probs(wires=[0])

        for _ in range(10):
            p = np.asarray(circuit())
            assert np.isclose(p.sum(), 1.0, atol=1e-9)
            assert np.isclose(max(p), 1.0, atol=1e-9)


class TestMidCircuitMeasurement:
    """MCM projects the network and renormalises."""

    def test_collapse_and_correlate(self):
        """A measurement conditioned onto another wire leaves only |00> and |11>."""

        @qjit
        @qp.qnode(tensor_device(2))
        def circuit():
            qp.Hadamard(0)
            m = measure(0)

            @qp.cond(m)
            def flip():
                qp.PauliX(1)

            flip()
            return qp.probs()

        for _ in range(10):
            p = np.asarray(circuit())
            assert np.isclose(p[1], 0.0) and np.isclose(p[2], 0.0)
            assert np.isclose(p[0] + p[3], 1.0)

    def test_postselection(self):
        """Postselecting |1> collapses the state onto that branch."""

        @qjit
        @qp.qnode(tensor_device(1))
        def circuit():
            qp.Hadamard(0)
            measure(0, postselect=1)
            return qp.probs()

        assert np.allclose(np.asarray(circuit()), [0.0, 1.0], atol=TOL)

    def test_seeded_execution_is_reproducible(self):
        """qjit(seed=...) makes measurement outcomes reproducible."""

        def build():
            @qjit(seed=1234)
            @qp.qnode(tensor_device(5))
            def circuit():
                for i in range(5):
                    qp.Hadamard(i)
                return [measure(i) for i in range(5)]

            return circuit

        assert [bool(x) for x in build()()] == [bool(x) for x in build()()]


class TestShots:
    """Shot-based measurement processes."""

    def test_sample_correlation(self):
        """Bell-state samples must agree on both wires."""

        @qjit
        @qp.set_shots(2000)
        @qp.qnode(tensor_device(2))
        def circuit():
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            return qp.sample()

        s = np.asarray(circuit())
        assert s.shape == (2000, 2)
        assert np.all(s[:, 0] == s[:, 1])

    def test_counts_sum_to_shots(self):
        """Counts total the shot number and only populate reachable states."""

        @qjit
        @qp.set_shots(1500)
        @qp.qnode(tensor_device(2))
        def circuit():
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            return qp.counts()

        # Assign before unpacking, matching the convention in
        # test_mid_circuit_measurement.py (and avoiding a pylint false positive).
        res = circuit()
        _eigvals, counts = res
        counts = np.asarray(counts)
        assert counts.sum() == 1500
        assert counts[1] == 0 and counts[2] == 0


class TestTensorNetworkBehaviour:
    """Properties that only hold because contraction is over a network."""

    def test_wide_circuit_beyond_dense_reach(self):
        """A 40-qubit GHZ chain: a dense statevector would need 2^40 amplitudes,
        and the intermediate cap is far below that."""
        n = 40

        @qjit
        @qp.qnode(tensor_device(n, max_intermediate_log2=20))
        def circuit():
            qp.Hadamard(0)
            for i in range(n - 1):
                qp.CNOT([i, i + 1])
            return qp.expval(qp.PauliZ(0) @ qp.PauliZ(n - 1))

        assert np.isclose(float(circuit()), 1.0, atol=TOL)

    def test_partial_probs_on_wide_register(self):
        """Marginals come from a reduced density matrix, so their cost scales
        with the number of requested wires, not the register width."""
        n = 30

        @qjit
        @qp.qnode(tensor_device(n, max_intermediate_log2=20))
        def circuit():
            qp.Hadamard(0)
            for i in range(n - 1):
                qp.CNOT([i, i + 1])
            return qp.probs(wires=[0, n - 1])

        assert np.allclose(np.asarray(circuit()), [0.5, 0, 0, 0.5], atol=TOL)

    def test_memory_guard_raises_cleanly(self):
        """An infeasible contraction must raise, not exhaust memory."""

        @qjit
        @qp.qnode(tensor_device(20, max_intermediate_log2=8))
        def circuit():
            for i in range(20):
                qp.Hadamard(i)
            return qp.probs()

        with pytest.raises(RuntimeError, match="exceeding the configured limit"):
            circuit()


class TestShotNoise:
    """Regression tests: finite shots must produce shot noise.

    The device originally returned analytic probabilities even when a shot count
    was set, so `probs()` came back exact while `sample()`/`counts()` on the same
    circuit were noisy -- and results disagreed with every other device for the
    same program. Lightning branches on `device_shots != 0` in
    `Probs`/`PartialProbs`/`Expval`/`Var`; this device must too.
    """

    def test_probs_are_quantised_by_shots(self):
        """With N shots, every probability must be a multiple of 1/N."""
        shots = 1000

        @qjit
        @qp.set_shots(shots)
        @qp.qnode(tensor_device(2))
        def circuit():
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            return qp.probs()

        p = np.asarray(circuit())
        counts = p * shots
        assert np.allclose(
            counts, np.round(counts), atol=1e-9
        ), f"probabilities are not multiples of 1/{shots}: {p}"
        # Unreachable basis states must stay exactly zero.
        assert p[1] == 0.0 and p[2] == 0.0

    def test_probs_vary_across_runs_with_shots(self):
        """Repeated shot-based runs must not all return the identical vector."""

        @qjit
        @qp.set_shots(500)
        @qp.qnode(tensor_device(2))
        def circuit():
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            return qp.probs()

        seen = {tuple(np.asarray(circuit())) for _ in range(15)}
        assert len(seen) > 1, "shot-based probs are deterministic; shots are being ignored"

    def test_probs_are_analytic_without_shots(self):
        """shots=None must still give exact probabilities."""

        @qjit
        @qp.qnode(tensor_device(2))
        def circuit():
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            return qp.probs()

        assert np.allclose(np.asarray(circuit()), [0.5, 0, 0, 0.5], atol=TOL)

    def test_expval_is_shot_based(self):
        """A finite shot count must make <Z> noisy but centred on the exact value."""
        exact = np.cos(0.7)

        def build(shots):
            @qjit
            @qp.set_shots(shots)
            @qp.qnode(tensor_device(1))
            def circuit():
                qp.RY(0.7, wires=0)
                return qp.expval(qp.PauliZ(0))

            return circuit

        vals = {float(build(500)()) for _ in range(12)}
        assert len(vals) > 1, "shot-based expval is deterministic; shots are being ignored"
        assert abs(np.mean(list(vals)) - exact) < 0.15

    def test_expval_matches_lightning_statistically(self):
        """Shot-based expectation values must agree with lightning within noise."""

        def build(dev):
            @qjit
            @qp.set_shots(4000)
            @qp.qnode(dev)
            def circuit():
                qp.RY(0.7, wires=0)
                return qp.expval(qp.PauliZ(0))

            return circuit

        mine = np.mean([float(build(tensor_device(1))()) for _ in range(10)])
        ref = np.mean([float(build(qp.device("lightning.qubit", wires=1))()) for _ in range(10)])
        assert abs(mine - ref) < 0.05, f"tn={mine} vs lightning={ref}"

    def test_pauli_x_expval_with_shots_needs_basis_rotation(self):
        """<X> on |+> is deterministic (+1). Sampling in the computational basis
        instead of the X eigenbasis would return ~0 here."""

        @qjit
        @qp.set_shots(2000)
        @qp.qnode(tensor_device(1))
        def circuit():
            qp.Hadamard(0)
            return qp.expval(qp.PauliX(0))

        assert np.isclose(float(circuit()), 1.0, atol=1e-9)

    def test_mcm_branches_are_both_reachable(self):
        """A conditional circuit must not always collapse the same way.

        Two devices running this program legitimately return different
        probability vectors on any single call, because each collapses onto its
        own random MCM branch. What must hold is that both branches occur.

        NOTE: `qp.capture.enable()` is required for the two-callable
        `qp.cond(m, if_true, if_false)()` form to register the false branch.
        Without capture, only the true branch is applied and every run lands on
        the same outcome -- on lightning.qubit too, so it is a tracing-mode
        property rather than a device one.
        """
        x = 0.1234
        qp.capture.enable()
        try:

            @qp.qjit(capture=True)
            @qp.set_shots(2000)
            @qp.qnode(tensor_device(3))
            def circuit(theta):
                qp.Hadamard(0)
                qp.RY(theta, wires=0)
                m = qp.measure(0)

                def ansatz_true():
                    qp.RX(theta, wires=0)

                def ansatz_false():
                    qp.RY(theta, wires=0)

                qp.cond(m, ansatz_true, ansatz_false)()
                qp.CNOT([0, 1])
                qp.CNOT([1, 2])
                return qp.probs()

            dominant = set()
            for _ in range(40):
                p = np.asarray(circuit(x))
                assert np.isclose(p.sum(), 1.0, atol=1e-9)
                # Only |000> and |111> are reachable through the CNOT chain.
                assert np.allclose(p[1:7], 0.0, atol=1e-12)
                dominant.add(int(np.argmax(p)))
        finally:
            qp.capture.disable()

        assert dominant == {
            0,
            7,
        }, f"expected both MCM branches over 40 runs, saw {sorted(dominant)}"

    def test_mcm_circuit_agrees_with_lightning_statistically(self):
        """Per-call vectors differ by branch, but the rate at which each branch
        is taken must match lightning.qubit."""
        x = 0.1234
        trials = 200

        def branch_rate(dev):
            @qp.qjit(capture=True)
            @qp.set_shots(2000)
            @qp.qnode(dev)
            def circuit(theta):
                qp.Hadamard(0)
                qp.RY(theta, wires=0)
                m = qp.measure(0)

                def ansatz_true():
                    qp.RX(theta, wires=0)

                def ansatz_false():
                    qp.RY(theta, wires=0)

                qp.cond(m, ansatz_true, ansatz_false)()
                qp.CNOT([0, 1])
                qp.CNOT([1, 2])
                return qp.probs()

            return np.mean(
                [1.0 if np.argmax(np.asarray(circuit(x))) == 7 else 0.0 for _ in range(trials)]
            )

        qp.capture.enable()
        try:
            frac_mine = branch_rate(tensor_device(3))
            frac_ref = branch_rate(qp.device("lightning.qubit", wires=3))
        finally:
            qp.capture.disable()

        sigma = np.sqrt(0.25 / trials)
        assert (
            abs(frac_mine - frac_ref) < 6 * sigma
        ), f"branch rates differ: tn={frac_mine} lightning={frac_ref}"


DIFF_CASES = {
    "ghz3": (lambda p: (qp.Hadamard(0), qp.CNOT([0, 1]), qp.CNOT([1, 2])), qp.probs, 3),
    "layered": (
        lambda p: (
            qp.RX(p[0], 0),
            qp.RY(p[1], 1),
            qp.CNOT([0, 1]),
            qp.RZ(p[2], 2),
            qp.CZ([1, 2]),
            qp.Hadamard(0),
        ),
        qp.probs,
        3,
    ),
    "state4": (
        lambda p: (
            qp.Hadamard(0),
            qp.RY(p[0], 1),
            qp.CNOT([0, 1]),
            qp.RZ(p[1], 2),
            qp.CNOT([1, 2]),
            qp.RX(p[2], 3),
            qp.CZ([2, 3]),
        ),
        qp.state,
        4,
    ),
    "expval_xy": (
        lambda p: (qp.RY(p[0], 0), qp.RX(p[1], 1), qp.CNOT([0, 1])),
        lambda: qp.expval(qp.PauliX(0) @ qp.PauliY(1)),
        2,
    ),
    "var_z": (lambda p: (qp.RY(p[0], 0), qp.CNOT([0, 1])), lambda: qp.var(qp.PauliZ(0)), 2),
    "hamiltonian": (
        lambda p: (qp.RY(p[0], 0), qp.CNOT([0, 1]), qp.RZ(p[1], 1)),
        lambda: qp.expval(
            0.5 * qp.PauliZ(0) @ qp.PauliZ(1) + 0.25 * qp.PauliX(0) - 0.75 * qp.PauliY(1)
        ),
        2,
    ),
    "hermitian": (
        lambda p: (qp.RY(p[0], 0), qp.CNOT([0, 1])),
        lambda: qp.expval(
            qp.Hermitian(np.array([[1.0, 0.5], [0.5, -1.0]], dtype=complex), wires=0)
        ),
        2,
    ),
    "ctrl_chain": (
        lambda p: (qp.PauliX(0), qp.PauliX(1), qp.ctrl(qp.PauliX(2), control=[0, 1])),
        qp.probs,
        3,
    ),
    "deep6": (
        lambda p: [qp.Hadamard(w) for w in range(6)] + [qp.CNOT([i, i + 1]) for i in range(5)],
        qp.probs,
        6,
    ),
    "partial_probs": (
        lambda p: (qp.Hadamard(0), qp.CNOT([0, 1]), qp.CNOT([1, 2]), qp.RY(p[0], 1)),
        lambda: qp.probs(wires=[0, 2]),
        3,
    ),
    "decomp_toffoli": (
        lambda p: (qp.PauliX(0), qp.PauliX(1), qp.Toffoli(wires=[0, 1, 2])),
        qp.probs,
        3,
    ),
    "decomp_ising": (lambda p: (qp.IsingXX(p[0], wires=[0, 1]),), qp.probs, 2),
    "decomp_s_t": (lambda p: (qp.Hadamard(0), qp.S(0), qp.T(0), qp.Hadamard(0)), qp.probs, 1),
    "decomp_swap": (lambda p: (qp.PauliX(0), qp.SWAP([0, 1])), qp.probs, 2),
}


@pytest.mark.parametrize("name", list(DIFF_CASES))
def test_matches_lightning_qubit(name):
    """Identical circuits on default.tensor and lightning.qubit must agree."""
    body, meas, nwires = DIFF_CASES[name]
    mine = run(tensor_device(nwires), body, meas, PARAMS)
    ref = run(qp.device("lightning.qubit", wires=nwires), body, meas, PARAMS)
    assert np.allclose(mine, ref, atol=TOL), f"{name}: max diff {np.max(np.abs(mine - ref))}"


if __name__ == "__main__":
    pytest.main(["-x", __file__])
