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
Unit tests for the dynamic work wire allocation.
Note that this feature is only available under the plxpr pipeline.
"""

import textwrap

import numpy as np
import pennylane as qp
import pytest
from jax import numpy as jnp
from pennylane.capture import subroutine

from catalyst import qjit
from catalyst.utils.exceptions import CompileError


def test_basic_dynamic_wire_alloc_plain_API(backend):
    """
    Test basic qp.allocate and qp.deallocate.
    """

    @qjit(capture=True)
    @qp.qnode(qp.device(backend, wires=3))
    def circuit():
        qp.X(1)  # |010>

        q = qp.allocate(1)  # |010> and |0>
        qp.X(q[0])  # |010> and |1>
        qp.CNOT(wires=[q[0], 2])  # |011> and |1>
        qp.deallocate(q[0])  # |011>

        return qp.probs(wires=[0, 1, 2])

    observed = circuit()

    expected = [0, 0, 0, 1, 0, 0, 0, 0]
    assert np.allclose(expected, observed)


def test_basic_dynamic_wire_alloc_ctx_API(backend):
    """
    Test basic qp.allocate with context manager API.
    """

    @qjit(capture=True)
    @qp.qnode(qp.device(backend, wires=3))
    def circuit():
        qp.X(1)

        with qp.allocate(1) as q:
            qp.X(q[0])
            qp.CNOT(wires=[q[0], 2])

        return qp.probs(wires=[0, 1, 2])

    observed = circuit()

    expected = [0, 0, 0, 1, 0, 0, 0, 0]
    assert np.allclose(expected, observed)


def test_measure(backend):
    """
    Test qp.allocate with qp.Measure ops.
    """

    @qjit(autograph=True, capture=True)
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        with qp.allocate(1) as q:
            qp.Hadamard(q[0])
            m = qp.measure(wires=q[0], postselect=1)

        if m:
            qp.X(0)

        return qp.probs(wires=[0])  # |1>

    observed = circuit()

    expected = [0, 1]
    assert np.allclose(expected, observed)


def test_measure_with_reset(backend):
    """
    Test qp.allocate with qp.Measure ops with resetting.
    """

    @qjit(autograph=True, capture=True)
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        with qp.allocate(1) as q:
            qp.Hadamard(q[0])
            # measure 1 and reset q[0] to |0>
            m1 = qp.measure(wires=q[0], reset=True, postselect=1)
            # measure 0
            m0 = qp.measure(wires=q[0])

        if m0:  # should not be hit
            qp.RX(37.42, wires=[0])

        if m1:  # should be hit
            qp.X(wires=[0])

        return qp.probs(wires=[0])

    observed = circuit()

    expected = [0, 1]
    assert np.allclose(expected, observed)


@pytest.mark.parametrize("ctrl_val, expected", [(False, [0, 1]), (True, [1, 0])])
def test_qp_ctrl(ctrl_val, expected, backend):
    """
    Test qp.allocate with qp.ctrl ops.
    """

    @qjit(capture=True)
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        with qp.allocate(1) as q:
            qp.ctrl(qp.X, (q), control_values=ctrl_val)(wires=0)
        return qp.probs(wires=[0])

    observed = circuit()

    assert np.allclose(expected, observed)


def test_QubitUnitary(backend):
    """
    Test qp.allocate with qp.QubitUnitary ops.
    """

    @qjit(capture=True)
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        with qp.allocate(2) as qs:
            qp.QubitUnitary(jnp.identity(8), wires=[0, qs[0], qs[1]])
        return qp.probs(wires=[0])

    observed = circuit()

    expected = [1, 0]
    assert np.allclose(expected, observed)


def test_StatePrep(backend):
    """
    Test qp.allocate with qp.StatePrep ops.
    """

    @qjit(capture=True)
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        with qp.allocate(1) as q:
            qp.StatePrep(jnp.array([0, 0, 0, 1]), wires=[0, q[0]])  # |11>
        return qp.probs(wires=[0])  # |1>

    observed = circuit()

    expected = [0, 1]
    assert np.allclose(expected, observed)


def test_BasisState(backend):
    """
    Test qp.allocate with qp.BasisState ops.
    """

    @qjit(capture=True)
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        with qp.allocate(1) as q:
            qp.BasisState(jnp.array([1, 0]), wires=[q[0], 0])  # |10>
        return qp.probs(wires=[0])  # |0>

    observed = circuit()

    expected = [1, 0]
    assert np.allclose(expected, observed)


@pytest.mark.parametrize("cond, expected", [(True, [0, 0, 1, 0]), (False, [0, 1, 0, 0])])
def test_dynamic_wire_alloc_cond(cond, expected, backend):
    """
    Test qp.allocate and qp.deallocate inside cond.
    """

    @qjit(autograph=True, capture=True)
    @qp.qnode(qp.device(backend, wires=2))
    def circuit(c):
        if c:
            q = qp.allocate(1)[0]
            qp.X(wires=q)
            qp.CNOT(wires=[q, 0])
            qp.deallocate(q)
        else:
            q = qp.allocate(1)[0]
            qp.X(wires=q)
            qp.CNOT(wires=[q, 1])
            qp.deallocate(q)

        return qp.probs(wires=[0, 1])

    observed = circuit(cond)

    assert np.allclose(expected, observed)


@pytest.mark.parametrize("cond, expected", [(True, [0, 1, 0, 0]), (False, [1, 0, 0, 0])])
def test_dynamic_wire_alloc_cond_outside(cond, expected, backend):
    """
    Test passing dynamically allocated wires into a cond.
    """

    @qjit(autograph=True, capture=True)
    @qp.qnode(qp.device(backend, wires=2))
    def circuit(c):
        with qp.allocate(1) as q1:
            with qp.allocate(1) as q2:
                qp.X(q1[0])
                qp.Identity(0)
                if c:
                    qp.CNOT(wires=[q1[0], 1])  # |01>
                else:
                    qp.CNOT(wires=[q2[0], 1])  # |00>

        return qp.probs(wires=[0, 1])

    observed = circuit(cond)

    assert np.allclose(expected, observed)


@pytest.mark.parametrize("cond, expected", [(True, [0, 1, 0, 0]), (False, [1, 0, 0, 0])])
def test_dynamic_wire_alloc_cond_outside_non_capture(cond, expected, backend):
    """
    Test passing dynamically allocated wires into qp.cond on the legacy pathway.
    """

    @qjit(capture=False)
    @qp.qnode(qp.device(backend, wires=2))
    def circuit(c):
        q = qp.allocate(1)[0]
        qp.X(q)

        def true_fn():
            qp.CNOT(wires=[q, 1])

        def false_fn():
            qp.Identity(0)

        qp.cond(c, true_fn, false_fn)()
        qp.deallocate(q)
        return qp.probs(wires=[0, 1])

    observed = circuit(cond)

    assert np.allclose(expected, observed)


@pytest.mark.parametrize(
    "num_iter, expected", [(3, [0, 0, 1, 0, 0, 0, 0, 0]), (4, [1, 0, 0, 0, 0, 0, 0, 0])]
)
def test_dynamic_wire_alloc_forloop(num_iter, expected, backend):
    """
    Test qp.allocate and qp.deallocate inside for loop.
    """

    @qjit(autograph=True, capture=True)
    @qp.qnode(qp.device(backend, wires=3))
    def circuit(N):
        for _ in range(N):
            q = qp.allocate(1)[0]
            qp.X(wires=q)
            qp.CNOT(wires=[q, 1])
            qp.deallocate(q)

        return qp.probs(wires=[0, 1, 2])

    observed = circuit(num_iter)

    assert np.allclose(expected, observed)


def test_dynamic_wire_alloc_forloop_outside(backend):
    """
    Test passing dynamically allocated wires into a for loop.
    """

    @qjit(autograph=True, capture=True)
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        with qp.allocate(1) as q:
            qp.X(wires=q[0])
            for _ in range(3):
                qp.CNOT(wires=[q[0], 0])

        return qp.probs(wires=[0])

    observed = circuit()
    expected = [0, 1]

    assert np.allclose(expected, observed)


def test_dynamic_wire_alloc_forloop_outside_multiple_regs(backend):
    """
    Test using multiple dynamically allocated registers from inside for loop.
    """

    @qjit(autograph=True, capture=True)
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        with qp.allocate(1) as q1:
            with qp.allocate(1) as q2:
                for _ in range(3):
                    qp.CNOT(wires=[q1[0], 0])
                    qp.CNOT(wires=[q2[0], 0])

        return qp.probs(wires=[0])

    observed = circuit()
    expected = [1, 0]

    assert np.allclose(expected, observed)


@pytest.mark.parametrize(
    "num_iter, expected", [(3, [0, 0, 1, 0, 0, 0, 0, 0]), (4, [1, 0, 0, 0, 0, 0, 0, 0])]
)
def test_dynamic_wire_alloc_whileloop(num_iter, expected, backend):
    """
    Test qp.allocate and qp.deallocate inside while loop.
    """

    @qjit(autograph=True, capture=True)
    @qp.qnode(qp.device(backend, wires=3))
    def circuit(N):
        i = 0
        while i < N:
            q = qp.allocate(1)[0]
            qp.X(wires=q)
            qp.CNOT(wires=[q, 1])
            qp.deallocate(q)
            i += 1

        return qp.probs(wires=[0, 1, 2])

    observed = circuit(num_iter)

    assert np.allclose(expected, observed)


@pytest.mark.parametrize("num_iter, expected", [(3, [0, 1, 0, 0]), (4, [1, 0, 0, 0])])
def test_dynamic_wire_alloc_whileloop_outside(num_iter, expected, backend):
    """
    Test passing dynamically allocated wires into a while loop.
    """

    @qjit(autograph=True, capture=True)
    @qp.qnode(qp.device(backend, wires=2))
    def circuit(N):
        i = 0
        with qp.allocate(1) as q1:
            with qp.allocate(1) as q2:
                qp.X(q1[0])
                while i < N:
                    qp.CNOT(wires=[q1[0], 1])
                    qp.CNOT(wires=[q2[0], 1])
                    i += 1

        return qp.probs(wires=[0, 1])

    observed = circuit(num_iter)

    assert np.allclose(expected, observed)


@pytest.mark.parametrize("flip_again, expected", [(True, [1, 0]), (False, [0, 1])])
def test_subroutine(flip_again, expected, backend):
    """
    Test passing dynamically allocated wires into a subroutine.
    """

    @subroutine
    def flip(w):
        qp.X(w)
        qp.CNOT(wires=[w, 0])

    @qjit(capture=True)
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        with qp.allocate(1) as q1:
            with qp.allocate(1) as q2:
                flip(q1[0])
                if flip_again:
                    flip(q2[0])
        return qp.probs(wires=[0])

    observed = circuit()
    assert np.allclose(expected, observed)


def test_subroutine_multiple_args(backend):
    """
    Test passing dynamically allocated wires into a subroutine with multiple arguments.
    """

    @subroutine
    def flip(w1, w2, theta):
        qp.X(w1)
        qp.X(w2)
        qp.ctrl(qp.RX, (w1, w2))(theta, wires=0)

    @qjit(capture=True)
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        with qp.allocate(1) as q1:
            with qp.allocate(2) as q2:
                flip(q1[0], q2[1], jnp.pi)
        return qp.probs(wires=[0])

    observed = circuit()
    expected = [0, 1]
    assert np.allclose(expected, observed)


def test_subroutine_and_loop(backend):
    """
    Test passing dynamically allocated wires into a subroutine with loops.
    """

    @subroutine
    def flip(wire, theta):
        """
        Apply three X gates to the input wire, effectively NOT-ing it.
        """

        @qp.for_loop(0, 3, 1)
        def loop(i, _theta):  # pylint: disable=unused-argument
            qp.X(wire)
            return jnp.sin(_theta)

        _ = loop(theta)

    @qjit(capture=True)
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        with qp.allocate(1) as q1:
            flip(q1[0], 0.0)
            qp.CNOT(wires=[q1[0], 0])
        return qp.probs(wires=[0])

    observed = circuit()
    expected = [0, 1]
    assert np.allclose(expected, observed)


def test_subroutine_and_loop_multiple_args(backend):
    """
    Test passing dynamically allocated wires into a subroutine with loops and multiple arguments.
    """

    @subroutine
    def flip(w1, w2, w3, theta):
        @qp.for_loop(0, 2, 1)
        def loop(i, _theta):  # pylint: disable=unused-argument
            qp.X(w1)
            qp.Y(w2)
            qp.Z(w3)
            qp.ctrl(qp.RX, (w1, w2))(_theta, wires=0)
            qp.ctrl(qp.RY, (w2, w3))(_theta, wires=1)
            return jnp.sin(_theta)

        _ = loop(theta)

    @qjit(capture=True)
    @qp.qnode(qp.device(backend, wires=2))
    def circuit():
        with qp.allocate(2) as q1:
            with qp.allocate(3) as q2:
                flip(q1[0], q1[1], q2[2], 1.23)

        return qp.probs(wires=[0, 1])

    @qp.qnode(qp.device("default.qubit", wires=7))
    def ref_circuit():
        for _ in range(2):
            qp.X(0)
            qp.Y(1)
            qp.Z(2)
            qp.ctrl(qp.RX, (0, 1))(1.23, wires=3)
            qp.ctrl(qp.RY, (1, 2))(1.23, wires=4)

        return qp.probs(wires=[3, 4])

    assert np.allclose(circuit(), ref_circuit())


@pytest.mark.parametrize(
    "measurement_fn, shots, expected",
    [
        (lambda: qp.expval(qp.Z(0)), None, 1.0),
        (lambda: qp.var(qp.Z(0)), None, 0.0),
        (lambda: qp.sample(wires=[0]), 10, [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
    ],
)
def test_non_probs_measurement_with_dynamic_wires(backend, measurement_fn, shots, expected):
    """
    Test that non-probs measurements with dynamic wire allocations work.
    """

    @qjit(capture=True)
    @qp.qnode(qp.device(backend, wires=1), shots=shots)
    def circuit():
        with qp.allocate(1) as q:
            qp.X(q[0])
        return measurement_fn()

    observed = circuit()
    assert np.allclose(observed, expected)


def test_adjoint(backend):
    """
    Test adjoints work.
    """

    @qjit(capture=True)
    @qp.qnode(qp.device(backend, wires=2))
    def circuit():
        with qp.allocate(1) as q:
            qp.adjoint(qp.RX)(0.12, q[0])
            qp.RX(0.12, q[0])
            qp.CNOT(wires=[q[0], 0])
        return qp.probs(wires=[0, 1])

    expected = [1, 0, 0, 0]
    observed = circuit()
    assert np.allclose(observed, expected)


def test_no_capture_zero_state(backend):
    """Test that zero-state dynamic allocation works without capture enabled."""

    @qjit
    @qp.qnode(qp.device(backend, wires=2))
    def circuit():
        with qp.allocate(1) as q:
            qp.X(q[0])
            qp.CNOT(wires=[q[0], 0])
        return qp.probs(wires=[0, 1])

    assert np.allclose([0, 0, 1, 0], circuit())

    @qjit(target="mlir")
    @qp.qnode(qp.device(backend, wires=2))
    def circuit_mlir():
        with qp.allocate(1) as q:
            qp.X(q[0])
            qp.CNOT(wires=[q[0], 0])
        return qp.probs(wires=[0, 1])

    mlir_str = circuit_mlir.mlir
    assert mlir_str.count("qref.dealloc") >= 2


def test_no_capture_magic_state(backend):
    """
    Test that magic state allocation works without capture enabled.
    """

    @qjit
    @qp.qnode(qp.device(backend, wires=2))
    def baseline():
        qp.H(1)
        qp.T(1)
        qp.CNOT(wires=[1, 0])
        return qp.probs(wires=[0])

    @qjit
    @qp.qnode(qp.device(backend, wires=2))
    def circuit():
        q = qp.allocate(1, state="magic-T")
        qp.CNOT(wires=[q[0], 0])
        return qp.probs(wires=[0])

    assert np.allclose(baseline(), circuit())


@pytest.mark.parametrize("capture", (True, False))
@pytest.mark.parametrize(
    ("state", "prep"),
    (
        ("magic-T", lambda w: (qp.H(w), qp.T(w))),
        ("magic-T-adj", lambda w: (qp.H(w), qp.adjoint(qp.T(w), lazy=False))),
    ),
)
def test_magic_state_allocation(backend, capture, state, prep):
    """Test magic state allocation lowers correctly and produces expected states."""

    @qjit(capture=capture)
    @qp.qnode(qp.device(backend, wires=2))
    def baseline():
        prep(1)
        qp.CNOT(wires=[1, 0])
        return qp.probs(wires=[0])

    @qjit(capture=capture)
    @qp.qnode(qp.device(backend, wires=2))
    def circuit():
        q = qp.allocate(1, state=state)
        qp.CNOT(wires=[q[0], 0])
        return qp.probs(wires=[0])

    assert np.allclose(baseline(), circuit())


def test_no_capture(backend):
    """Test error message when allocate is used without program capture."""
    with pytest.raises(
        CompileError,
        match=r".*\.allocate\(\) with qjit is only supported with program capture enabled\.",
    ):

        @qjit
        @qp.qnode(qp.device(backend, wires=1))
        def circuit():
            with qp.allocate(1) as _:
                pass
            return qp.probs(wires=[0])


def test_deallocate_mixed_fabricate_and_register():
    """Test deallocating fabricate and register wires together is rejected."""
    with pytest.raises(ValueError, match="same allocation instruction"):

        @qjit(capture=True)
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def circuit():
            q_magic = qp.allocate(1, state="magic-T")
            q_reg = qp.allocate(1)
            qp.deallocate([q_magic[0], q_reg[0]])
            return qp.probs(wires=[0])


def test_deallocate_multiple_register_allocations():
    """Test deallocating wires from separate register allocations is rejected."""
    with pytest.raises(ValueError, match="same allocation instruction"):

        @qjit(capture=True)
        @qp.qnode(qp.device("lightning.qubit", wires=3))
        def circuit():
            q1 = qp.allocate(1)
            q2 = qp.allocate(1)
            qp.deallocate([q1[0], q2[0]])
            return qp.probs(wires=[0])


def test_deallocate_non_allocated_wire():
    """Test deallocating a device wire is rejected at compile time."""
    with pytest.raises(TypeError, match="Manual deallocation is only supported"):

        @qjit(capture=True)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit():
            qp.deallocate(0)
            return qp.probs(wires=[0])


def test_magic_state_manual_deallocate(backend):
    """Test manual deallocation of a magic state wire without entangling gates."""

    @qjit(capture=True)
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        q = qp.allocate(1, state="magic-T-adj")
        qp.deallocate(q[0])
        return qp.probs(wires=[0])

    assert np.allclose([1, 0], circuit())


@pytest.mark.parametrize("capture", (True, False))
def test_magic_state_pauli_measure(backend, capture):
    """Test Pauli product measurement on a fabricated magic wire."""

    @qjit(capture=capture, target="mlir")
    @qp.qnode(qp.device(backend, wires=2))
    def circuit():
        qp.CNOT(wires=[0, 1])
        magic = qp.allocate(1, state="magic-T")
        qp.pauli_measure("ZZ", wires=[0, magic[0]])
        return qp.expval(qp.Z(0))

    mlir_str = circuit.mlir
    assert "pbc.ref.fabricate" in mlir_str
    assert "pauli" in mlir_str.lower()


@pytest.mark.parametrize("capture", (True, False))
def test_magic_state_mid_measure(backend, capture):
    """Test mid-circuit measure on device wires after magic state allocation."""

    @qjit(capture=capture, target="mlir")
    @qp.qnode(qp.device(backend, wires=3))
    def circuit():
        magic = qp.allocate(1, state="magic-T")
        qp.pauli_measure("Z", wires=[magic[0]])
        qp.measure(2, reset=True)
        return qp.expval(qp.Z(0))

    circuit()


def test_magic_state_mlir_lowering(backend):
    """Test that magic state allocation lowers to pbc.ref.fabricate."""

    @qjit(target="mlir", capture=True)
    @qp.qnode(qp.device(backend, wires=1))
    def circuit():
        with qp.allocate(1, state="magic-T") as q:
            qp.X(q[0])
        return qp.probs(wires=[0])

    mlir_str = circuit.mlir
    assert "pbc.ref.fabricate" in mlir_str
    assert "magic" in mlir_str
    assert "qref.dealloc_qb" in mlir_str


def test_use_after_free(backend):
    """
    Test error message when used after free.
    """

    with pytest.raises(
        CompileError,
        match="Detected use of a qubit after deallocation",
    ):

        @qjit(capture=True)
        @qp.qnode(qp.device(backend, wires=1))
        def circuit():
            with qp.allocate(1) as q:
                qp.X(q[0])
            qp.Hadamard(q[0])
            return qp.probs(wires=[0])


def test_terminal_MP_all_wires(backend):
    """
    Test error message when used with terminal measurements on all wires.
    """

    with pytest.raises(
        CompileError,
        match=textwrap.dedent("""
            Terminal measurements must take in an explicit list of wires when
            dynamically allocated wires are present in the program.
            """),
    ):

        @qjit(capture=True)
        @qp.qnode(qp.device(backend, wires=1))
        def circuit():
            with qp.allocate(1) as _:
                pass
            return qp.probs()


def test_allocate_state_any_unsupported():
    """Test error when allocating with state=\"any\"."""

    with pytest.raises(
        CompileError,
        match='qp.allocate with state="any" is not supported in Catalyst',
    ):

        @qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            qp.allocate(1, state="any")
            return qp.expval(qp.Z(0))


def test_allocate_restored_unsupported():
    """Test error when allocating with restored=True."""

    with pytest.raises(
        CompileError,
        match="qp.allocate with restored=True is not supported in Catalyst",
    ):

        @qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            qp.allocate(1, restored=True)
            return qp.expval(qp.Z(0))


def test_terminal_MP_dynamic_wires(backend):
    """
    Test error message when used with terminal measurements on dynamic wires.
    """

    with pytest.raises(
        CompileError,
        match=textwrap.dedent("""
            Terminal measurements cannot take in dynamically allocated wires
            since they must be temporary.
            """),
    ):

        @qjit(capture=True)
        @qp.qnode(qp.device(backend, wires=1))
        def circuit():
            q = qp.allocate(1)
            return qp.probs(q)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
