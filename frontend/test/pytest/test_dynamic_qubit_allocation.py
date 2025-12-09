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

import re
import textwrap

import numpy as np
import pennylane as qml
import pytest
from jax import numpy as jnp

from catalyst import qjit
from catalyst.jax_primitives import subroutine
from catalyst.utils.exceptions import CompileError


@pytest.mark.usefixtures("use_capture")
def test_basic_dynamic_wire_alloc_plain_API(backend):
    """
    Test basic qml.allocate and qml.deallocate.
    """

    @qjit
    @qml.qnode(qml.device(backend, wires=3))
    def circuit():
        qml.X(1)  # |010>

        q = qml.allocate(1)  # |010> and |0>
        qml.X(q[0])  # |010> and |1>
        qml.CNOT(wires=[q[0], 2])  # |011> and |1>
        qml.deallocate(q[0])  # |011>

        return qml.probs(wires=[0, 1, 2])

    observed = circuit()

    expected = [0, 0, 0, 1, 0, 0, 0, 0]
    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
def test_basic_dynamic_wire_alloc_ctx_API(backend):
    """
    Test basic qml.allocate with context manager API.
    """

    @qjit
    @qml.qnode(qml.device(backend, wires=3))
    def circuit():
        qml.X(1)

        with qml.allocate(1) as q:
            qml.X(q[0])
            qml.CNOT(wires=[q[0], 2])

        return qml.probs(wires=[0, 1, 2])

    observed = circuit()

    expected = [0, 0, 0, 1, 0, 0, 0, 0]
    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
def test_measure(backend):
    """
    Test qml.allocate with qml.Measure ops.
    """

    @qjit(autograph=True)
    @qml.qnode(qml.device(backend, wires=1))
    def circuit():
        with qml.allocate(1) as q:
            qml.Hadamard(q[0])
            m = qml.measure(wires=q[0], postselect=1)

        if m:
            qml.X(0)

        return qml.probs(wires=[0])  # |1>

    observed = circuit()

    expected = [0, 1]
    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
def test_measure_with_reset(backend):
    """
    Test qml.allocate with qml.Measure ops with resetting.
    """

    @qjit(autograph=True)
    @qml.qnode(qml.device(backend, wires=1))
    def circuit():
        with qml.allocate(1) as q:
            qml.Hadamard(q[0])
            # measure 1 and reset q[0] to |0>
            m1 = qml.measure(wires=q[0], reset=True, postselect=1)
            # measure 0
            m0 = qml.measure(wires=q[0])

        if m0:  # should not be hit
            qml.RX(37.42, wires=[0])

        if m1:  # should be hit
            qml.X(wires=[0])

        return qml.probs(wires=[0])

    observed = circuit()

    expected = [0, 1]
    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
@pytest.mark.parametrize("ctrl_val, expected", [(False, [0, 1]), (True, [1, 0])])
def test_qml_ctrl(ctrl_val, expected, backend):
    """
    Test qml.allocate with qml.ctrl ops.
    """

    @qjit
    @qml.qnode(qml.device(backend, wires=1))
    def circuit():
        with qml.allocate(1) as q:
            qml.ctrl(qml.X, (q), control_values=ctrl_val)(wires=0)
        return qml.probs(wires=[0])

    observed = circuit()

    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
def test_QubitUnitary(backend):
    """
    Test qml.allocate with qml.QubitUnitary ops.
    """

    @qjit
    @qml.qnode(qml.device(backend, wires=1))
    def circuit():
        with qml.allocate(2) as qs:
            qml.QubitUnitary(jnp.identity(8), wires=[0, qs[0], qs[1]])
        return qml.probs(wires=[0])

    observed = circuit()

    expected = [1, 0]
    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
def test_StatePrep(backend):
    """
    Test qml.allocate with qml.StatePrep ops.
    """

    @qjit
    @qml.qnode(qml.device(backend, wires=1))
    def circuit():
        with qml.allocate(1) as q:
            qml.StatePrep(jnp.array([0, 0, 0, 1]), wires=[0, q[0]])  # |11>
        return qml.probs(wires=[0])  # |1>

    observed = circuit()

    expected = [0, 1]
    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
def test_BasisState(backend):
    """
    Test qml.allocate with qml.BasisState ops.
    """

    @qjit
    @qml.qnode(qml.device(backend, wires=1))
    def circuit():
        with qml.allocate(1) as q:
            qml.BasisState(jnp.array([1, 0]), wires=[q[0], 0])  # |10>
        return qml.probs(wires=[0])  # |0>

    observed = circuit()

    expected = [1, 0]
    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
@pytest.mark.parametrize("cond, expected", [(True, [0, 0, 1, 0]), (False, [0, 1, 0, 0])])
def test_dynamic_wire_alloc_cond(cond, expected, backend):
    """
    Test qml.allocate and qml.deallocate inside cond.
    """

    @qjit(autograph=True)
    @qml.qnode(qml.device(backend, wires=2))
    def circuit(c):
        if c:
            q = qml.allocate(1)[0]
            qml.X(wires=q)
            qml.CNOT(wires=[q, 0])
            qml.deallocate(q)
        else:
            q = qml.allocate(1)[0]
            qml.X(wires=q)
            qml.CNOT(wires=[q, 1])
            qml.deallocate(q)

        return qml.probs(wires=[0, 1])

    observed = circuit(cond)

    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
@pytest.mark.parametrize("cond, expected", [(True, [0, 1, 0, 0]), (False, [1, 0, 0, 0])])
def test_dynamic_wire_alloc_cond_outside(cond, expected, backend):
    """
    Test passing dynamically allocated wires into a cond.
    """

    @qjit(autograph=True)
    @qml.qnode(qml.device(backend, wires=2))
    def circuit(c):
        with qml.allocate(1) as q1:
            with qml.allocate(1) as q2:
                qml.X(q1[0])
                if c:
                    qml.CNOT(wires=[q1[0], 1])  # |01>
                else:
                    qml.CNOT(wires=[q2[0], 1])  # |00>

        return qml.probs(wires=[0, 1])

    observed = circuit(cond)

    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
@pytest.mark.parametrize(
    "num_iter, expected", [(3, [0, 0, 1, 0, 0, 0, 0, 0]), (4, [1, 0, 0, 0, 0, 0, 0, 0])]
)
def test_dynamic_wire_alloc_forloop(num_iter, expected, backend):
    """
    Test qml.allocate and qml.deallocate inside for loop.
    """

    @qjit(autograph=True)
    @qml.qnode(qml.device(backend, wires=3))
    def circuit(N):
        for _ in range(N):
            q = qml.allocate(1)[0]
            qml.X(wires=q)
            qml.CNOT(wires=[q, 1])
            qml.deallocate(q)

        return qml.probs(wires=[0, 1, 2])

    observed = circuit(num_iter)

    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
def test_dynamic_wire_alloc_forloop_outside(backend):
    """
    Test passing dynamically allocated wires into a for loop.
    """

    @qjit(autograph=True)
    @qml.qnode(qml.device(backend, wires=1))
    def circuit():
        with qml.allocate(1) as q:
            qml.X(wires=q[0])
            for _ in range(3):
                qml.CNOT(wires=[q[0], 0])

        return qml.probs(wires=[0])

    observed = circuit()
    expected = [0, 1]

    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
def test_dynamic_wire_alloc_forloop_outside_multiple_regs(backend):
    """
    Test using multiple dynamically allocated registers from inside for loop.
    """

    @qjit(autograph=True)
    @qml.qnode(qml.device(backend, wires=1))
    def circuit():
        with qml.allocate(1) as q1:
            with qml.allocate(1) as q2:
                for _ in range(3):
                    qml.CNOT(wires=[q1[0], 0])
                    qml.CNOT(wires=[q2[0], 0])

        return qml.probs(wires=[0])

    observed = circuit()
    expected = [1, 0]

    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
@pytest.mark.parametrize(
    "num_iter, expected", [(3, [0, 0, 1, 0, 0, 0, 0, 0]), (4, [1, 0, 0, 0, 0, 0, 0, 0])]
)
def test_dynamic_wire_alloc_whileloop(num_iter, expected, backend):
    """
    Test qml.allocate and qml.deallocate inside while loop.
    """

    @qjit(autograph=True)
    @qml.qnode(qml.device(backend, wires=3))
    def circuit(N):
        i = 0
        while i < N:
            q = qml.allocate(1)[0]
            qml.X(wires=q)
            qml.CNOT(wires=[q, 1])
            qml.deallocate(q)
            i += 1

        return qml.probs(wires=[0, 1, 2])

    observed = circuit(num_iter)

    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
@pytest.mark.parametrize("num_iter, expected", [(3, [0, 1, 0, 0]), (4, [1, 0, 0, 0])])
def test_dynamic_wire_alloc_whileloop_outside(num_iter, expected, backend):
    """
    Test passing dynamically allocated wires into a while loop.
    """

    @qjit(autograph=True)
    @qml.qnode(qml.device(backend, wires=2))
    def circuit(N):
        i = 0
        with qml.allocate(1) as q1:
            with qml.allocate(1) as q2:
                qml.X(q1[0])
                while i < N:
                    qml.CNOT(wires=[q1[0], 1])
                    qml.CNOT(wires=[q2[0], 1])
                    i += 1

        return qml.probs(wires=[0, 1])

    observed = circuit(num_iter)

    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
@pytest.mark.parametrize("flip_again, expected", [(True, [1, 0]), (False, [0, 1])])
def test_subroutine(flip_again, expected, backend):
    """
    Test passing dynamically allocated wires into a subroutine.
    """

    @subroutine
    def flip(w):
        qml.X(w)
        qml.CNOT(wires=[w, 0])

    @qjit
    @qml.qnode(qml.device(backend, wires=1))
    def circuit():
        with qml.allocate(1) as q1:
            with qml.allocate(1) as q2:
                flip(q1[0])
                if flip_again:
                    flip(q2[0])
        return qml.probs(wires=[0])

    observed = circuit()
    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
def test_subroutine_multiple_args(backend):
    """
    Test passing dynamically allocated wires into a subroutine with multiple arguments.
    """

    @subroutine
    def flip(w1, w2, theta):
        qml.X(w1)
        qml.X(w2)
        qml.ctrl(qml.RX, (w1, w2))(theta, wires=0)

    @qjit
    @qml.qnode(qml.device(backend, wires=1))
    def circuit():
        with qml.allocate(1) as q1:
            with qml.allocate(2) as q2:
                flip(q1[0], q2[1], jnp.pi)
        return qml.probs(wires=[0])

    observed = circuit()
    expected = [0, 1]
    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
def test_subroutine_and_loop(backend):
    """
    Test passing dynamically allocated wires into a subroutine with loops.
    """

    @subroutine
    def flip(wire, theta):
        """
        Apply three X gates to the input wire, effectively NOT-ing it.
        """

        @qml.for_loop(0, 3, 1)
        def loop(i, _theta):  # pylint: disable=unused-argument
            qml.X(wire)
            return jnp.sin(_theta)

        _ = loop(theta)

    @qjit
    @qml.qnode(qml.device(backend, wires=1))
    def circuit():
        with qml.allocate(1) as q1:
            flip(q1[0], 0.0)
            qml.CNOT(wires=[q1[0], 0])
        return qml.probs(wires=[0])

    observed = circuit()
    expected = [0, 1]
    assert np.allclose(expected, observed)


@pytest.mark.usefixtures("use_capture")
def test_subroutine_and_loop_multiple_args(backend):
    """
    Test passing dynamically allocated wires into a subroutine with loops and multiple arguments.
    """

    @subroutine
    def flip(w1, w2, w3, theta):
        @qml.for_loop(0, 2, 1)
        def loop(i, _theta):  # pylint: disable=unused-argument
            qml.X(w1)
            qml.Y(w2)
            qml.Z(w3)
            qml.ctrl(qml.RX, (w1, w2))(_theta, wires=0)
            qml.ctrl(qml.RY, (w2, w3))(_theta, wires=1)
            return jnp.sin(_theta)

        _ = loop(theta)

    @qjit
    @qml.qnode(qml.device(backend, wires=2))
    def circuit():
        with qml.allocate(2) as q1:
            with qml.allocate(3) as q2:
                flip(q1[0], q1[1], q2[2], 1.23)

        return qml.probs(wires=[0, 1])

    @qml.qnode(qml.device("default.qubit", wires=7))
    def ref_circuit():
        for _ in range(2):
            qml.X(0)
            qml.Y(1)
            qml.Z(2)
            qml.ctrl(qml.RX, (0, 1))(1.23, wires=3)
            qml.ctrl(qml.RY, (1, 2))(1.23, wires=4)

        return qml.probs(wires=[3, 4])

    assert np.allclose(circuit(), ref_circuit())


def test_no_capture(backend):
    """
    Test error message when used without capture.
    """
    with pytest.raises(
        CompileError,
        match=re.escape("qml.allocate() is only supported with program capture enabled."),
    ):

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit():
            with qml.allocate(1) as _:
                pass
            return qml.probs(wires=[0])


@pytest.mark.usefixtures("use_capture")
def test_use_after_free(backend):
    """
    Test error message when used after free.
    """

    with pytest.raises(
        CompileError,
        match="Deallocated qubits cannot be used, but used in Hadamard.",
    ):

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit():
            with qml.allocate(1) as q:
                qml.X(q[0])
            qml.Hadamard(q[0])
            return qml.probs(wires=[0])


@pytest.mark.usefixtures("use_capture")
def test_terminal_MP_all_wires(backend):
    """
    Test error message when used with terminal measurements on all wires.
    """

    with pytest.raises(
        CompileError,
        match=textwrap.dedent(
            """
            Terminal measurements must take in an explicit list of wires when
            dynamically allocated wires are present in the program.
            """
        ),
    ):

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit():
            with qml.allocate(1) as _:
                pass
            return qml.probs()


@pytest.mark.usefixtures("use_capture")
def test_terminal_MP_dynamic_wires(backend):
    """
    Test error message when used with terminal measurements on dynamic wires.
    """

    with pytest.raises(
        CompileError,
        match=textwrap.dedent(
            """
            Terminal measurements cannot take in dynamically allocated wires
            since they must be temporary.
            """
        ),
    ):

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit():
            q = qml.allocate(1)
            return qml.probs(q)


@pytest.mark.usefixtures("use_capture")
def test_unsupported_adjoint(backend):
    """
    Test that an error is raised when a dynamically allocated wire is passed into a adjoint.
    """

    with pytest.raises(
        NotImplementedError,
        match="Dynamically allocated wires cannot be used in quantum adjoints yet.",
    ):

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit():
            with qml.allocate(1) as q:
                qml.adjoint(qml.X)(q[0])
            return qml.probs(wires=[0, 1])


if __name__ == "__main__":
    pytest.main(["-x", __file__])
