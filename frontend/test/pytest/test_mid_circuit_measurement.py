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

import jax.numpy as jnp
import pennylane as qml
import pytest
from catalyst import CompileError, cond, dynamic_one_shot, measure, qjit
from conftest import validate_measurements
from pennylane.transforms.dynamic_one_shot import fill_in_value

# TODO: add tests with other measurement processes (e.g. qml.sample, qml.probs, ...)


class TestMidCircuitMeasurement:
    def test_pl_measure(self, backend):
        """Test PL measure."""

        def circuit():
            return qml.measure(0)

        with pytest.raises(CompileError, match="Must use 'measure' from Catalyst"):
            qjit(qml.qnode(qml.device(backend, wires=1))(circuit))()

    def test_measure_outside_qjit(self):
        """Test measure outside qjit."""

        def circuit():
            return measure(0)

        with pytest.raises(CompileError, match="can only be used from within @qjit"):
            circuit()

    def test_measure_outside_qnode(self):
        """Test measure outside qnode."""

        def circuit():
            return measure(0)

        with pytest.raises(CompileError, match="can only be used from within a qml.qnode"):
            qjit(circuit)()

    def test_invalid_arguments(self, backend):
        """Test too many arguments to the wires parameter."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit():
            qml.RX(0.0, wires=0)
            m = measure(wires=[1, 2])
            return m

        with pytest.raises(
            TypeError, match="Only one element is supported for the 'wires' parameter"
        ):
            qjit(circuit)()

    def test_invalid_arguments2(self, backend):
        """Test too large array for the wires parameter."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit():
            qml.RX(0.0, wires=0)
            m = measure(wires=jnp.array([1, 2]))
            return m

        with pytest.raises(TypeError, match="Measure is only supported on 1 qubit"):
            qjit(circuit)()

    def test_basic(self, backend):
        """Test measure (basic)."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):
            qml.RX(x, wires=0)
            m = measure(wires=0)
            return m

        assert circuit(jnp.pi)  # m will be equal to True if wire 0 is measured in 1 state

    def test_scalar_array_wire(self, backend):
        """Test a scalar array wire."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(w):
            qml.PauliX(0)
            m = measure(wires=w)
            return m

        assert circuit(jnp.array(0)) == 1

    def test_1element_array_wire(self, backend):
        """Test a 1D single-element array wire."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(w):
            qml.PauliX(0)
            m = measure(wires=w)
            return m

        assert circuit(jnp.array([0])) == 1

    def test_more_complex(self, backend):
        """Test measure (more complex)."""

        @qjit
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(x: float):
            qml.RX(x, wires=0)
            m1 = measure(wires=0)
            maybe_pi = m1 * jnp.pi
            qml.RX(maybe_pi, wires=1)
            m2 = measure(wires=1)
            return m2

        assert circuit(jnp.pi)  # m will be equal to True if wire 0 is measured in 1 state
        assert not circuit(0.0)

    def test_with_postselect_zero(self, backend):
        """Test measure (postselect = 0)."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):
            qml.RX(x, wires=0)
            m = measure(wires=0, postselect=0)
            return m

        assert not circuit(jnp.pi)  # m will be equal to False

    def test_with_postselect_one(self, backend):
        """Test measure (postselect = 1)."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):
            qml.RX(x, wires=0)
            m = measure(wires=0, postselect=1)
            return m

        assert circuit(jnp.pi)  # m will be equal to True

    def test_with_reset_false(self, backend):
        """Test measure (reset = False)."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit():
            qml.Hadamard(wires=0)
            m1 = measure(wires=0, reset=False, postselect=1)
            m2 = measure(wires=0)
            return m1 == m2

        assert circuit()  # both measures are the same

    def test_with_reset_true(self, backend):
        """Test measure (reset = True)."""

        @qjit
        @qml.qnode(qml.device(backend, wires=1))
        def circuit():
            qml.Hadamard(wires=0)
            m1 = measure(wires=0, reset=True, postselect=1)
            m2 = measure(wires=0)
            return m1 != m2

        assert circuit()  # measures are different

    def test_return_mcm_with_sample_single(self, backend):
        """Test that a measurement result can be returned with qml.sample and shots."""

        dev = qml.device(backend, wires=1, shots=1)

        @qjit
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x, wires=0)
            m = measure(0)
            qml.PauliX(0)
            return qml.sample(m)

        assert circuit(0.0) == 0
        assert circuit(jnp.pi) == 1

    @pytest.mark.xfail(reason="not yet implemented, coming in one-shot support")
    def test_return_mcm_with_sample_multiple(self, backend):
        """Test that a measurement result can be returned with qml.sample and shots."""

        dev = qml.device(backend, wires=1, shots=10)

        @qjit
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x, wires=0)
            m = measure(0)
            qml.PauliX(0)
            return qml.sample(m)

        assert circuit(0.0) == [0] * 10
        assert circuit(jnp.pi) == [1] * 10

    def test_return_mcm_with_sample_single(self, backend):
        """Test that a measurement result can be returned with qml.sample and shots."""

        dev = qml.device(backend, wires=1, shots=10)

        @qjit
        @dynamic_one_shot
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x, wires=0)
            m = measure(0)
            qml.PauliX(0)
            return qml.sample(m)

        assert jnp.allclose(circuit(0.0), 0)
        assert jnp.allclose(circuit(jnp.pi), 1)

    @pytest.mark.parametrize("shots", [5000])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    @pytest.mark.parametrize("reset", [False, True])
    @pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
    @pytest.mark.parametrize(
        "meas_obj", [qml.PauliZ(0), qml.Hadamard(0) @ qml.PauliZ(1), [0], [0, 1], "mcm"]
    )
    def test_simple_mcm(self, backend, shots, postselect, reset, measure_f, meas_obj):
        """Tests that Catalyst yields the same results as PennyLane's DefaultQubit for a simple
        circuit with a mid-circuit measurement."""
        if measure_f in (qml.counts, qml.probs, qml.sample) and (
            not isinstance(meas_obj, list) and not meas_obj == "mcm"
        ):
            pytest.skip("Can't use observables with counts, probs or sample")

        if measure_f in (qml.var, qml.expval) and (isinstance(meas_obj, list)):
            pytest.skip("Can't use wires/mcm lists with var or expval")

        if measure_f == qml.var and (not isinstance(meas_obj, list) and not meas_obj == "mcm"):
            pytest.xfail("isa<UnrealizedConversionCastOp>")

        dq = qml.device("default.qubit", shots=shots)

        @qml.qnode(dq)
        def ref_func(x, y):
            qml.RX(x, wires=0)
            m0 = qml.measure(0, reset=reset, postselect=postselect)
            qml.cond(m0, qml.RY)(y, wires=1)

            meas_key = "wires" if isinstance(meas_obj, list) else "op"
            meas_value = m0 if isinstance(meas_obj, str) else meas_obj
            kwargs = {meas_key: meas_value}
            if measure_f == qml.counts:
                kwargs["all_outcomes"] = True
            return measure_f(**kwargs)

        if postselect is None:
            ref_func = qml.defer_measurements(ref_func)
        else:
            ref_func = qml.dynamic_one_shot(ref_func)

        dev = qml.device(backend, wires=2, shots=shots)

        @qjit
        @dynamic_one_shot
        @qml.qnode(dev)
        def func(x, y):
            qml.RX(x, wires=0)
            m0 = measure(0, reset=reset, postselect=postselect)

            @cond(m0 == 1)
            def ansatz():
                qml.RY(y, wires=1)

            ansatz()

            meas_key = "wires" if isinstance(meas_obj, list) else "op"
            meas_value = m0 if isinstance(meas_obj, str) else meas_obj
            kwargs = {meas_key: meas_value}
            return measure_f(**kwargs)

        params = jnp.pi / 4 * jnp.ones(2)
        results0 = ref_func(*params)
        results1 = func(*params)
        if measure_f == qml.counts and isinstance(meas_obj, list):
            results1 = {
                format(int(state), f"0{len(meas_obj)}b"): count for state, count in zip(*results1)
            }
        if measure_f == qml.sample:
            results0 = results0[results0 != fill_in_value]
            results1 = results1[results1 != fill_in_value]
        validate_measurements(measure_f, shots, results1, results0)

    @pytest.mark.parametrize("shots", [5000])
    @pytest.mark.parametrize("postselect", [None])
    @pytest.mark.parametrize("reset", [False, True])
    def test_mcm_multiple_measurements(self, backend, shots, postselect, reset):
        """Tests that Catalyst yields the same results as PennyLane's DefaultQubit for a simple
        circuit with a mid-circuit measurement and several terminal measurements."""
        obs = qml.PauliY(0)

        dq = qml.device("default.qubit", shots=shots)

        @qml.defer_measurements
        @qml.qnode(dq)
        def ref_func(x, y):
            qml.RX(x, wires=0)
            m0 = qml.measure(0, reset=reset, postselect=postselect)
            qml.cond(m0, qml.RY)(y, wires=1)

            return (
                qml.expval(op=m0),
                qml.probs(wires=[1]),
                qml.probs(wires=[0, 1]),
                qml.probs(op=m0),
                qml.sample(wires=[1]),
                qml.sample(wires=[0, 1]),
                qml.sample(op=m0),
                qml.expval(obs),
            )

        dev = qml.device(backend, wires=2, shots=shots)

        @qjit
        @dynamic_one_shot
        @qml.qnode(dev)
        def func(x, y):
            qml.RX(x, wires=0)
            m0 = measure(0, reset=reset, postselect=postselect)

            @cond(m0 == 1)
            def ansatz():
                qml.RY(y, wires=1)

            ansatz()

            return (
                qml.expval(op=m0),
                qml.probs(wires=[1]),
                qml.probs(wires=[0, 1]),
                qml.probs(op=m0),
                qml.sample(wires=[1]),
                qml.sample(wires=[0, 1]),
                qml.sample(op=m0),
                qml.expval(obs),
            )

        params = jnp.pi / 4 * jnp.ones(2)
        results0 = ref_func(*params)
        results1 = func(*params)
        for r1, r0 in zip(results1, results0):
            r1, r0 = qml.math.array(r1).ravel(), qml.math.array(r0).ravel()
            qml.math.allclose(r1, r0)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
