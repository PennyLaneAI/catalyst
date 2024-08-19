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
"""Tests for mid-circuit measurements in Catalyst"""

from dataclasses import asdict
from functools import reduce
from typing import Iterable, Sequence

import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest
from jax.tree_util import tree_flatten
from pennylane.transforms.dynamic_one_shot import fill_in_value

import catalyst
from catalyst import CompileError, cond, measure, qjit

# TODO: add tests with other measurement processes (e.g. qml.sample, qml.probs, ...)

# pylint: disable=too-many-public-methods


class TestMidCircuitMeasurement:
    """Tests for mid-circuit behaviour."""

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

        assert not circuit(0.0)  # m will be equal to False

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

        assert jnp.allclose(circuit(0.0), 0)
        assert jnp.allclose(circuit(jnp.pi), 1)

    def test_mcm_method_deferred_error(self, backend):
        """Test that an error is raised if trying to execute with mcm_method="deferred"."""
        dev = qml.device(backend, wires=1)

        @qjit
        @qml.qnode(dev, mcm_method="deferred")
        def circuit(x):
            qml.RX(x, 0)
            measure(0)
            return qml.expval(qml.Z(0))

        with pytest.raises(
            ValueError, match="mcm_method='deferred' is not supported with Catalyst"
        ):
            _ = circuit(1.8)

    def test_mcm_method_one_shot_analytic_error(self, backend):
        """Test that an error is raised if using mcm_method="one-shot" without shots."""
        dev = qml.device(backend, wires=1, shots=None)

        @qjit
        @qml.qnode(dev, mcm_method="one-shot")
        def circuit(x):
            qml.RX(x, 0)
            measure(0)
            return qml.expval(qml.Z(0))

        with pytest.raises(
            ValueError, match="Cannot use the 'one-shot' method for mid-circuit measurements"
        ):
            _ = circuit(1.8)

    def test_single_branch_statistics_hw_like_error(self, backend):
        """Test that an error is raised if using `mcm_method="single-branch-statistics"` and
        `postselect_mode="hw-like"`"""
        dev = qml.device(backend, wires=1, shots=10)

        @qjit
        @qml.qnode(dev, mcm_method="single-branch-statistics", postselect_mode="hw-like")
        def circuit(x):
            qml.RX(x, 0)
            measure(0)
            return qml.expval(qml.Z(0))

        with pytest.raises(
            ValueError,
            match=("Cannot use postselect_mode='hw-like' with Catalyst when"),
        ):
            _ = circuit(1.8)

    @pytest.mark.parametrize("postselect_mode", [None, "fill-shots", "hw-like"])
    @pytest.mark.parametrize("mcm_method", [None, "one-shot", "single-branch-statistics"])
    def test_mcm_config_not_mutated(self, backend, postselect_mode, mcm_method):
        """Test that executing a QJIT-ed QNode does not mutate its mid-circuit measurements
        config."""
        if postselect_mode == "hw-like" and mcm_method == "single-branch-statistics":
            pytest.skip("Invalid MCM configuration")

        dev = qml.device(backend, wires=2, shots=10)

        original_config = qml.devices.MCMConfig(
            postselect_mode=postselect_mode, mcm_method=mcm_method
        )

        @qjit
        @qml.qnode(dev, **asdict(original_config))
        def circuit(x):
            qml.RX(x, 0)
            measure(0, postselect=1)
            return qml.expval(qml.PauliZ(0))

        _ = circuit(1.8)
        assert circuit.execute_kwargs["mcm_config"] == original_config

    @pytest.mark.parametrize("postselect_mode", [None, "fill-shots", "hw-like"])
    def test_default_mcm_method(self, backend, postselect_mode, mocker):
        """Test that the correct default mcm_method is chosen based on postselect_mode"""
        dev = qml.device(backend, wires=1, shots=10)

        @qjit
        @qml.qnode(dev, mcm_method=None, postselect_mode=postselect_mode)
        def circuit(x):
            qml.RX(x, 0)
            measure(0)
            return qml.expval(qml.Z(0))

        spy = mocker.spy(catalyst.qfunc, "dynamic_one_shot")
        _ = circuit(1.8)
        expected_call_count = 1 if postselect_mode == "hw-like" else 0
        assert spy.call_count == expected_call_count

    @pytest.mark.xfail(
        reason="Midcircuit measurements with sampling is unseeded and hence this test is flaky"
    )
    @pytest.mark.parametrize("postselect_mode", [None, "fill-shots", "hw-like"])
    @pytest.mark.parametrize("mcm_method", [None, "one-shot"])
    def test_mcm_method_with_dict_output(self, backend, postselect_mode, mcm_method):
        """Test that the correct default mcm_method is chosen based on postselect_mode"""
        dev = qml.device(backend, wires=1, shots=20)

        @qml.qjit
        @qml.qnode(dev, mcm_method=mcm_method, postselect_mode=postselect_mode)
        def circuit(x):
            qml.RX(x, wires=0)
            measure(0, postselect=1)
            return {"hi": qml.expval(qml.Z(0))}

        observed = circuit(0.9)
        expected = {"hi": jnp.array(-1.0, dtype=jnp.float64)}
        assert np.allclose(expected["hi"], observed["hi"])

    @pytest.mark.parametrize("postselect_mode", [None, "fill-shots", "hw-like"])
    @pytest.mark.parametrize("mcm_method", ["one-shot"])
    def test_mcm_method_with_count_mesurement(self, backend, postselect_mode, mcm_method):
        """Test that the correct default mcm_method is chosen based on postselect_mode"""
        dev = qml.device(backend, wires=1, shots=20)

        @qml.qjit
        @qml.qnode(dev, mcm_method=mcm_method, postselect_mode=postselect_mode)
        def circuit(x):
            qml.RX(x, wires=0)
            measure(0, postselect=1)
            return {"hi": qml.counts()}, {"bye": qml.expval(qml.Z(0))}, {"hi": qml.counts()}

        observed = circuit(0.9)
        expected = (
            {"hi": (jnp.array((0, 1), dtype=jnp.int64), jnp.array((0, 3), dtype=jnp.int64))},
            {"bye": jnp.array(-1, dtype=jnp.float64)},
            {"hi": (jnp.array((0, 1), dtype=jnp.int64), jnp.array((0, 3), dtype=jnp.int64))},
        )
        _, expected_shape = tree_flatten(expected)
        _, observed_shape = tree_flatten(observed)
        assert expected_shape == observed_shape

    @pytest.mark.parametrize("postselect_mode", [None, "fill-shots", "hw-like"])
    @pytest.mark.parametrize("mcm_method", [None, "one-shot"])
    def test_mcm_method_with_dict_output_used_measurements(
        self, backend, postselect_mode, mcm_method
    ):
        """Test that the correct default mcm_method is chosen based on postselect_mode"""
        dev = qml.device(backend, wires=1, shots=5)

        @qml.qjit
        @qml.qnode(dev, mcm_method=mcm_method, postselect_mode=postselect_mode)
        def circuit(x):
            qml.RX(x, wires=0)
            m_0 = measure(0, postselect=1)
            return (
                {"1": qml.probs(wires=[0])},
                {"2": qml.probs(wires=[0])},
                {"3": qml.probs(op=m_0)},
                {"4": qml.sample(op=m_0)},
            )

        observed = circuit(0.9)
        expected = (
            {"1": jnp.array((0, 1), dtype=jnp.float64)},
            {"2": jnp.array((0, 1), dtype=jnp.float64)},
            {"3": jnp.array((0, 1), dtype=jnp.float64)},
            {
                "4": jnp.array(
                    (-2147483648, -2147483648, -2147483648, 1, -2147483648), dtype=jnp.int64
                )
            },
        )
        _, expected_shape = tree_flatten(expected)
        _, observed_shape = tree_flatten(observed)
        assert expected_shape == observed_shape

    @pytest.mark.parametrize("mcm_method", [None, "single-branch-statistics", "one-shot"])
    def test_invalid_postselect_error(self, backend, mcm_method):
        """Test that an error is raised if postselecting on an invalid value"""
        dev = qml.device(backend, wires=1, shots=10)

        @qjit
        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit(x):
            qml.RX(x, 0)
            measure(0, postselect=-1)
            return qml.expval(qml.Z(0))

        with pytest.raises(TypeError, match="postselect must be '0' or '1'"):
            _ = circuit(1.8)


class TestDynamicOneShotIntegration:
    """Integration tests for QNodes using mcm_method="one-shot"/dynamic_one_shot."""

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize(
        "postselect, reset, expected",
        [
            (None, False, (1, 1)),
            (0, False, (0, 0)),
            (1, False, (1, 1)),
            (None, True, (1, 0)),
            (0, True, (0, 0)),
            (1, True, (1, 0)),
        ],
    )
    @pytest.mark.parametrize("postselect_mode", ["hw-like", "fill-shots"])
    def test_mcm_method_one_shot_with_single_shot(
        self, backend, postselect, reset, expected, postselect_mode
    ):
        """Test that the result is correct when using mcm_method="one-shot" with a single shot"""
        if postselect == 0 and postselect_mode == "fill-shots":
            pytest.xfail(
                reason="fill-shots not currently working when postselecting a zero probability state"
            )

        dev = qml.device(backend, wires=1, shots=1)

        @qjit
        @qml.qnode(dev, mcm_method="one-shot", postselect_mode=postselect_mode)
        def circuit(x):
            qml.RY(x, wires=0)
            m0 = measure(0, reset=reset, postselect=postselect)
            m1 = measure(0)
            return qml.sample(m0), qml.sample(m1)

        param = jnp.pi
        res = circuit(param)
        if postselect_mode == "hw-like" and postselect == 0:
            assert qml.math.allclose(res, fill_in_value)
        else:
            assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize("shots", [1, 10])
    def test_dynamic_one_shot_only_called_once(self, backend, shots, mocker):
        """Test that when using mcm_method="one-shot", dynamic_one_shot does not get
        called multiple times"""
        dev = qml.device(backend, wires=1, shots=shots)
        spy = mocker.spy(catalyst.qfunc, "dynamic_one_shot")

        @qjit
        @qml.qnode(dev, mcm_method="one-shot")
        def circuit(x):
            qml.RY(x, wires=0)
            measure(0)
            return qml.sample(wires=0)

        param = jnp.array(0.1)
        _ = circuit(param)

        assert spy.call_count == 1

    def test_dynamic_one_shot_unsupported_measurement(self, backend):
        """Test that circuits with unsupported measurements raise an error."""
        shots = 10
        dev = qml.device(backend, wires=1, shots=shots)
        param = np.pi / 4

        @qjit
        @qml.qnode(dev, mcm_method="one-shot")
        def func(x):
            qml.RX(x, wires=0)
            _ = measure(0)
            return qml.classical_shadow(wires=0)

        with pytest.raises(
            TypeError,
            match="Native mid-circuit measurement mode does not support",
        ):
            func(param)

    def test_dynamic_one_shot_unsupported_none_shots(self, backend):
        """Test that `dynamic_one_shot` raises when used with non-finite shots."""
        dev = qml.device(backend, wires=1, shots=None)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="dynamic_one_shot is only supported with finite shots.",
        ):

            @qjit
            @catalyst.qfunc.dynamic_one_shot
            @qml.qnode(dev)
            def _(x, y):
                qml.RX(x, wires=0)
                _ = measure(0)
                qml.RX(y, wires=0)
                return qml.probs(wires=0)

    def test_dynamic_one_shot_unsupported_broadcast(self, backend):
        """Test that `dynamic_one_shot` raises when used with parameter broadcasting."""
        shots = 10
        dev = qml.device(backend, wires=1, shots=shots)
        param = np.pi / 4 * jnp.ones(2)

        @qjit
        @qml.qnode(dev, mcm_method="one-shot")
        def func(x, y):
            qml.RX(x, wires=0)
            _ = measure(0)
            qml.RX(y, wires=0)
            return qml.probs(wires=0)

        with pytest.raises(
            ValueError,
            match="mcm_method='one-shot' is not compatible with broadcasting",
        ):
            func(param, param)

    @pytest.mark.parametrize("param, expected", [(0.0, 0.0), (jnp.pi, 1.0)])
    def test_dynamic_one_shot_with_sample_single(self, backend, param, expected):
        """Test that a measurement result can be returned with qml.sample and shots."""
        shots = 10
        dev = qml.device(backend, wires=1, shots=shots)

        @qjit
        @qml.qnode(dev, mcm_method="one-shot")
        def circuit(x):
            qml.RY(x, wires=0)
            m = measure(0)
            qml.PauliX(0)
            return qml.sample(m)

        result = circuit(param)
        assert result.shape == (shots,)
        assert jnp.allclose(result, expected)

    @pytest.mark.parametrize("debug", [False, True])
    def test_dynamic_one_shot_nested_qnodes(self, backend, debug):
        """Test that `dynamic_one_shot` handle nested calls correctly."""

        if not debug:
            pytest.xfail("Error in Catalyst Runtime: Cannot re-initialize an ACTIVE device")

        shots = 10
        dev = qml.device(backend, wires=2, shots=shots)

        @qml.qnode(dev)
        def inner():
            qml.PauliX(0)
            return qml.sample(wires=0)

        @qjit
        @qml.qnode(dev, mcm_method="one-shot")
        def outer():
            x = inner()
            if debug:
                catalyst.debug.print("Value of x = {x}", x=x)
            qml.RY(jnp.pi * qml.math.sum(x) / shots, wires=0)
            m0 = measure(0)
            return qml.sample(m0)

        result = outer()
        assert result.shape == (shots,)
        assert jnp.allclose(result, 1.0)

    @pytest.mark.xfail(
        reason="Midcircuit measurements with sampling is unseeded and hence this test is flaky"
    )
    @pytest.mark.parametrize("shots", [10000])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    @pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
    @pytest.mark.parametrize(
        "meas_obj", [qml.PauliZ(0), qml.Hadamard(0) @ qml.PauliZ(1), [0], [0, 1], "mcm"]
    )
    @pytest.mark.parametrize("postselect_mode", ["fill-shots", "hw-like"])
    # pylint: disable=too-many-arguments
    def test_dynamic_one_shot_several_mcms(
        self, backend, shots, postselect, measure_f, meas_obj, postselect_mode
    ):
        """Tests that Catalyst yields the same results as PennyLane's DefaultQubit for a simple
        circuit with a mid-circuit measurement."""
        if measure_f in (qml.counts, qml.probs, qml.sample) and (
            not isinstance(meas_obj, list) and not meas_obj == "mcm"
        ):
            pytest.skip("Can't use observables with counts, probs or sample")

        if measure_f in (qml.var, qml.expval) and (isinstance(meas_obj, list)):
            pytest.skip("Can't use wires/mcm lists with var or expval")

        dq = qml.device("default.qubit", shots=shots, seed=8237945)

        @qml.qnode(dq, postselect_mode=postselect_mode, mcm_method="deferred")
        def ref_func(x, y):
            qml.RX(x, 0)
            m0 = qml.measure(0)
            qml.RX(0.5 * x, 1)
            m1 = qml.measure(1, postselect=postselect)
            qml.cond(m0 & m1, qml.RY)(2.0 * y, 0)
            m2 = qml.measure(0)

            meas_key = "wires" if isinstance(meas_obj, list) else "op"
            meas_value = m2 if isinstance(meas_obj, str) else meas_obj
            kwargs = {meas_key: meas_value}
            if measure_f == qml.counts:
                kwargs["all_outcomes"] = True
            return measure_f(**kwargs)

        dev = qml.device(backend, wires=2, shots=shots)

        @qjit(seed=123456)
        @qml.qnode(dev, postselect_mode=postselect_mode, mcm_method="one-shot")
        def func(x, y):
            qml.RX(x, 0)
            m0 = measure(0)
            qml.RX(0.5 * x, 1)
            m1 = measure(1, postselect=postselect)

            @cond(m0 & m1)
            def cfun0():
                qml.RY(2.0 * y, 0)

            cfun0()
            m2 = measure(0)

            meas_key = "wires" if isinstance(meas_obj, list) else "op"
            meas_value = m2 if isinstance(meas_obj, str) else meas_obj
            kwargs = {meas_key: meas_value}
            return measure_f(**kwargs)

        if measure_f in (qml.expval,):
            params = jnp.pi / 3 * jnp.ones(2)
        else:
            params = jnp.pi / 2.1 * jnp.ones(2)

        if measure_f == qml.var and not isinstance(meas_obj, str):
            with pytest.raises(TypeError, match="qml.var\\(obs\\) cannot be returned when"):
                func(*params)
            return

        results0 = ref_func(*params)
        results1 = func(*params)

        if measure_f == qml.counts:

            def fname(x):
                return format(x, f"0{len(meas_obj)}b") if isinstance(meas_obj, list) else x

            results1 = {fname(int(state)): count for state, count in zip(*results1)}
        elif measure_f == qml.sample:
            results0 = sample_to_counts(results0, meas_obj)
            results1 = sample_to_counts(results1, meas_obj)
            measure_f = qml.counts

        validate_measurements(measure_f, shots, results1, results0)

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize("shots", [10000])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    @pytest.mark.parametrize("reset", [False, True])
    @pytest.mark.parametrize("postselect_mode", ["fill-shots", "hw-like"])
    def test_dynamic_one_shot_multiple_measurements(
        self, backend, shots, postselect, reset, postselect_mode
    ):
        """Tests that Catalyst yields the same results as PennyLane's DefaultQubit for a simple
        circuit with a mid-circuit measurement and several terminal measurements."""
        if backend == "lightning.kokkos":
            obs = qml.PauliZ(0)
        else:
            obs = qml.PauliY(0)

        dq = qml.device("default.qubit", shots=shots, seed=8237945)

        @qml.qnode(dq, postselect_mode=postselect_mode, mcm_method="deferred")
        def ref_func(x, y):
            qml.RX(x, 0)
            m0 = qml.measure(0)
            qml.RX(0.5 * x, 1)
            m1 = qml.measure(1, reset=reset, postselect=postselect)
            qml.cond(m0 & m1, qml.RY)(2.0 * y, 0)
            _ = qml.measure(0)

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
        @qml.qnode(dev, mcm_method="one-shot", postselect_mode=postselect_mode)
        def func(x, y):
            qml.RX(x, 0)
            m0 = measure(0)
            qml.RX(0.5 * x, 1)
            m1 = measure(1, reset=reset, postselect=postselect)

            @cond(m0 & m1)
            def cfun0():
                qml.RY(2.0 * y, 0)

            cfun0()
            _ = measure(0)

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

        meas_args = (
            "mcm",
            [1],
            [0, 1],
            "mcm",
            [1],
            [0, 1],
            "mcm",
            obs,
        )
        measures = (
            qml.expval,
            qml.probs,
            qml.probs,
            qml.probs,
            qml.sample,
            qml.sample,
            qml.sample,
            qml.expval,
        )

        params = jnp.pi / 3 * jnp.ones(2)
        results0 = ref_func(*params)
        results1 = func(*params)
        for meas_obj, m, r1, r0 in zip(meas_args, measures, results1, results0):
            if m == qml.sample:
                r0 = list(sample_to_counts(r0, meas_obj).values())
                r1 = list(sample_to_counts(r1, meas_obj).values())

            r1, r0 = qml.math.array(r1).ravel(), qml.math.array(r0).ravel()
            assert qml.math.allclose(r1, r0, atol=20, rtol=0.2)


def sample_to_counts(results, meas_obj):
    """Helper function to convert samples array to counts dictionary"""
    meas_key = "wires" if isinstance(meas_obj, list) else "op"
    meas_value = qml.measure(0) if isinstance(meas_obj, str) else meas_obj
    kwargs = {meas_key: meas_value, "all_outcomes": True}
    wires = (
        meas_obj
        if isinstance(meas_obj, list)
        else (qml.wires.Wires([0]) if isinstance(meas_obj, str) else meas_obj.wires)
    )
    results = (
        results.reshape((-1, 1)) if not isinstance(meas_obj, list) or len(meas_obj) < 2 else results
    )
    mask = qml.math.logical_not(qml.math.any(results == fill_in_value, axis=1))
    return qml.counts(**kwargs).process_samples(results[mask, :], wire_order=wires)


def validate_counts(shots, results1, results2, batch_size=None):
    """Compares two counts.

    If the results are ``Sequence``s, loop over entries.

    Fails if a key of ``results1`` is not found in ``results2``.
    Passes if counts are too low, chosen as ``100``.
    Otherwise, fails if counts differ by more than ``20`` plus 20 percent.
    """
    if isinstance(shots, Sequence):
        assert isinstance(results1, tuple)
        assert isinstance(results2, tuple)
        assert len(results1) == len(results2) == len(shots)
        for s, r1, r2 in zip(shots, results1, results2):
            validate_counts(s, r1, r2, batch_size=batch_size)
        return

    if batch_size is not None:
        assert isinstance(results1, Iterable)
        assert isinstance(results2, Iterable)
        assert len(results1) == len(results2) == batch_size
        for r1, r2 in zip(results1, results2):
            validate_counts(shots, r1, r2, batch_size=None)
        return

    for key1, val1 in results1.items():
        val2 = results2[key1]
        if abs(val1 + val2) > 100:
            assert np.allclose(val1, val2, atol=20, rtol=0.2)


def validate_expval(shots, results1, results2, batch_size=None):
    """Compares two expval, probs or var.

    If the results are ``Sequence``s, validate the average of items.

    If ``shots is None``, validate using ``np.allclose``'s default parameters.
    Otherwise, fails if the results do not match within ``0.01`` plus 20 percent.
    """
    if isinstance(shots, Sequence):
        assert isinstance(results1, tuple)
        assert isinstance(results2, tuple)
        assert len(results1) == len(results2) == len(shots)
        results1 = reduce(lambda x, y: x + y, results1) / len(results1)
        results2 = reduce(lambda x, y: x + y, results2) / len(results2)
        validate_expval(sum(shots), results1, results2, batch_size=batch_size)
        return

    if shots is None:
        assert np.allclose(results1, results2)
        return

    if batch_size is not None:
        assert len(results1) == len(results2) == batch_size
        for r1, r2 in zip(results1, results2):
            validate_expval(shots, r1, r2, batch_size=None)

    assert np.allclose(results1, results2, atol=0.01, rtol=0.2)


def validate_measurements(func, shots, results1, results2, batch_size=None):
    """Calls the correct validation function based on measurement type."""
    if func is qml.counts:
        validate_counts(shots, results1, results2, batch_size=batch_size)
        return

    validate_expval(shots, results1, results2, batch_size=batch_size)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
