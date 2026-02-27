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
from functools import partial, reduce
from typing import Iterable, Sequence

import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest
from jax.tree_util import tree_flatten
from pennylane import exceptions, measure
from pennylane.transforms.dynamic_one_shot import fill_in_value

import catalyst
from catalyst import CompileError, cond, grad
from catalyst import jvp as C_jvp
from catalyst import qjit, value_and_grad
from catalyst import vjp as C_vjp

# TODO: add tests with other measurement processes (e.g. qml.sample, qml.probs, ...)

# pylint: disable=too-many-lines,too-many-public-methods


class TestMidCircuitMeasurement:
    """Tests for mid-circuit behaviour."""

    @pytest.mark.old_frontend
    def test_measure_outside_qjit(self):
        """Test measure outside qjit."""

        def circuit():
            return catalyst.measure(0)

        with pytest.raises(CompileError, match="can only be used from within @qjit"):
            circuit()

    # Capture gap: capture path raises tracing/lowering errors (e.g., invalid JAX type or missing measure lowering) before expected validation error.
    # Classification: Catalyst integration gap. Fix: preserve user-facing measure validation order under capture-enabled qjit.
    @pytest.mark.capture_todo
    def test_measure_outside_qnode(self, capture_mode):
        """Test measure outside qnode."""

        def circuit():
            return measure(0)

        with pytest.raises(CompileError, match="can only be used from within a qml.qnode"):
            qjit(circuit, capture=capture_mode)()

    # Capture gap: capture path raises tracing/lowering errors (e.g., invalid JAX type or missing measure lowering) before expected validation error.
    # Classification: Catalyst integration gap. Fix: preserve user-facing measure validation order under capture-enabled qjit.
    @pytest.mark.capture_todo
    def test_invalid_arguments(self, backend, capture_mode):
        """Test too many arguments to the wires parameter."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit():
            qml.RX(0.0, wires=0)
            m = measure(wires=[1, 2])
            return m

        with pytest.raises(
            TypeError, match="Only one element is supported for the 'wires' parameter"
        ):
            qjit(circuit, capture=capture_mode)()

    # Capture gap: capture path raises tracing/lowering errors (e.g., invalid JAX type or missing measure lowering) before expected validation error.
    # Classification: Catalyst integration gap. Fix: preserve user-facing measure validation order under capture-enabled qjit.
    @pytest.mark.capture_todo
    def test_invalid_arguments2(self, backend, capture_mode):
        """Test too large array for the wires parameter."""

        @qml.qnode(qml.device(backend, wires=2))
        def circuit():
            qml.RX(0.0, wires=0)
            m = measure(wires=jnp.array([1, 2]))
            return m

        with pytest.raises(TypeError, match="Measure is only supported on 1 qubit"):
            qjit(circuit, capture=capture_mode)()

    @pytest.mark.capture_todo  # Capture gap: returning MCM-derived sample(op=m) still raises "Measurements of mcms are not yet supported".
    # Fix direction: add program-capture execution support for MCM-valued measurement processes.
    def test_basic(self, backend, capture_mode):
        """Test measure (basic)."""

        @qjit(capture=capture_mode)
        @qml.set_shots(1)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):
            qml.RX(x, wires=0)
            m = measure(wires=0)
            return qml.sample(m)

        # Rewrite for capture compatibility: avoid returning raw MCM from QNode.
        assert circuit(jnp.pi)[0] == 1
        assert circuit(0.0)[0] == 0

    # Capture gap: MCM capture execution still unsupported (e.g., "Measurements of mcms are not yet supported").
    # Classification: missing PL feature/integration gap. Fix: add capture-time lowering/execution for MCM-derived measurements.
    @pytest.mark.capture_todo
    def test_scalar_array_wire(self, backend, capture_mode):
        """Test a scalar array wire."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(w):
            qml.PauliX(0)
            m = measure(wires=w)
            return m

        assert circuit(jnp.array(0)) == 1

    # Capture gap: MCM capture execution still unsupported (e.g., "Measurements of mcms are not yet supported").
    # Classification: missing PL feature/integration gap. Fix: add capture-time lowering/execution for MCM-derived measurements.
    @pytest.mark.capture_todo
    def test_1element_array_wire(self, backend, capture_mode):
        """Test a 1D single-element array wire."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=2))
        def circuit(w):
            qml.PauliX(0)
            m = measure(wires=w)
            return m

        assert circuit(jnp.array([0])) == 1

    # Capture gap: MCM capture execution still unsupported (e.g., "Measurements of mcms are not yet supported").
    # Classification: missing PL feature/integration gap. Fix: add capture-time lowering/execution for MCM-derived measurements.
    @pytest.mark.capture_todo
    def test_more_complex(self, backend, capture_mode):
        """Test measure (more complex)."""

        @qjit(capture=capture_mode)
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

    # Capture gap: MCM capture execution still unsupported (e.g., "Measurements of mcms are not yet supported").
    # Classification: missing PL feature/integration gap. Fix: add capture-time lowering/execution for MCM-derived measurements.
    @pytest.mark.capture_todo
    def test_with_postselect_zero(self, backend, capture_mode):
        """Test measure (postselect = 0)."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):
            qml.RX(x, wires=0)
            m = measure(wires=0, postselect=0)
            return m

        assert not circuit(0.0)  # m will be equal to False

    # Capture gap: MCM capture execution still unsupported (e.g., "Measurements of mcms are not yet supported").
    # Classification: missing PL feature/integration gap. Fix: add capture-time lowering/execution for MCM-derived measurements.
    @pytest.mark.capture_todo
    def test_with_postselect_one(self, backend, capture_mode):
        """Test measure (postselect = 1)."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit(x: float):
            qml.RX(x, wires=0)
            m = measure(wires=0, postselect=1)
            return m

        assert circuit(jnp.pi)  # m will be equal to True

    # Capture gap: MCM capture execution still unsupported (e.g., "Measurements of mcms are not yet supported").
    # Classification: missing PL feature/integration gap. Fix: add capture-time lowering/execution for MCM-derived measurements.
    @pytest.mark.capture_todo
    def test_with_reset_false(self, backend, capture_mode):
        """Test measure (reset = False)."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit():
            qml.Hadamard(wires=0)
            m1 = measure(wires=0, reset=False, postselect=1)
            m2 = measure(wires=0)
            return m1 == m2

        assert circuit()  # both measures are the same

    # Capture gap: MCM capture execution still unsupported (e.g., "Measurements of mcms are not yet supported").
    # Classification: missing PL feature/integration gap. Fix: add capture-time lowering/execution for MCM-derived measurements.
    @pytest.mark.capture_todo
    def test_with_reset_true(self, backend, capture_mode):
        """Test measure (reset = True)."""

        @qjit(capture=capture_mode)
        @qml.qnode(qml.device(backend, wires=1))
        def circuit():
            qml.Hadamard(wires=0)
            m1 = measure(wires=0, reset=True, postselect=1)
            m2 = measure(wires=0)
            return m1 != m2

        assert circuit()  # measures are different

    # Capture gap: MCM capture execution still unsupported (e.g., "Measurements of mcms are not yet supported").
    # Classification: missing PL feature/integration gap. Fix: add capture-time lowering/execution for MCM-derived measurements.
    @pytest.mark.capture_todo
    def test_return_mcm_with_sample_single(self, backend, capture_mode):
        """Test that a measurement result can be returned with qml.sample and shots."""

        dev = qml.device(backend, wires=1)

        @qjit(capture=capture_mode)
        @qml.set_shots(1)
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x, wires=0)
            m = measure(0)
            qml.PauliX(0)
            return qml.sample(m)

        assert circuit(0.0) == 0
        assert circuit(jnp.pi) == 1

    # Capture gap: MCM capture execution still unsupported (e.g., "Measurements of mcms are not yet supported").
    # Classification: missing PL feature/integration gap. Fix: add capture-time lowering/execution for MCM-derived measurements.
    @pytest.mark.capture_todo
    def test_return_mcm_with_sample_multiple(self, backend, capture_mode):
        """Test that a measurement result can be returned with qml.sample and shots."""

        dev = qml.device(backend, wires=1)

        @qjit(capture=capture_mode)
        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x, wires=0)
            m = measure(0)
            qml.PauliX(0)
            return qml.sample(m)

        assert jnp.allclose(circuit(0.0), 0)
        assert jnp.allclose(circuit(jnp.pi), 1)

    # Capture gap: MCM capture execution still unsupported (e.g., "Measurements of mcms are not yet supported").
    # Classification: missing PL feature/integration gap. Fix: add capture-time lowering/execution for MCM-derived measurements.
    @pytest.mark.capture_todo
    def test_mcm_method_deferred_error(self, backend, capture_mode):
        """Test that an error is raised if trying to execute with mcm_method="deferred"."""
        dev = qml.device(backend, wires=1)

        @qjit(capture=capture_mode)
        @qml.qnode(dev, mcm_method="deferred")
        def circuit(x):
            qml.RX(x, 0)
            measure(0)
            return qml.expval(qml.Z(0))

        with pytest.raises(
            ValueError, match="mcm_method='deferred' is not supported with Catalyst"
        ):
            _ = circuit(1.8)

    # Capture gap: dynamic-one-shot path raises NotImplementedError for captured MCM-valued measurements.
    # Classification: missing PL feature. Fix: implement capture execution support for dynamic_one_shot measurement processes.
    @pytest.mark.capture_todo
    def test_mcm_method_one_shot_analytic_error(self, backend, capture_mode):
        """Test that an error is raised if using mcm_method="one-shot" without shots."""
        dev = qml.device(backend, wires=1)

        @qjit(capture=capture_mode)
        @qml.set_shots(None)
        @qml.qnode(dev, mcm_method="one-shot")
        def circuit(x):
            qml.RX(x, 0)
            measure(0)
            return qml.expval(qml.Z(0))

        with pytest.raises(
            ValueError, match="mcm_method='one-shot' is not supported in analytic shot mode"
        ):
            _ = circuit(1.8)

    # Capture gap: MCM capture execution still unsupported (e.g., "Measurements of mcms are not yet supported").
    # Classification: missing PL feature/integration gap. Fix: add capture-time lowering/execution for MCM-derived measurements.
    @pytest.mark.capture_todo
    def test_single_branch_statistics_hw_like_error(self, backend, capture_mode):
        """Test that an error is raised if using `mcm_method="single-branch-statistics"` and
        `postselect_mode="hw-like"`"""
        dev = qml.device(backend, wires=1)

        @qjit(capture=capture_mode)
        @qml.set_shots(10)
        @qml.qnode(dev, mcm_method="single-branch-statistics", postselect_mode="hw-like")
        def circuit(x):
            qml.RX(x, 0)
            measure(0)
            return qml.expval(qml.Z(0))

        with pytest.raises(
            ValueError,
            match=("'hw-like' post-selection requires mcm_method='one-shot'"),
        ):
            _ = circuit(1.8)

    @pytest.mark.parametrize("postselect_mode", [None, "fill-shots", "hw-like"])
    @pytest.mark.parametrize("mcm_method", [None, "one-shot", "single-branch-statistics"])
    def test_mcm_config_not_mutated(self, backend, postselect_mode, mcm_method, capture_mode):
        """Test that executing a QJIT-ed QNode does not mutate its mid-circuit measurements
        config."""
        if postselect_mode == "hw-like" and mcm_method == "single-branch-statistics":
            pytest.skip("Invalid MCM configuration")

        dev = qml.device(backend, wires=2)

        original_config = qml.devices.MCMConfig(
            postselect_mode=postselect_mode, mcm_method=mcm_method
        )

        @qjit(capture=capture_mode)
        @qml.set_shots(10)
        @qml.qnode(dev, **asdict(original_config))
        def circuit(x):
            qml.RX(x, 0)
            measure(0, postselect=1)
            return qml.expval(qml.PauliZ(0))

        _ = circuit(1.8)
        assert circuit.execute_kwargs["postselect_mode"] == original_config.postselect_mode
        assert circuit.execute_kwargs["mcm_method"] == original_config.mcm_method

    # Capture gap: MCM capture execution still unsupported (e.g., "Measurements of mcms are not yet supported").
    # Classification: missing PL feature/integration gap. Fix: add capture-time lowering/execution for MCM-derived measurements.
    @pytest.mark.capture_todo
    @pytest.mark.parametrize("postselect_mode", [None, "fill-shots", "hw-like"])
    def test_default_mcm_method(self, backend, postselect_mode, mocker, capture_mode):
        """Test that the correct default mcm_method is chosen based on postselect_mode"""
        dev = qml.device(backend, wires=1)

        @qjit(capture=capture_mode)
        @qml.set_shots(10)
        @qml.qnode(dev, mcm_method=None, postselect_mode=postselect_mode)
        def circuit(x):
            qml.RX(x, 0)
            measure(0)
            return qml.expval(qml.Z(0))

        spy = mocker.spy(catalyst.qfunc, "dynamic_one_shot")
        _ = circuit(1.8)
        assert spy.call_count == 1

    @pytest.mark.xfail(
        reason="Midcircuit measurements with sampling is unseeded and hence this test is flaky",
        strict=False,
    )
    @pytest.mark.parametrize("postselect_mode", [None, "fill-shots", "hw-like"])
    @pytest.mark.parametrize("mcm_method", [None, "one-shot"])
    def test_mcm_method_with_dict_output(self, backend, postselect_mode, mcm_method, capture_mode):
        """Test that the correct default mcm_method is chosen based on postselect_mode"""
        dev = qml.device(backend, wires=1)

        @qjit(capture=capture_mode)
        @qml.set_shots(20)
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
    def test_mcm_method_with_count_mesurement(
        self, backend, postselect_mode, mcm_method, capture_mode
    ):
        """Test that the correct default mcm_method is chosen based on postselect_mode"""
        dev = qml.device(backend, wires=1)

        @qjit(capture=capture_mode)
        @qml.set_shots(20)
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

    # Capture gap: MCM capture execution still unsupported (e.g., "Measurements of mcms are not yet supported").
    # Classification: missing PL feature/integration gap. Fix: add capture-time lowering/execution for MCM-derived measurements.
    @pytest.mark.capture_todo
    @pytest.mark.parametrize("postselect_mode", [None, "fill-shots", "hw-like"])
    @pytest.mark.parametrize("mcm_method", [None, "one-shot"])
    def test_mcm_method_with_dict_output_used_measurements(
        self, backend, postselect_mode, mcm_method, capture_mode
    ):
        """Test that the correct default mcm_method is chosen based on postselect_mode"""
        dev = qml.device(backend, wires=1)

        @qjit(capture=capture_mode)
        @qml.set_shots(5)
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

    # Capture gap: MCM capture execution still unsupported (e.g., "Measurements of mcms are not yet supported").
    # Classification: missing PL feature/integration gap. Fix: add capture-time lowering/execution for MCM-derived measurements.
    @pytest.mark.capture_todo
    @pytest.mark.parametrize("mcm_method", [None, "single-branch-statistics", "one-shot"])
    def test_invalid_postselect_error(self, backend, mcm_method, capture_mode):
        """Test that an error is raised if postselecting on an invalid value"""
        dev = qml.device(backend, wires=1)

        @qjit(capture=capture_mode)
        @qml.set_shots(10)
        @qml.qnode(dev, mcm_method=mcm_method)
        def circuit(x):
            qml.RX(x, 0)
            measure(0, postselect=-1)
            return qml.expval(qml.Z(0))

        with pytest.raises(TypeError, match="postselect must be '0' or '1'"):
            _ = circuit(1.8)

    # Capture gap: MCM capture execution still unsupported (e.g., "Measurements of mcms are not yet supported").
    # Classification: missing PL feature/integration gap. Fix: add capture-time lowering/execution for MCM-derived measurements.
    @pytest.mark.capture_todo
    @pytest.mark.parametrize("measurement_process", [qml.counts, qml.var, qml.expval, qml.probs])
    def test_single_branch_statistics_not_implemented_error(
        self, backend, measurement_process, capture_mode
    ):
        """
        Test that NotImplementedError is raised when using mid-circuit
        measurements inside measurement processes with single-branch-statistics.
        """

        err = "single-branch-statistics does not support measurement processes"
        with pytest.raises(NotImplementedError, match=err):

            @qjit(capture=capture_mode)
            @qml.set_shots(5)
            @qml.qnode(qml.device(backend, wires=2), mcm_method="single-branch-statistics")
            def measurement():
                qml.Hadamard(0)
                m = measure(0)
                return measurement_process(op=m)

            measurement()


class TestDynamicOneShotIntegration:
    """Integration tests for QNodes using mcm_method="one-shot"/dynamic_one_shot."""

    @pytest.mark.parametrize("shots", [1, 2])
    def test_dynamic_one_shot_static_argnums(self, backend, shots, capture_mode):
        """
        Test static argnums is passed correctly to the one shot qnodes.
        """

        @qjit(static_argnums=0, capture=capture_mode)
        def workflow(N):
            dev = qml.device(backend, wires=N)

            @qml.set_shots(N)
            @qml.qnode(dev, mcm_method="one-shot")
            def circ():
                return qml.probs()

            return circ()

        assert np.allclose(workflow(shots), [1 if i == 0 else 0 for i in range(2**shots)])

    # pylint: disable=too-many-arguments
    # Capture gap: dynamic-one-shot path raises NotImplementedError for captured MCM-valued measurements.
    # Classification: missing PL feature. Fix: implement capture execution support for dynamic_one_shot measurement processes.
    @pytest.mark.capture_todo
    @pytest.mark.parametrize(
        "postselect, reset, expected",
        [
            (None, False, (1, 1)),
            (0, False, (0, 0)),
            (1, False, (1, 1)),
            pytest.param(
                None,
                True,
                (1, 0),
                marks=pytest.mark.xfail(reason="waiting for PennyLane squeeze issue fix"),
            ),
            (0, True, (0, 0)),
            pytest.param(
                1,
                True,
                (1, 0),
                marks=pytest.mark.xfail(reason="waiting for PennyLane squeeze issue fix"),
            ),
        ],
    )
    @pytest.mark.parametrize("postselect_mode", ["hw-like", "fill-shots"])
    def test_mcm_method_one_shot_with_single_shot(
        self, backend, postselect, reset, expected, postselect_mode, capture_mode
    ):
        """Test that the result is correct when using mcm_method="one-shot" with a single shot"""
        if postselect == 0 and postselect_mode == "fill-shots":
            pytest.xfail(
                reason="fill-shots not currently working when postselecting a zero probability state"
            )

        dev = qml.device(backend, wires=1)

        @qjit(capture=capture_mode)
        @qml.set_shots(1)
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

    # Capture gap: dynamic-one-shot path raises NotImplementedError for captured MCM-valued measurements.
    # Classification: missing PL feature. Fix: implement capture execution support for dynamic_one_shot measurement processes.
    @pytest.mark.capture_todo
    @pytest.mark.parametrize("shots", [1, 10])
    def test_dynamic_one_shot_only_called_once(self, backend, shots, mocker, capture_mode):
        """Test that when using mcm_method="one-shot", dynamic_one_shot does not get
        called multiple times"""
        dev = qml.device(backend, wires=1)
        spy = mocker.spy(catalyst.qfunc, "dynamic_one_shot")

        @qjit(capture=capture_mode)
        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method="one-shot")
        def circuit(x):
            qml.RY(x, wires=0)
            measure(0)
            return qml.sample(wires=0)

        param = jnp.array(0.1)
        _ = circuit(param)

        assert spy.call_count == 1

    # Capture gap: dynamic-one-shot path raises NotImplementedError for captured MCM-valued measurements.
    # Classification: missing PL feature. Fix: implement capture execution support for dynamic_one_shot measurement processes.
    @pytest.mark.capture_todo
    def test_dynamic_one_shot_unsupported_measurement(self, backend, capture_mode):
        """Test that circuits with unsupported measurements raise an error."""
        shots = 10
        dev = qml.device(backend, wires=1)
        param = np.pi / 4

        @qjit(capture=capture_mode)
        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method="one-shot")
        def func(x):
            qml.RX(x, wires=0)
            _ = measure(0)
            return qml.classical_shadow(wires=0)

        with pytest.raises(
            NotImplementedError,
            match="measurement process is not compatible with the chosen or default mcm_method",
        ):
            func(param)

    def test_dynamic_one_shot_unsupported_none_shots(self, backend, capture_mode):
        """Test that `dynamic_one_shot` raises when used with non-finite shots."""
        dev = qml.device(backend, wires=1)

        with pytest.raises(
            exceptions.QuantumFunctionError,
            match="dynamic_one_shot is only supported with finite shots.",
        ):

            @qjit(capture=capture_mode)
            @catalyst.qfunc.dynamic_one_shot
            @qml.set_shots(None)
            @qml.qnode(dev)
            def _(x, y):
                qml.RX(x, wires=0)
                _ = measure(0)
                qml.RX(y, wires=0)
                return qml.probs(wires=0)

    # Capture gap: dynamic-one-shot path raises NotImplementedError for captured MCM-valued measurements.
    # Classification: missing PL feature. Fix: implement capture execution support for dynamic_one_shot measurement processes.
    @pytest.mark.capture_todo
    def test_dynamic_one_shot_unsupported_broadcast(self, backend, capture_mode):
        """Test that `dynamic_one_shot` raises when used with parameter broadcasting."""
        shots = 10
        dev = qml.device(backend, wires=1)
        param = np.pi / 4 * jnp.ones(2)

        @qjit(capture=capture_mode)
        @qml.set_shots(shots)
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

    # Capture gap: dynamic-one-shot path raises NotImplementedError for captured MCM-valued measurements.
    # Classification: missing PL feature. Fix: implement capture execution support for dynamic_one_shot measurement processes.
    @pytest.mark.capture_todo
    @pytest.mark.parametrize("param, expected", [(0.0, 0.0), (jnp.pi, 1.0)])
    def test_dynamic_one_shot_with_sample_single(self, backend, param, expected, capture_mode):
        """Test that a measurement result can be returned with qml.sample and shots."""
        shots = 10
        dev = qml.device(backend, wires=1)

        @qjit(capture=capture_mode)
        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method="one-shot")
        def circuit(x):
            qml.RY(x, wires=0)
            m = measure(0)
            qml.PauliX(0)
            return qml.sample(m)

        result = circuit(param)
        assert result.shape == (shots,)
        assert jnp.allclose(result, expected)

    # Capture gap: dynamic-one-shot path raises NotImplementedError for captured MCM-valued measurements.
    # Classification: missing PL feature. Fix: implement capture execution support for dynamic_one_shot measurement processes.
    @pytest.mark.capture_todo
    @pytest.mark.parametrize("shots", [10000])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    @pytest.mark.parametrize("measure_f", [qml.counts, qml.expval, qml.probs, qml.sample, qml.var])
    @pytest.mark.parametrize(
        "meas_obj", [qml.PauliZ(0), qml.Hadamard(0) @ qml.PauliZ(1), [0], [0, 1], "mcm"]
    )
    @pytest.mark.parametrize("postselect_mode", ["fill-shots", "hw-like"])
    # pylint: disable=too-many-arguments
    def test_dynamic_one_shot_several_mcms(
        self, backend, shots, postselect, measure_f, meas_obj, postselect_mode, capture_mode
    ):
        """Tests that Catalyst yields the same results as PennyLane's DefaultQubit for a simple
        circuit with a mid-circuit measurement."""
        if measure_f in (qml.counts, qml.probs, qml.sample) and (
            not isinstance(meas_obj, list) and not meas_obj == "mcm"
        ):
            pytest.skip("Can't use observables with counts, probs or sample")

        if measure_f in (qml.var, qml.expval) and (isinstance(meas_obj, list)):
            pytest.skip("Can't use wires/mcm lists with var or expval")

        dq = qml.device("default.qubit", seed=8237945)

        @partial(qml.set_shots, shots=shots)
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

        dev = qml.device(backend, wires=2)

        @qjit(seed=123456, capture=capture_mode)
        @partial(qml.set_shots, shots=shots)
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
            with pytest.raises(
                NotImplementedError, match=r"qml.var\(\) cannot be used on observables"
            ):
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
    # Capture gap: dynamic-one-shot path raises NotImplementedError for captured MCM-valued measurements.
    # Classification: missing PL feature. Fix: implement capture execution support for dynamic_one_shot measurement processes.
    @pytest.mark.capture_todo
    @pytest.mark.parametrize("shots", [10000])
    @pytest.mark.parametrize("postselect", [None, 0, 1])
    @pytest.mark.parametrize("reset", [False, True])
    @pytest.mark.parametrize("postselect_mode", ["fill-shots", "hw-like"])
    def test_dynamic_one_shot_multiple_measurements(
        self, backend, shots, postselect, reset, postselect_mode, capture_mode
    ):
        """Tests that Catalyst yields the same results as PennyLane's DefaultQubit for a simple
        circuit with a mid-circuit measurement and several terminal measurements."""
        if backend in ("lightning.kokkos", "lightning.gpu"):
            obs = qml.PauliZ(0)
        else:
            obs = qml.PauliY(0)

        dq = qml.device("default.qubit", seed=8237945)

        @qml.set_shots(shots)
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

        dev = qml.device(backend, wires=2)

        @qjit(seed=37, capture=capture_mode)
        @qml.set_shots(shots)
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

    def test_dynamic_one_shot_with_no_mcm_iterable_output(self, backend, capture_mode):
        """Test that `dynamic_one_shot` can work when there is no mcm and have iterable output."""
        qubits = 3
        shots = 10
        dev = qml.device(backend, wires=qubits)

        @qjit(capture=capture_mode)
        @qml.set_shots(shots)
        @qml.qnode(dev, mcm_method="one-shot")
        def cost():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            return [qml.expval(qml.Z(i)) for i in range(qubits)]

        result = cost()
        assert jnp.array(result).shape == (qubits,)

    # Capture gap: ValueError "Only Measurement Processes can be returned from QNode's" for MCM/classical returns.
    # Classification: missing PL capture feature. Fix: support classical/MCM return pathways or rewrite to measurement-process outputs.
    @pytest.mark.capture_todo
    def test_dynamic_one_shot_mcm_result(self, capture_mode):
        """Test mcm result with one-shot"""
        dev = qml.device("lightning.qubit", wires=1)

        @qjit(capture=capture_mode)
        @qml.set_shots(10)
        @qml.qnode(dev, mcm_method="one-shot")
        def circuit():
            qml.Hadamard(0)
            return measure(0)

        result = circuit()
        assert result.shape == (10,)

    # Capture gap: ValueError "Only Measurement Processes can be returned from QNode's" for MCM/classical returns.
    # Classification: missing PL capture feature. Fix: support classical/MCM return pathways or rewrite to measurement-process outputs.
    @pytest.mark.capture_todo
    def test_dynamic_one_shot_classical_return_values_with_mcm(self, capture_mode):
        """Test classical return value with one-shot"""

        @qjit(autograph=True, capture=capture_mode)
        @qml.set_shots(10)
        @qml.qnode(qml.device("lightning.qubit", wires=1), mcm_method="one-shot")
        def circuit():
            qml.Hadamard(wires=0)
            if measure(0):
                return 42
            else:
                return 43

        result = circuit()
        assert result.shape == (10,)  # pylint: disable=no-member

    # Capture gap: ValueError "Only Measurement Processes can be returned from QNode's" for MCM/classical returns.
    # Classification: missing PL capture feature. Fix: support classical/MCM return pathways or rewrite to measurement-process outputs.
    @pytest.mark.capture_todo
    def test_dynamic_one_shot_with_classical_return_values(self, capture_mode):
        """Test classical return values with one-shot"""
        dev = qml.device("lightning.qubit", wires=1)

        @qjit(capture=capture_mode)
        @qml.set_shots(12)
        @qml.qnode(dev, mcm_method="one-shot")
        def circuit():
            qml.Hadamard(0)
            return {
                "first": qml.sample(),
                "second": [100, qml.sample()],
                "third": (qml.sample(), qml.sample()),
            }

        result = circuit()

        assert list(result.keys()) == ["first", "second", "third"]
        assert jnp.array(result["first"]).shape == (12, 1)
        assert jnp.allclose(result["second"][0], jnp.full(12, 100))
        assert jnp.array(result["second"][1]).shape == (12, 1)
        assert jnp.array(result["third"]).shape == (2, 12, 1)

    @pytest.mark.skip(
        reason="grad with dynamic one-shot is not yet supported.",
    )
    def test_mcm_method_with_grad(self, backend, capture_mode):
        """Test that the dynamic_one_shot works with grad."""

        dev = qml.device(backend, wires=1)

        @qml.set_shots(5)
        @qml.qnode(dev, diff_method="best", mcm_method="one-shot")
        def f(x: float):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        @qml.set_shots(5)
        @qml.qnode(dev, diff_method="best")
        def g(x: float):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        @qjit(capture=capture_mode)
        def grad_f(x):
            return grad(f, method="auto")(x)

        @qjit(capture=capture_mode)
        def grad_g(x):
            return grad(g, method="auto")(x)

        assert np.allclose(grad_f(1.0), grad_g(1.0))

    # value_and_grad now will be supported, but jax has an issue with the current version we use
    # as it results in a random memory error. https://github.com/tensorflow/tensorflow/pull/97681
    # It will be fixed in the next jax release. we will re-enable this test when the jax
    # release is available and tested on our end.
    @pytest.mark.skip(
        reason="https://github.com/tensorflow/tensorflow/pull/97681",
    )
    def test_mcm_method_with_value_and_grad(self, capture_mode):
        """Test that the dynamic_one_shot works with value_and_grad."""

        @qjit(capture=capture_mode)
        def workflow1(x: float):
            @qml.set_shots(10)
            @qml.qnode(qml.device("lightning.qubit", wires=3), mcm_method="one-shot")
            def circuit1():
                qml.CNOT(wires=[0, 1])
                qml.RX(0, wires=[2])
                return qml.probs()  # This is [1, 0, 0, ...]

            return x * (circuit1()[0])

        @qjit(capture=capture_mode)
        def workflow2(x: float):
            @qml.set_shots(10)
            @qml.qnode(qml.device("lightning.qubit", wires=3))
            def circuit2():
                qml.CNOT(wires=[0, 1])
                qml.RX(0, wires=[2])
                return qml.probs()  # This is [1, 0, 0, ...]

            return x * (circuit2()[0])

        result1 = qjit(value_and_grad(workflow1), capture=capture_mode)(3.0)
        result2 = qjit(value_and_grad(workflow2), capture=capture_mode)(3.0)
        assert np.allclose(result1, result2)

    @pytest.mark.parametrize("diff_method", ["auto", "fd"])
    @pytest.mark.xfail(
        reason="jvp with dynamic one-shot is not yet supported.",
        run=False,
    )
    def test_mcm_method_with_jvp(self, backend, diff_method, capture_mode):
        """Test that the dynamic_one_shot works with jvp."""
        dev = qml.device(backend, wires=1)
        x, t = (
            [-0.1, 0.5],
            [0.1, 0.33],
        )

        def circuit_rx(x1, x2):
            """A test quantum function"""
            qml.RX(x1, wires=0)
            qml.RX(x2, wires=0)
            return qml.expval(qml.PauliY(0))

        @qjit(capture=capture_mode)
        def C_workflow():
            f = qml.set_shots(qml.QNode(circuit_rx, device=dev, mcm_method="one-shot"), shots=5)
            return C_jvp(f, x, t, method=diff_method, argnums=list(range(len(x))))

        @qjit(capture=capture_mode)
        def J_workflow():
            f = qml.set_shots(qml.QNode(circuit_rx, device=dev), shots=5)
            return C_jvp(f, x, t, method=diff_method, argnums=list(range(len(x))))

        r1 = C_workflow()
        r2 = J_workflow()
        res_jax, tree_jax = tree_flatten(r1)
        res_cat, tree_cat = tree_flatten(r2)
        assert tree_jax == tree_cat
        assert np.allclose(res_jax, res_cat)

    @pytest.mark.parametrize("diff_method", ["auto", "fd"])
    @pytest.mark.xfail(
        reason="vjp with dynamic one-shot is not yet supported.",
        run=False,
    )
    def test_mcm_method_with_vjp(self, backend, diff_method, capture_mode):
        """Test that the dynamic_one_shot works with vjp."""
        dev = qml.device(backend, wires=1)

        def circuit_rx(x1, x2):
            """A test quantum function"""
            qml.RX(x1, wires=0)
            qml.RX(x2, wires=0)
            return qml.expval(qml.PauliY(0))

        x, ct = (
            [-0.1, 0.5],
            [0.111],
        )

        @qjit(capture=capture_mode)
        def C_workflow():
            f = qml.set_shots(qml.QNode(circuit_rx, device=dev, mcm_method="one-shot"), shots=5)
            return C_vjp(f, x, ct, method=diff_method, argnums=list(range(len(x))))

        @qjit(capture=capture_mode)
        def J_workflow():
            f = qml.set_shots(qml.QNode(circuit_rx, device=dev), shots=5)
            return C_vjp(f, x, ct, method=diff_method, argnums=list(range(len(x))))

        r1 = C_workflow()
        r2 = J_workflow()
        res_jax, tree_jax = tree_flatten(r1)
        res_cat, tree_cat = tree_flatten(r2)
        assert tree_jax == tree_cat
        assert np.allclose(res_jax, res_cat)


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
