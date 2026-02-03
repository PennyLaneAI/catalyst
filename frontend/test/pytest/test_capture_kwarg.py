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
"""Tests for the capture kwarg in @qjit decorator."""

import jax.numpy as jnp
import pennylane as qml
import pytest

from catalyst import qjit
from catalyst.tracing.contexts import ensure_capture_mode


@pytest.mark.usefixtures("use_both_frontend")
class TestCaptureKwarg:
    """Test suite for the capture kwarg functionality."""

    def test_capture_kwarg_default_is_global(self):
        """Test that the default value of capture is 'global'."""

        @qjit
        def f(x):
            return x * 2

        assert f.compile_options.capture == "global"

    def test_capture_kwarg_true(self):
        """Test that capture=True is accepted."""

        @qjit(capture=True)
        def f(x):
            return x * 2

        assert f.compile_options.capture is True

    def test_capture_kwarg_false(self):
        """Test that capture=False is accepted."""

        @qjit(capture=False)
        def f(x):
            return x * 2

        assert f.compile_options.capture is False

    def test_capture_kwarg_invalid_raises(self):
        """Test that invalid capture values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid value for capture"):

            @qjit(capture="invalid")
            def f(x):
                return x * 2


class TestEnsureCaptureMode:
    """Test suite for the ensure_capture_mode context manager (Scope Enforcement pattern)."""

    def test_enable_capture_when_disabled(self):
        """Test that ensure_capture_mode(True) enables capture when globally disabled."""
        qml.capture.disable()
        assert not qml.capture.enabled()

        with ensure_capture_mode(True):
            assert qml.capture.enabled()

        # After context exits, should be restored to disabled
        assert not qml.capture.enabled()

    def test_disable_capture_when_enabled(self):
        """Test that ensure_capture_mode(False) disables capture when globally enabled."""
        qml.capture.enable()
        try:
            assert qml.capture.enabled()

            with ensure_capture_mode(False):
                assert not qml.capture.enabled()

            # After context exits, should be restored to enabled
            assert qml.capture.enabled()
        finally:
            qml.capture.disable()

    def test_no_op_when_already_in_target_state_enabled(self):
        """Test that ensure_capture_mode(True) is a no-op when already enabled."""
        qml.capture.enable()
        try:
            assert qml.capture.enabled()

            with ensure_capture_mode(True):
                assert qml.capture.enabled()

            assert qml.capture.enabled()
        finally:
            qml.capture.disable()

    def test_no_op_when_already_in_target_state_disabled(self):
        """Test that ensure_capture_mode(False) is a no-op when already disabled."""
        qml.capture.disable()
        assert not qml.capture.enabled()

        with ensure_capture_mode(False):
            assert not qml.capture.enabled()

        assert not qml.capture.enabled()

    def test_nesting_different_states(self):
        """Test that nested contexts properly restore their respective states (snapshot pattern)."""
        qml.capture.disable()
        assert not qml.capture.enabled()

        with ensure_capture_mode(True):
            assert qml.capture.enabled()

            with ensure_capture_mode(False):
                assert not qml.capture.enabled()

            # After inner context exits, should be restored to True
            assert qml.capture.enabled()

        # After outer context exits, should be restored to disabled
        assert not qml.capture.enabled()

    def test_exception_safety(self):
        """Test that state is restored even if an exception is raised."""
        qml.capture.disable()
        assert not qml.capture.enabled()

        try:
            with ensure_capture_mode(True):
                assert qml.capture.enabled()
                raise ValueError("Test exception")
        except ValueError:
            pass

        # State should still be restored
        assert not qml.capture.enabled()


class TestCaptureKwargIntegration:
    """Integration tests for capture kwarg with actual quantum circuits."""

    def test_capture_true_uses_capture_pathway(self, backend):
        """Test that capture=True uses the capture pathway without global enable."""
        qml.capture.disable()  # Make sure global capture is off
        assert not qml.capture.enabled()

        dev = qml.device(backend, wires=2)

        @qjit(capture=True)
        @qml.qnode(dev)
        def circuit(theta):
            qml.RX(theta, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        result = circuit(0.5)
        assert jnp.allclose(result, jnp.cos(0.5))

        # Verify global capture is still off
        assert not qml.capture.enabled()

    def test_capture_false_uses_old_pathway(self, backend):
        """Test that capture=False uses old pathway even if global is on."""
        qml.capture.enable()
        try:
            dev = qml.device(backend, wires=2)

            @qjit(capture=False)
            @qml.qnode(dev)
            def circuit(theta):
                qml.RX(theta, wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.expval(qml.PauliZ(1))

            result = circuit(0.5)
            assert jnp.allclose(result, jnp.cos(0.5))
        finally:
            qml.capture.disable()

    def test_capture_global_follows_global_setting(self, backend):
        """Test that capture='global' respects the global setting."""
        dev = qml.device(backend, wires=2)

        # With global capture off
        qml.capture.disable()

        @qjit(capture="global")
        @qml.qnode(dev)
        def circuit_old_pathway(theta):
            qml.RX(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        result = circuit_old_pathway(0.5)
        assert jnp.allclose(result, jnp.cos(0.5))

    def test_capture_true_classical_only(self):
        """Test that capture=True works for classical-only functions."""
        qml.capture.disable()

        @qjit(capture=True)
        def classical_fn(x, y):
            return x**2 + y**2

        result = classical_fn(3.0, 4.0)
        assert jnp.allclose(result, 25.0)

    def test_multiple_qjits_different_capture_modes(self, backend):
        """Test multiple QJIT functions with different capture modes."""
        dev = qml.device(backend, wires=1)
        qml.capture.disable()

        @qjit(capture=True)
        @qml.qnode(dev)
        def circuit_capture(theta):
            qml.RY(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(capture=False)
        @qml.qnode(dev)
        def circuit_old(theta):
            qml.RY(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        result_capture = circuit_capture(0.5)
        result_old = circuit_old(0.5)

        # Both should give the same result
        assert jnp.allclose(result_capture, result_old)
        assert jnp.allclose(result_capture, jnp.cos(0.5))
