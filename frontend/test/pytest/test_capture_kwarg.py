# Copyright 2024-2026 Xanadu Quantum Technologies Inc.

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

import catalyst
from catalyst import qjit
from catalyst.tracing.contexts import CaptureContext


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

    def test_capture_context_fallback_to_global(self):
        """Test that CaptureContext falls back to qml.capture.enabled() when no local context."""
        # When no capture context is active, should follow global setting
        qml.capture.disable()
        assert not CaptureContext.is_capture_enabled()

        qml.capture.enable()
        try:
            assert CaptureContext.is_capture_enabled()
        finally:
            qml.capture.disable()

    def test_capture_context_local_true_overrides_global(self):
        """Test that local capture=True overrides global disable."""
        qml.capture.disable()
        assert not qml.capture.enabled()

        with CaptureContext(True):
            assert CaptureContext.is_capture_enabled()

        # After context exits, should revert
        assert not CaptureContext.is_capture_enabled()

    def test_capture_context_local_false_overrides_global(self):
        """Test that local capture=False overrides global enable and pauses global capture."""
        qml.capture.enable()
        try:
            assert qml.capture.enabled()

            with CaptureContext(False):
                # Both local check and global should be False
                assert not CaptureContext.is_capture_enabled()
                # Crucially, qml.capture.enabled() should also be False (paused)
                assert not qml.capture.enabled(), "Global capture should be paused"

            # After context exits, should revert
            assert CaptureContext.is_capture_enabled()
            assert qml.capture.enabled(), "Global capture should be restored"
        finally:
            qml.capture.disable()

    def test_capture_context_pause_isolation(self):
        """Test that capture=False properly pauses PennyLane's global capture."""
        qml.capture.enable()
        try:
            # Before: global is enabled
            assert qml.capture.enabled()

            with CaptureContext(False):
                # Inside: global should be paused
                assert not qml.capture.enabled()

                # This ensures PennyLane won't produce AbstractMeasurement objects
                # which would break the old tracing pathway

            # After: global should be restored
            assert qml.capture.enabled()
        finally:
            qml.capture.disable()

    def test_capture_context_global_follows_global(self):
        """Test that capture='global' follows the global setting."""
        with CaptureContext("global"):
            qml.capture.disable()
            assert not CaptureContext.is_capture_enabled()

            qml.capture.enable()
            try:
                assert CaptureContext.is_capture_enabled()
            finally:
                qml.capture.disable()

    def test_capture_context_nesting(self):
        """Test that capture contexts can be nested."""
        qml.capture.disable()

        with CaptureContext(True):
            assert CaptureContext.is_capture_enabled()

            with CaptureContext(False):
                assert not CaptureContext.is_capture_enabled()

            # After inner context exits, outer context's mode should apply
            assert CaptureContext.is_capture_enabled()

        assert not CaptureContext.is_capture_enabled()


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


class TestCaptureKwargWithControlFlow:
    """Test capture kwarg interaction with control flow."""

    def test_catalyst_for_loop_with_capture_false(self, backend):
        """Test that catalyst.for_loop works with capture=False."""
        qml.capture.disable()

        @qjit(capture=False)
        def func():
            @catalyst.for_loop(0, 5, 1)
            def loop_fn(i, acc):
                return acc + i

            return loop_fn(0)

        result = func()
        assert result == 10  # 0+1+2+3+4 = 10

    def test_catalyst_for_loop_with_capture_true_raises(self):
        """Test that catalyst.for_loop raises error with capture=True."""
        # When capture=True, catalyst.for_loop should raise an error
        # telling users to use qml.for_loop instead
        from catalyst.utils.exceptions import PlxprCaptureCFCompatibilityError

        with pytest.raises(PlxprCaptureCFCompatibilityError):

            @qjit(capture=True)
            def func():
                @catalyst.for_loop(0, 5, 1)
                def loop_fn(i, acc):
                    return acc + i

                return loop_fn(0)

            func()
