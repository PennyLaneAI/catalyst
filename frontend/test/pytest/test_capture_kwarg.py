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
from catalyst.compiler import CompileOptions
from catalyst.jit import _ensure_capture_mode as ensure_capture_mode


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
            assert qml.capture.enabled()
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
                assert not qml.capture.enabled()
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
            assert not qml.capture.enabled()
            qml.RX(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        result = circuit_old_pathway(0.5)
        assert jnp.allclose(result, jnp.cos(0.5))

    def test_capture_true_classical_only(self):
        """Test that capture=True works for classical-only functions."""
        qml.capture.disable()

        @qjit(capture=True)
        def classical_fn(x, y):
            assert qml.capture.enabled()
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
            assert qml.capture.enabled()
            qml.RY(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qjit(capture=False)
        @qml.qnode(dev)
        def circuit_old(theta):
            assert not qml.capture.enabled()
            qml.RY(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

        result_capture = circuit_capture(0.5)
        result_old = circuit_old(0.5)

        # Both should give the same result
        assert jnp.allclose(result_capture, result_old)
        assert jnp.allclose(result_capture, jnp.cos(0.5))


class TestCaptureCompileOptionsIdentity:
    """Test that capture flag is part of CompileOptions identity.

    These tests ensure cache collisions cannot occur between capture=True
    and capture=False compilations. Since `capture` fundamentally changes
    the compilation pipeline (different lowering paths, different MLIR dialects),
    it must be distinguishable in CompileOptions.
    """

    def test_capture_flag_creates_distinct_compile_options(self):
        """Test that qjit(capture=True) and qjit(capture=False) have distinct CompileOptions."""
        qml.capture.disable()  # Ensure global state doesn't affect the test

        @qjit(capture=True)
        def fn_capture(x):
            return x * 2

        @qjit(capture=False)
        def fn_old(x):
            return x * 2

        @qjit(capture="global")
        def fn_global(x):
            return x * 2

        # The compile options must be different
        assert fn_capture.compile_options.capture != fn_old.compile_options.capture
        assert fn_capture.compile_options.capture is True
        assert fn_old.compile_options.capture is False
        assert fn_global.compile_options.capture == "global"

        # CompileOptions should not be equal when capture differs
        assert fn_capture.compile_options != fn_old.compile_options

    def test_aot_preserves_capture_flag(self):
        """Test that the capture flag persists in CompileOptions for AOT compilation.

        This ensures the flag isn't lost when using qjit without immediately calling it.
        """
        qml.capture.disable()  # Global state is disabled

        @qjit(capture=True)
        def circuit(x: float):  # Type hint enables AOT compilation
            return x * 2

        # Inspect the options BEFORE running
        options = circuit.compile_options

        # Check that it's stored and set to True (not the global state)
        assert options.capture is True

        # Should differ from the global state
        assert options.capture != "global"
        # If capture was accidentally defaulting to global, this would fail
        assert options.capture != qml.capture.enabled()

    def test_capture_false_aot_preserves_flag(self):
        """Test that capture=False persists in AOT mode even when global is enabled."""
        qml.capture.enable()
        try:

            @qjit(capture=False)
            def circuit(x: float):  # Type hint enables AOT compilation
                return x * 2

            options = circuit.compile_options
            assert options.capture is False
            # Should not have defaulted to global state
            assert options.capture != qml.capture.enabled()
        finally:
            qml.capture.disable()

    def test_compile_options_capture_field_exists(self):
        """Test that CompileOptions dataclass has the capture field."""
        # Create a CompileOptions with explicit capture value
        opts_true = CompileOptions(capture=True)
        opts_false = CompileOptions(capture=False)
        opts_global = CompileOptions(capture="global")
        opts_default = CompileOptions()

        assert opts_true.capture is True
        assert opts_false.capture is False
        assert opts_global.capture == "global"
        assert opts_default.capture == "global"  # Default should be "global"

    def test_different_capture_produces_different_qjit_objects(self, backend):
        """Test that different capture settings produce independent QJIT objects.

        This is the "collision" test - ensures that calling with different
        capture modes doesn't accidentally reuse cached results.
        """
        qml.capture.disable()
        dev = qml.device(backend, wires=1)

        # Use a counter to track compilations
        compilation_count = {"capture": 0, "old": 0}

        def make_circuit_capture():
            compilation_count["capture"] += 1

            @qjit(capture=True)
            @qml.qnode(dev)
            def circuit(x):
                qml.RX(x, wires=0)
                return qml.expval(qml.PauliZ(0))

            return circuit

        def make_circuit_old():
            compilation_count["old"] += 1

            @qjit(capture=False)
            @qml.qnode(dev)
            def circuit(x):
                qml.RX(x, wires=0)
                return qml.expval(qml.PauliZ(0))

            return circuit

        # Create circuits with different capture modes
        circuit_capture = make_circuit_capture()
        circuit_old = make_circuit_old()

        # Both should work and produce same mathematical results
        result_capture = circuit_capture(0.5)
        result_old = circuit_old(0.5)

        assert jnp.allclose(result_capture, result_old)
        assert jnp.allclose(result_capture, jnp.cos(0.5))

        # They should be completely separate QJIT objects
        assert circuit_capture is not circuit_old
        assert circuit_capture.compile_options != circuit_old.compile_options
