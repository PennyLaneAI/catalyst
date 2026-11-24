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
"""Tests for the jax_extras.patches module"""

from functools import partial

import jax
import pytest
from jax._src import core

from catalyst.jax_extras import transient_jax_config
from catalyst.jax_extras.patches import get_jax_patches, mock_attributes
from catalyst.utils.patching import Patcher


@pytest.mark.parametrize("check_leaks", [False, True])
def test_dynamic_trace_invalidation(check_leaks):
    """Test that capture invalidates dynamic jaxpr traces and preserves the resulting Jaxpr."""
    traces = []
    original_trace = core.trace_ctx.trace
    original_exit = core.set_current_trace.__exit__

    def fn(value):
        traces.append(core.trace_ctx.trace)
        return jax.lax.add(value, 1.0)

    with (
        transient_jax_config({"jax_dynamic_shapes": True}),
        jax.check_tracer_leaks(check_leaks),
        Patcher(*get_jax_patches()),
    ):
        jaxpr = jax.make_jaxpr(fn)(2.0)

    assert traces
    assert all(not trace.is_valid() for trace in traces)
    assert core.trace_ctx.trace is original_trace
    assert original_trace.is_valid()
    assert core.set_current_trace.__exit__ is original_exit
    assert jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 2.0) == [3.0]


@pytest.mark.parametrize("check_leaks", [False, True])
def test_dynamic_trace_invalidation_on_error(check_leaks):
    """Test that a capture error invalidates dynamic jaxpr traces and restores the previous context."""
    traces = []
    original_trace = core.trace_ctx.trace
    original_exit = core.set_current_trace.__exit__

    def fn():
        traces.append(core.trace_ctx.trace)
        raise ValueError("capture failed")

    with (
        transient_jax_config({"jax_dynamic_shapes": True}),
        jax.check_tracer_leaks(check_leaks),
        Patcher(*get_jax_patches()),
        pytest.raises(ValueError, match="capture failed"),
    ):
        jax.make_jaxpr(fn)()

    assert traces
    assert all(not trace.is_valid() for trace in traces)
    assert core.trace_ctx.trace is original_trace
    assert original_trace.is_valid()
    assert core.set_current_trace.__exit__ is original_exit


def test_inline_jit_preserves_enclosing_trace():
    """Test that an inlined JIT call leaves the enclosing trace valid for subsequent operations."""

    @partial(jax.jit, inline=True)
    def inner(value):
        return jax.lax.mul(value, 2.0)

    def fn(value):
        trace = core.trace_ctx.trace
        result = inner(value)
        assert core.trace_ctx.trace is trace
        assert trace.is_valid()
        return jax.lax.add(result, 1.0)

    with transient_jax_config({"jax_dynamic_shapes": True}), Patcher(*get_jax_patches()):
        jaxpr = jax.make_jaxpr(fn)(3.0)

    assert jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 3.0) == [7.0]


def test_exit_preserves_non_dynamic_jaxpr_trace():
    """Test that the exit patch restores the previous context without invalidating other traces."""
    trace = core.Trace()
    original_trace = core.trace_ctx.trace

    with Patcher(*get_jax_patches()):
        with core.set_current_trace(trace):
            assert core.trace_ctx.trace is trace

        assert trace.is_valid()
        assert core.trace_ctx.trace is original_trace


# pylint: disable=missing-class-docstring,missing-function-docstring
class TestMockAttributes:
    """Test the mock_attributes function and MockAttributeWrapper class."""

    def test_mock_attributes_returns_mocked_value(self):
        """Test that accessing a mocked attribute returns the mocked value."""

        class DummyClass:
            def __init__(self):
                self.original_attr = "original"

        obj = DummyClass()
        mocked = mock_attributes(obj, {"mocked_attr": "mocked_value"})

        # Access the mocked attribute - this should come from the attrs dict
        assert mocked.mocked_attr == "mocked_value"

    def test_mock_attributes_returns_original_value(self):
        """Test that accessing an unmocked attribute returns the original value."""

        class DummyClass:
            def __init__(self):
                self.original_attr = "original"

        obj = DummyClass()
        mocked = mock_attributes(obj, {"mocked_attr": "mocked_value"})

        # Access the original attribute - this should come from the original object
        # This tests the else branch in __getattr__
        assert mocked.original_attr == "original"

    def test_mock_attributes_with_methods(self):
        """Test that calling original methods works through the wrapper."""

        class DummyClass:
            def __init__(self):
                self.value = 42

            def get_value(self):
                return self.value

        obj = DummyClass()

        def mocked_method():
            return "mocked"

        mocked = mock_attributes(obj, {"mocked_method": mocked_method})

        # Access the mocked method
        assert mocked.mocked_method() == "mocked"

        # Access the original method - tests the getattr fallback
        assert mocked.get_value() == 42

    def test_mock_attributes_with_callable(self):
        """Test mocking with callable attributes like lambda functions."""

        class DummyClass:
            def __init__(self):
                self.original_func = lambda x: x * 2

        obj = DummyClass()
        mocked = mock_attributes(obj, {"new_func": lambda x: x * 3})

        # Access the mocked callable
        assert mocked.new_func(5) == 15

        # Access the original callable - tests the getattr fallback
        assert mocked.original_func(5) == 10

    def test_mock_attributes_override_existing(self):
        """Test that mocking can override existing attributes."""

        class DummyClass:
            def __init__(self):
                self.attr = "original"

        obj = DummyClass()
        mocked = mock_attributes(obj, {"attr": "overridden"})

        # The mocked value should take precedence
        assert mocked.attr == "overridden"

    def test_mock_attributes_stores_original(self):
        """Test that the original object is accessible through the wrapper."""

        class DummyClass:
            def __init__(self):
                self.value = 100

        obj = DummyClass()
        mocked = mock_attributes(obj, {})

        # The wrapper should store the original object
        assert mocked.original is obj
        assert mocked.original.value == 100
