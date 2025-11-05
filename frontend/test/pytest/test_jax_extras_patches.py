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

from catalyst.jax_extras.patches import mock_attributes


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
