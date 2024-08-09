# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for contents of c_template
"""
import numpy as np

from catalyst.utils.c_template import CType, CVariable


class TestCType:
    """Unit tests for CType"""

    def test_get_name(self):
        """Test for name generation of types."""
        # pylint: disable=protected-access
        assert CType._get_name("int", 3) == "struct memref_intx3_t"

    def test_get_sizes_and_strides_zero_rank(self):
        """Test zero sizes and strides."""
        # pylint: disable=protected-access
        assert CType._get_template_for_sizes_and_strides(0) == ""

    def test_get_sizes_and_strides_nonzero_rank(self):
        """Test non-zero sizes and strides."""
        # pylint: disable=protected-access
        template = CType._get_template_for_sizes_and_strides((3, 2, 1))
        assert "size_t sizes[3];" in template
        assert "size_t strides[3];" in template

    def test_get_definition(self):
        """Test type definition."""
        # pylint: disable=protected-access
        template = CType._get_definition("struct foo", "mytype", 0)
        assert "struct foo" in template
        assert "mytype *allocated;" in template
        assert "mytype *aligned;" in template


class TestCVariable:
    """Unit tests for CVariable."""

    def test_get_buffer_name(self):
        """Test get name of buffer."""
        # pylint: disable=protected-access
        assert CVariable._get_buffer_name(0) == "buff_0"

    def test_get_variable_name(self):
        """Test get name of variable."""
        # pylint: disable=protected-access
        assert CVariable._get_variable_name(0) == "arg_0"

    def test_get_buffer_size_zero_rank(self):
        """Test get buffer size for zero rank."""
        x = np.array(1)
        # pylint: disable=protected-access
        assert CVariable._get_buffer_size(x) == 0

    def test_get_buffer_size_nonzero_rank(self):
        """Test get buffer size for non-zero rank."""
        x = np.array([1])
        # pylint: disable=protected-access
        assert CVariable._get_buffer_size(x) == 1

    def test_get_array_data_zero_rank(self):
        """Test get array data for zero rank."""
        x = np.array(1)
        # pylint: disable=protected-access
        assert CVariable._get_array_data(x) == x

    def test_get_array_data_nonzero_rank(self):
        """Test get array data for non-zero rank."""
        x = np.array([1])
        # pylint: disable=protected-access
        assert CVariable._get_array_data(x) == "1"

    def test_get_sizes_zero_rank(self):
        """Test get sizes for zero rank."""
        x = np.array(1)
        # pylint: disable=protected-access
        assert CVariable._get_sizes(x) == ""

    def test_get_sizes_nonzero_rank(self):
        """Test get sizes for non-zero rank."""
        x = np.array([1])
        # pylint: disable=protected-access
        assert CVariable._get_sizes(x) == "1"

    def test_get_strides_zero_rank(self):
        """Test get strides for zero rank."""
        x = np.array(1)
        # pylint: disable=protected-access
        assert CVariable._get_strides(x) == ""

    def test_get_strides_nonzero_rank(self):
        """Test get strides for non-zero rank."""
        x = np.array([1])
        # pylint: disable=protected-access
        assert CVariable._get_strides(x) == "1"
