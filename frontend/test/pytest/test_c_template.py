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
import pennylane as qml
import pytest

from catalyst import qjit
from catalyst.debug import get_cmain
from catalyst.utils.c_template import CType, CVariable
from catalyst.utils.exceptions import CompileError


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
        template = CType._get_template_for_sizes_and_strides(3)
        assert "size_t sizes[3];" in template
        assert "size_t strides[3];" in template

    def test_get_definition(self):
        """Test type definition."""
        # pylint: disable=protected-access
        template = CType._get_definition("struct foo", "mytype", 0)
        assert "struct foo" in template
        assert "mytype* allocated;" in template
        assert "mytype* aligned;" in template


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


# pylint: disable=too-few-public-methods
class TestCProgramGeneration:
    """Test C Program generation"""

    def test_program_generation(self):
        """Test C Program generation"""
        dev = qml.device("lightning.qubit", wires=2)

        @qjit
        @qml.qnode(dev)
        def f(x: float):
            """Returns two states."""
            qml.RX(x, wires=1)
            return qml.state(), qml.state()

        template = get_cmain(f, 4.0)
        assert "main" in template
        assert "struct result_t result_val;" in template
        assert "buff_0 = 4.0" in template
        assert "arg_0 = { &buff_0, &buff_0, 0 }" in template
        assert "_catalyst_ciface_jit_f(&result_val, &arg_0);" in template

    def test_program_without_return_nor_arguments(self):
        """Test program without return value nor arguments."""

        @qjit
        def f():
            """No-op function."""
            return None

        template = get_cmain(f)
        assert "struct result_t result_val;" not in template
        assert "buff_0" not in template
        assert "arg_0" not in template


class TestCProgramGenerationErrors:
    """Test errors raised from the c program generation feature."""

    def test_raises_error_if_tracing(self):
        """Test errors if c program generation requested during tracing."""

        @qjit
        def f(x: float):
            """Identity function."""
            return x

        with pytest.raises(CompileError, match="C interface cannot be generated"):

            @qjit
            def error_fn(x: float):
                """Should raise an error as we try to generate the C template during tracing."""
                return get_cmain(f, x)

    def test_error_non_qjit_object(self):
        """An error should be raised if the object supplied to the debug function is not a QJIT."""

        def f(x: float):
            """Identity function."""
            return x

        with pytest.raises(TypeError, match="First argument needs to be a 'QJIT' object"):
            get_cmain(f, 0.5)
