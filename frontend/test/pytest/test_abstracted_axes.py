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

"""
Tests for abstracted axes
"""
from catalyst import qjit
import jax
from numpy.testing import assert_allclose

class TestBasicInterface:
    """Test thas abstracted_axes kwarg does not change any functionality for the time being
    """

    def test_abstracted_axes_dictionary(self):
        """This is a temporary test while dynamism is in development."""

        @qjit(abstracted_axes={0: "n"})
        def identity(a):
            return a

        param = jax.numpy.array([1, 2, 3])
        result = identity(param)
        assert_allclose(param, result)

