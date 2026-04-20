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

"""Test suite for the catalyst.python_interface.transforms.qecp.qec_code_lib module."""

import pytest

from catalyst.python_interface.transforms.qecp.qec_code_lib import QecCode

SUPPORTED_CODES = ["Steane"]


class TestQecCode:
    """Units tests for the QecCode class."""

    @pytest.mark.parametrize(
        "name, n, k, d", [("Steane", 7, 1, 3), ("Shor", 9, 1, 3), ("Surface_d3", 17, 1, 3)]
    )
    def test_constructor(self, name: str, n: int, k: int, d: int):
        qec_code = QecCode(name, n, k, d)

        assert qec_code.name == name
        assert qec_code.n == n
        assert qec_code.k == k
        assert qec_code.d == d

    @pytest.mark.parametrize("name", SUPPORTED_CODES)
    def test_get(self, name: str):
        qec_code = QecCode.get(name)

        assert qec_code.name == name
