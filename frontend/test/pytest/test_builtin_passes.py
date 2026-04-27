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
"""Tests the passes found in 'builtin_passes.py'"""

import pytest
from pennylane.transforms.core import Transform

from catalyst.passes import builtin_passes


@pytest.mark.parametrize("name", builtin_passes.__all__)
def test_exported_as_transform(name):
    """Tests that these passes are transform objects."""

    obj = getattr(builtin_passes, name)
    assert isinstance(obj, Transform)
