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
Unit tests for the Catalyst package in general.
"""

# pylint: disable=import-outside-toplevel

import sys

import pytest


def test_jaxlib_mismatch(monkeypatch):
    """Test import warning if jaxlib package version doesn't match expected."""
    import jaxlib

    monkeypatch.setattr(jaxlib, "__version__", "0.0")
    monkeypatch.delitem(sys.modules, "catalyst")

    with pytest.warns(UserWarning, match="version mismatch for the installed 'jaxlib' package"):
        import catalyst  # pylint: disable=unused-import


if __name__ == "__main__":
    pytest.main(["-x", __file__])
