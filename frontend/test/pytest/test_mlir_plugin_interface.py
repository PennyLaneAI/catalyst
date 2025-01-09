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

"""Testing interface around main plugin functionality"""

from pathlib import Path

import pytest

import catalyst


def test_path_does_not_exists():
    """Test what happens when a pass_plugin is given an path that does not exist"""

    with pytest.raises(FileNotFoundError, match="does not exist"):
        catalyst.apply_pass_plugin("this-path-does-not-exist", "this-pass-also-doesnt-exists")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        catalyst.apply_pass_plugin(Path("this-path-does-not-exist"), "this-pass-also-doesnt-exists")
