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


"""
Pytest configuration file for OQD test suite.
"""
import os
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture()
def temporary_openapl_file():
    """Ensure the OpenAPL output file is clean before and after each test."""
    with TemporaryDirectory() as temp_dir:
        openapl_file = os.path.join(temp_dir, "__openapl__output.json")
        yield openapl_file
