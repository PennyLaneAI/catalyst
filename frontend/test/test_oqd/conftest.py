# Copyright 2024 Xanadu Quantum Technologies Inc.

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

import pytest


def pytest_addoption(parser):
    """Add pytest custom options."""
    parser.addoption(
        "--skip-oqd",
        action="store",
        default="false",
        choices=["true", "false"],
        help="Skip the OQD test suite (true/false)",
    )


def pytest_collection_modifyitems(config, items):
    """A pytest items modifier method"""
    if config.getoption("--skip-oqd") == "true":
        skip_oqd = pytest.mark.skip(reason="Skipping the OQD test suite")
        for item in items:
            if "oqd" in item.nodeid:
                item.add_marker(skip_oqd)
