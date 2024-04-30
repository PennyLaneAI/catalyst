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
Pytest configuration file for OQC test suite.
"""
# pylint: disable=unused-import
import os

import pytest

try:
    import qcaas_client

    oqc_available = True
except ImportError:
    oqc_available = False


@pytest.fixture()
def set_dummy_oqc_env():
    """Set OQC env var."""
    os.environ["OQC_EMAIL"] = "hello@world.com"
    os.environ["OQC_PASSWORD"] = "abcd"
    os.environ["OQC_URL"] = "https://qcaas.oqc.app/"

    yield

    del os.environ["OQC_EMAIL"]
    del os.environ["OQC_PASSWORD"]
    del os.environ["OQC_URL"]


def pytest_collection_modifyitems(items):
    """A pytest items modifier method"""
    if not oqc_available:
        # If OQC QCAAS is not installed, mark all collected tests to be skipped.
        for item in items:
            item.add_marker(pytest.mark.skip(reason="OQC qcaas is not installed"))
