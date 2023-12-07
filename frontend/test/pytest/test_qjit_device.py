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

import tempfile
from pathlib import Path

import pytest
from catalyst.pennylane_extensions import QJITDevice
from catalyst.utils.exceptions import CompileError


def test_toml_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(
            b"""
[compilation]
qjit_compatible = false
        """
        )
        f.close()

        p = Path(f.name)
        with pytest.raises(
            CompileError, match="Attempting to compile to device without qjit support"
        ):
            QJITDevice._check_qjit_compatible(p)


def test_toml_file_exists():
    with pytest.raises(CompileError, match="foo is not supported for compilation at the moment"):
        QJITDevice._check_config_exists(Path("this-file-does-not-exist"), "foo")
