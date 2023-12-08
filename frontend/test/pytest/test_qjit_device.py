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

from pennylane_lightning.lightning_qubit.lightning_qubit import LightningQubit
import pytest

from catalyst.pennylane_extensions import QJITDevice
from catalyst.utils.exceptions import CompileError
from catalyst.utils.runtime import check_qjit_compatibility
from catalyst.utils.toml import toml_load


def test_toml_file():
    with tempfile.NamedTemporaryFile(mode="w+b") as f:
        f.write(
            b"""
[compilation]
qjit_compatible = false
        """
        )
        f.flush()
        f.seek(0)
        config = toml_load(f)
        f.close()

        name = LightningQubit.name
        with pytest.raises(
            CompileError, match=f"Attempting to compile program for incompatible device {name}"
        ):
            check_qjit_compatibility(LightningQubit, config)


def test_toml_file_exists():
    with pytest.raises(CompileError, match="foo is not supported for compilation at the moment"):
        QJITDevice._check_config_exists(Path("this-file-does-not-exist"), "foo")
