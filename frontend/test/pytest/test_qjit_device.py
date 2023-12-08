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

from catalyst.utils.exceptions import CompileError
from catalyst.utils.patching import Patcher
from catalyst.utils.runtime import (
    check_no_overlap,
    check_qjit_compatibility,
    check_device_config,
    get_native_gates,
    get_decomposable_gates,
    get_matrix_decomposable_gates,
)
from catalyst.utils.toml import toml_load


def test_toml_file():
    with tempfile.NamedTemporaryFile() as f:
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
            CompileError, match=f"Attempting to compile program for incompatible device {name}."
        ):
            check_qjit_compatibility(LightningQubit, config)


def test_device_has_config_attr():
    name = LightningQubit.name
    msg = f"Attempting to compile program for incompatible device {name}."
    with pytest.raises(CompileError, match=msg):
        check_device_config(LightningQubit)


def test_device_with_invalid_config_attr():
    name = LightningQubit.name
    with tempfile.NamedTemporaryFile() as f:
        f.close()
        setattr(LightningQubit, "config", Path(f.name))
        msg = f"Attempting to compile program for incompatible device {name}."
        with pytest.raises(CompileError, match=msg):
            check_device_config(LightningQubit)
        delattr(LightningQubit, "config")


def test_get_native_gates():
    with tempfile.NamedTemporaryFile(mode="w+") as f:
        test_gates = ["TestNativeGate"]
        payload = f"""
[[operations.gates]]
native = {str(test_gates)}
        """
        f.write(payload)
        f.flush()
        f.seek(0)
        config = toml_load(f)
        f.close()
    assert test_gates == get_native_gates(config)


def test_get_decomp_gates():
    with tempfile.NamedTemporaryFile(mode="w+") as f:
        test_gates = ["TestDecompGate"]
        payload = f"""
[[operations.gates]]
decomp = {str(test_gates)}
        """
        f.write(payload)
        f.flush()
        f.seek(0)
        config = toml_load(f)
        f.close()
    assert test_gates == get_decomposable_gates(config)


def test_get_matrix_decomposable_gates():
    with tempfile.NamedTemporaryFile(mode="w+") as f:
        test_gates = ["TestMatrixGate"]
        payload = f"""
[[operations.gates]]
matrix = {str(test_gates)}
        """
        f.write(payload)
        f.flush()
        f.seek(0)
        config = toml_load(f)
        f.close()
    assert test_gates == get_matrix_decomposable_gates(config)


def test_check_overlap_msg():
    msg = "Device has overlapping gates in native and decomposable sets."
    with pytest.raises(CompileError, match=msg):
        check_no_overlap(["A"], ["A"], ["A"])
