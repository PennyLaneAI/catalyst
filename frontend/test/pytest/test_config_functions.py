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

"""Unit tests for functions to check config validity."""

import tempfile
from pathlib import Path

import pennylane as qml
import pytest

from catalyst.utils.exceptions import CompileError
from catalyst.utils.runtime import (
    check_device_config,
    check_full_overlap,
    check_no_overlap,
    check_qjit_compatibility,
    get_decomposable_gates,
    get_matrix_decomposable_gates,
    get_native_gates,
)
from catalyst.utils.toml import toml_load


class DummyDevice(qml.QubitDevice):
    """Test device"""

    name = "Test Device"
    short_name = "test.device"


def test_toml_file():
    """Test error is raised if checking for qjit compatibility and field is false in toml file."""
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

        name = DummyDevice.name
        with pytest.raises(
            CompileError, match=f"Attempting to compile program for incompatible device {name}."
        ):
            check_qjit_compatibility(DummyDevice, config)


def test_device_has_config_attr():
    """Test error is raised when device has no config attr."""
    name = DummyDevice.name
    msg = f"Attempting to compile program for incompatible device {name}."
    with pytest.raises(CompileError, match=msg):
        check_device_config(DummyDevice)


def test_device_with_invalid_config_attr():
    """Test error is raised when device has invalid config attr."""
    name = DummyDevice.name
    with tempfile.NamedTemporaryFile(mode="w+b") as f:
        f.close()
        setattr(DummyDevice, "config", Path(f.name))
        msg = f"Attempting to compile program for incompatible device {name}."
        with pytest.raises(CompileError, match=msg):
            check_device_config(DummyDevice)
        delattr(DummyDevice, "config")


def test_get_native_gates():
    """Test native gates are properly obtained from the toml."""
    with tempfile.NamedTemporaryFile(mode="w+b") as f:
        test_gates = ["TestNativeGate"]
        payload = f"""
[[operators.gates]]
native = {str(test_gates)}
        """
        f.write(str.encode(payload))
        f.flush()
        f.seek(0)
        config = toml_load(f)
        f.close()
    assert test_gates == get_native_gates(config)


def test_get_decomp_gates():
    """Test native decomposition gates are properly obtained from the toml."""
    with tempfile.NamedTemporaryFile(mode="w+b") as f:
        test_gates = ["TestDecompGate"]
        payload = f"""
[[operators.gates]]
decomp = {str(test_gates)}
        """
        f.write(str.encode(payload))
        f.flush()
        f.seek(0)
        config = toml_load(f)
        f.close()
    assert test_gates == get_decomposable_gates(config)


def test_get_matrix_decomposable_gates():
    """Test native matrix gates are properly obtained from the toml."""
    with tempfile.NamedTemporaryFile(mode="w+b") as f:
        test_gates = ["TestMatrixGate"]
        payload = f"""
[[operators.gates]]
matrix = {str(test_gates)}
        """
        f.write(str.encode(payload))
        f.flush()
        f.seek(0)
        config = toml_load(f)
        f.close()
    assert test_gates == get_matrix_decomposable_gates(config)


def test_check_overlap_msg():
    """Test error is raised if there is an overlap in sets."""
    msg = "Device has overlapping gates in native and decomposable sets."
    with pytest.raises(CompileError, match=msg):
        check_no_overlap(["A"], ["A"], ["A"])


def test_check_full_overlap():
    """Test that if there is no full overlap of operations, then an error is raised."""

    class Device:
        operations = ["A", "B", "C"]

    msg = f"Gates in qml.device.operations and specification file do not match"
    with pytest.raises(CompileError, match=msg):
        check_full_overlap(Device(), ["A", "A", "A"], ["B", "B"])


if __name__ == "__main__":
    pytest.main([__file__])
