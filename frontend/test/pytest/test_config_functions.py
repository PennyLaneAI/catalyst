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

from os.path import join
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

import pennylane as qml
import pytest

from catalyst.utils.exceptions import CompileError
from catalyst.utils.runtime import (  # check_device_config,
    check_full_overlap,
    check_no_overlap,
    get_decomposable_gates,
    get_matrix_decomposable_gates,
    get_native_gates_PL,
    validate_config_with_device,
)
from catalyst.utils.toml import toml_load


class DummyDevice(qml.QubitDevice):
    """Test device"""

    name = "Dummy Device"
    short_name = "dummy.device"
    pennylane_requires = "0.33.0"
    version = "0.0.1"
    author = "Dummy"

    operations = []
    observables = []

    def apply(self, operations, **kwargs):
        """Unused"""
        raise RuntimeError("Only C/C++ interface is defined")


ALL_SCHEMAS = [1, 2]


@pytest.mark.parametrize("schema", ALL_SCHEMAS)
def test_validate_config_with_device(schema):
    """Test error is raised if checking for qjit compatibility and field is false in toml file."""
    with TemporaryDirectory() as d:
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    f"""
                        schema = {schema}
                        [compilation]
                        qjit_compatible = false
                    """
                )
            )
        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)

        device = DummyDevice()
        with pytest.raises(
            CompileError,
            match=f"Attempting to compile program for incompatible device '{device.name}'",
        ):
            validate_config_with_device(device, config)


def test_get_native_gates_schema1():
    """Test native gates are properly obtained from the toml."""
    with TemporaryDirectory() as d:
        test_deduced_gates = {"C(TestNativeGate)", "TestNativeGate"}

        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    r"""
                        schema = 1
                        [[operators.gates]]
                        native = [ "TestNativeGate" ]
                    """
                )
            )
        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)
    assert test_deduced_gates == get_native_gates_PL(config)


def test_get_native_gates_schema2():
    """Test native gates are properly obtained from the toml."""
    with TemporaryDirectory() as d:
        test_deduced_gates = {"C(TestNativeGate1)", "TestNativeGate1", "TestNativeGate2"}

        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    r"""
                        schema = 2
                        [operators.gates.native]
                        TestNativeGate1 = { controllable = true }
                        TestNativeGate2 = { }
                    """
                )
            )
        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)
    assert test_deduced_gates == get_native_gates_PL(config)


def test_get_decomp_gates_schema1():
    """Test native decomposition gates are properly obtained from the toml."""
    with TemporaryDirectory() as d:
        test_gates = {"TestDecompGate"}
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    f"""
                        schema = 1
                        [[operators.gates]]
                        decomp = {str(list(test_gates))}
                    """
                )
            )

        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)

    assert test_gates == get_decomposable_gates(config)


def test_get_decomp_gates_schema2():
    """Test native decomposition gates are properly obtained from the toml."""
    with TemporaryDirectory() as d:
        test_gates = {"TestDecompGate"}
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    f"""
                        schema = 2
                        [operators.gates]
                        decomp = {str(list(test_gates))}
                    """
                )
            )

        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)

    assert test_gates == get_decomposable_gates(config)


def test_get_matrix_decomposable_gates_schema1():
    """Test native matrix gates are properly obtained from the toml."""
    with TemporaryDirectory() as d:
        test_gates = {"TestMatrixGate"}
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    f"""
                        schema = 1
                        [[operators.gates]]
                        matrix = {str(list(test_gates))}
                    """
                )
            )

        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)

    assert test_gates == get_matrix_decomposable_gates(config)


def test_get_matrix_decomposable_gates_schema2():
    """Test native matrix gates are properly obtained from the toml."""
    with TemporaryDirectory() as d:
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    r"""
                        schema = 2
                        [[operators.gates.matrix.TestMatrixGate]]
                    """
                )
            )

        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)

    assert {"TestMatrixGate"} == get_matrix_decomposable_gates(config)


def test_check_overlap_msg():
    """Test error is raised if there is an overlap in sets."""
    msg = "Device has overlapping gates."
    with pytest.raises(CompileError, match=msg):
        check_no_overlap(["A"], ["A"], ["A"])


def test_check_full_overlap():
    """Test that if there is no full overlap of operations, then an error is raised."""

    msg = f"Gates in qml.device.operations and specification file do not match"
    with pytest.raises(CompileError, match=msg):
        check_full_overlap({"A", "B", "C", "C(X)"}, {"A", "A", "A", "B", "B", "Adjoint(Y)"})


if __name__ == "__main__":
    pytest.main([__file__])
