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
from tempfile import TemporaryDirectory
from textwrap import dedent

import pennylane as qml
import pytest

from catalyst.utils.exceptions import CompileError
from catalyst.utils.runtime import (
    check_full_overlap,
    check_no_overlap,
    check_quantum_control_flag,
    get_decomposable_gates,
    get_matrix_decomposable_gates,
    get_native_gates,
    get_pennylane_observables,
    get_pennylane_operations,
    validate_config_with_device,
)
from catalyst.utils.toml import check_adjoint_flag, toml_load


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


def test_get_observables_schema1():
    """Test observables are properly obtained from the toml schema 1."""
    with TemporaryDirectory() as d:
        test_deduced_gates = {"TestNativeGate"}

        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    r"""
                        schema = 1
                        [operators]
                        observables = [ "TestNativeGate" ]
                    """
                )
            )
        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)
    assert test_deduced_gates == get_pennylane_observables(config, False, "device_name")


def test_get_observables_schema2():
    """Test observables are properly obtained from the toml schema 2."""
    with TemporaryDirectory() as d:
        test_deduced_gates = {"TestNativeGate1"}

        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    r"""
                        schema = 2
                        [operators.observables]
                        TestNativeGate1 = { }
                    """
                )
            )
        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)
    assert test_deduced_gates == get_pennylane_observables(config, False, "device_name")


def test_get_native_gates_schema1_no_qcontrol():
    """Test native gates are properly obtained from the toml."""
    with TemporaryDirectory() as d:
        test_deduced_gates = {"TestNativeGate"}

        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    r"""
                        schema = 1
                        [[operators.gates]]
                        native = [ "TestNativeGate" ]
                        [compilation]
                        quantum_control = false
                    """
                )
            )
        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)
    assert test_deduced_gates == get_pennylane_operations(config, False, "device_name")


def test_get_native_gates_schema1_qcontrol():
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
                        [compilation]
                        quantum_control = true
                    """
                )
            )
        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)
    assert test_deduced_gates == get_pennylane_operations(config, False, "device_name")


def test_get_adjoint_schema2():
    """Test native gates are properly obtained from the toml."""
    with TemporaryDirectory() as d:

        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    r"""
                        schema = 2
                        [operators.gates.native]
                        TestNativeGate1 = { properties = [ 'invertible' ] }
                        TestNativeGate2 = { properties = [ 'invertible' ] }
                    """
                )
            )
        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)
    assert check_adjoint_flag(config, False)


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
                        TestNativeGate1 = { properties = [ 'controllable' ] }
                        TestNativeGate2 = { }
                    """
                )
            )
        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)
    assert test_deduced_gates == get_pennylane_operations(config, False, "device_name")


def test_get_native_gates_schema2_optional_shots():
    """Test native gates are properly obtained from the toml."""
    with TemporaryDirectory() as d:
        test_deduced_gates = {"TestNativeGate1"}

        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    r"""
                        schema = 2
                        [operators.gates.native]
                        TestNativeGate1 = { condition = ['finiteshots'] }
                        TestNativeGate2 = { condition = ['analytic'] }
                    """
                )
            )
        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)
    assert test_deduced_gates == get_pennylane_operations(config, True, "device_name")


def test_get_native_gates_schema2_optional_noshots():
    """Test native gates are properly obtained from the toml."""
    with TemporaryDirectory() as d:
        test_deduced_gates = {"TestNativeGate2"}
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    r"""
                        schema = 2
                        [operators.gates.native]
                        TestNativeGate1 = { condition = ['finiteshots'] }
                        TestNativeGate2 = { condition = ['analytic'] }
                    """
                )
            )
        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)
    assert test_deduced_gates == get_pennylane_operations(config, False, "device")


def test_get_decomp_gates_schema1():
    """Test native decomposition gates are properly obtained from the toml."""
    with TemporaryDirectory() as d:
        test_gates = {"TestDecompGate": {}}
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    f"""
                        schema = 1
                        [[operators.gates]]
                        decomp = {str(list(test_gates.keys()))}
                    """
                )
            )

        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)

    assert test_gates == get_decomposable_gates(config, False)


def test_get_decomp_gates_schema2():
    """Test native decomposition gates are properly obtained from the toml."""
    with TemporaryDirectory() as d:
        test_gates = {"TestDecompGate": {}}
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    f"""
                        schema = 2
                        [operators.gates]
                        decomp = {str(list(test_gates.keys()))}
                    """
                )
            )

        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)

    assert test_gates == get_decomposable_gates(config, False)


def test_get_matrix_decomposable_gates_schema1():
    """Test native matrix gates are properly obtained from the toml."""
    with TemporaryDirectory() as d:
        test_gates = {"TestMatrixGate": {}}
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    f"""
                        schema = 1
                        [[operators.gates]]
                        matrix = {str(list(test_gates.keys()))}
                    """
                )
            )

        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)

    assert test_gates == get_matrix_decomposable_gates(config, False)


def test_get_matrix_decomposable_gates_schema2():
    """Test native matrix gates are properly obtained from the toml."""
    with TemporaryDirectory() as d:
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    r"""
                        schema = 2
                        [operators.gates.matrix]
                        TestMatrixGate = {}
                    """
                )
            )

        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)

    assert {"TestMatrixGate": {}} == get_matrix_decomposable_gates(config, False)


def test_check_overlap_msg():
    """Test error is raised if there is an overlap in sets."""
    msg = "Device has overlapping gates."
    with pytest.raises(CompileError, match=msg):
        check_no_overlap(["A"], ["A"], ["A"])


def test_check_full_overlap():
    """Test that if there is no full overlap of operations, then an error is raised."""

    msg = f"Gates in qml.device.operations and specification file do not match"
    with pytest.raises(CompileError, match=msg):
        check_full_overlap({"A", "B", "C", "C(X)"}, {"A", "B", "Adjoint(Y)"})


def test_config_invalid_attr():
    """Check the gate condition handling logic"""
    with TemporaryDirectory() as d:
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    r"""
                        schema = 2
                        [operators.gates.native]
                        TestGate = { unknown_attribute = 33 }
                    """
                )
            )

        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)

        with pytest.raises(
            CompileError, match="Configuration for gate 'TestGate' has unknown attributes"
        ):
            get_native_gates(config, True)


def test_config_invalid_condition_unknown():
    """Check the gate condition handling logic"""
    with TemporaryDirectory() as d:
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    r"""
                        schema = 2
                        [operators.gates.native]
                        TestGate = { condition = ["unknown", "analytic"] }
                    """
                )
            )

        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)

        with pytest.raises(
            CompileError, match="Configuration for gate 'TestGate' has unknown conditions"
        ):
            get_native_gates(config, True)


def test_config_invalid_property_unknown():
    """Check the gate condition handling logic"""
    with TemporaryDirectory() as d:
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    r"""
                        schema = 2
                        [operators.gates.native]
                        TestGate = { properties = ["unknown", "invertible"] }
                    """
                )
            )

        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)

        with pytest.raises(
            CompileError, match="Configuration for gate 'TestGate' has unknown properties"
        ):
            get_native_gates(config, True)


def test_config_invalid_condition_duplicate():
    """Check the gate condition handling logic"""
    with TemporaryDirectory() as d:
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    r"""
                        schema = 2
                        [operators.gates.native]
                        TestGate = { condition = ["finiteshots", "analytic"] }
                    """
                )
            )

        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)

        with pytest.raises(CompileError, match="Configuration for gate 'TestGate'"):
            get_native_gates(config, True)

        with pytest.raises(CompileError, match="Configuration for gate 'TestGate'"):
            get_native_gates(config, False)


def test_config_unsupported_schema():
    """Test native matrix gates are properly obtained from the toml."""
    with TemporaryDirectory() as d:
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(
                dedent(
                    r"""
                        schema = 999
                    """
                )
            )

        with open(toml_file, encoding="utf-8") as f:
            config = toml_load(f)

        with pytest.raises(CompileError):
            check_quantum_control_flag(config)
        with pytest.raises(CompileError):
            get_native_gates(config, False)
        with pytest.raises(CompileError):
            get_decomposable_gates(config, False)
        with pytest.raises(CompileError):
            get_matrix_decomposable_gates(config, False)
        with pytest.raises(CompileError):
            get_pennylane_operations(config, False, "device_name")
        with pytest.raises(CompileError):
            check_adjoint_flag(config, False)


if __name__ == "__main__":
    pytest.main([__file__])
