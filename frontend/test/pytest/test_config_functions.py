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

"""Unit tests for device toml config parsing and validation."""

from os.path import join
from tempfile import TemporaryDirectory
from textwrap import dedent

import pennylane as qml
import pytest

from catalyst.device import QJITDevice
from catalyst.utils.exceptions import CompileError
from catalyst.utils.runtime import check_no_overlap, validate_device_capabilities
from catalyst.utils.toml import (
    DeviceCapabilities,
    ProgramFeatures,
    TOMLDocument,
    check_quantum_control_flag,
    get_decomposable_gates,
    get_matrix_decomposable_gates,
    get_native_ops,
    load_device_capabilities,
    pennylane_operation_set,
    read_toml_file,
)


class DeviceToBeTested(qml.QubitDevice):
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


def get_test_config(config_text: str) -> TOMLDocument:
    """Parse test config into the TOMLDocument structure"""
    with TemporaryDirectory() as d:
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(config_text)
        config = read_toml_file(toml_file)
        return config


def get_test_device_capabilities(
    program_features: ProgramFeatures, config_text: str
) -> DeviceCapabilities:
    """Parse test config into the DeviceCapabilities structure"""
    config = get_test_config(config_text)
    device_capabilities = load_device_capabilities(config, program_features, "dummy")
    return device_capabilities


@pytest.mark.parametrize("schema", ALL_SCHEMAS)
def test_config_qjit_incompatible_device(schema):
    """Test error is raised if checking for qjit compatibility and field is false in toml file."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            f"""
                schema = {schema}
                [compilation]
                qjit_compatible = false
            """
        ),
    )

    name = DeviceToBeTested.name
    with pytest.raises(
        CompileError,
        match=f"Attempting to compile program for incompatible device '{name}'",
    ):
        validate_device_capabilities(DeviceToBeTested(), device_capabilities)


def test_get_observables_schema1():
    """Test observables are properly obtained from the toml schema 1."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            r"""
                schema = 1
                [operators]
                observables = [ "PauliX" ]
            """
        ),
    )
    assert {"PauliX"} == pennylane_operation_set(device_capabilities.native_obs)


def test_get_observables_schema2():
    """Test observables are properly obtained from the toml schema 2."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            r"""
                schema = 2
                [operators.observables]
                PauliX = { }
            """
        ),
    )
    assert {"PauliX"} == pennylane_operation_set(device_capabilities.native_obs)


def test_get_native_ops_schema1_no_qcontrol():
    """Test native gates are properly obtained from the toml."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            r"""
                schema = 1
                [[operators.gates]]
                native = [ "PauliX" ]
                [compilation]
                quantum_control = false
            """
        ),
    )
    assert {"PauliX"} == pennylane_operation_set(device_capabilities.native_ops)


def test_get_native_ops_schema1_qcontrol():
    """Test native gates are properly obtained from the toml."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            r"""
                schema = 1
                [[operators.gates]]
                native = [ "PauliZ" ]
                [compilation]
                quantum_control = true
            """
        ),
    )
    assert {"PauliZ", "C(PauliZ)"} == pennylane_operation_set(device_capabilities.native_ops)


@pytest.mark.parametrize("qadjoint", [True, False])
def test_get_native_ops_schema1_qadjoint(qadjoint):
    """Test native gates are properly obtained from the toml."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            rf"""
                schema = 1
                [[operators.gates]]
                native = [ "PauliZ" ]
                [compilation]
                quantum_adjoint = {str(qadjoint).lower()}
            """
        ),
    )
    assert device_capabilities.native_ops["PauliZ"].invertible is qadjoint


def test_get_native_ops_schema2():
    """Test native gates are properly obtained from the toml."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            r"""
                schema = 2
                [operators.gates.native]
                PauliX = { properties = [ 'controllable' ] }
                PauliY = { }
            """
        ),
    )

    assert {"PauliX", "C(PauliX)", "PauliY"} == pennylane_operation_set(
        device_capabilities.native_ops
    )


def test_get_native_ops_schema2_optional_shots():
    """Test native gates are properly obtained from the toml."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(True),
        dedent(
            r"""
                schema = 2
                [operators.gates.native]
                PauliX = { condition = ['finiteshots'] }
                PauliY = { condition = ['analytic'] }
            """
        ),
    )
    assert "PauliX" in device_capabilities.native_ops
    assert "PauliY" not in device_capabilities.native_ops


def test_get_native_ops_schema2_optional_noshots():
    """Test native gates are properly obtained from the toml."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            r"""
                schema = 2
                [operators.gates.native]
                PauliX = { condition = ['finiteshots'] }
                PauliY = { condition = ['analytic'] }
            """
        ),
    )
    assert "PauliX" not in device_capabilities.native_ops
    assert "PauliY" in device_capabilities.native_ops


def test_get_decomp_gates_schema1():
    """Test native decomposition gates are properly obtained from the toml."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            """
                schema = 1
                [[operators.gates]]
                decomp = ["PauliX", "PauliY"]
            """
        ),
    )

    assert "PauliX" in device_capabilities.to_decomp_ops
    assert "PauliY" in device_capabilities.to_decomp_ops
    assert "PauliZ" not in device_capabilities.to_decomp_ops


def test_get_decomp_gates_schema2():
    """Test native decomposition gates are properly obtained from the toml."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            """
            schema = 2
            [operators.gates]
            decomp = ["PauliX", "PauliY"]
        """
        ),
    )

    assert "PauliX" in device_capabilities.to_decomp_ops
    assert "PauliY" in device_capabilities.to_decomp_ops


def test_get_matrix_decomposable_gates_schema1():
    """Test native matrix gates are properly obtained from the toml."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            """
            schema = 1
            [[operators.gates]]
            matrix = ["PauliX", "PauliY"]
        """
        ),
    )

    assert "PauliX" in device_capabilities.to_matrix_ops
    assert "PauliY" in device_capabilities.to_matrix_ops


def test_get_matrix_decomposable_gates_schema2():
    """Test native matrix gates are properly obtained from the toml."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            r"""
            schema = 2
            [operators.gates.matrix]
            PauliZ = {}
        """
        ),
    )

    assert "PauliZ" in device_capabilities.to_matrix_ops


def test_check_overlap_msg():
    """Test error is raised if there is an overlap in sets."""
    msg = "Device 'test' has overlapping gates."
    with pytest.raises(CompileError, match=msg):
        check_no_overlap(["A"], ["A"], ["A"], device_name="test")


def test_config_invalid_attr():
    """Check the gate condition handling logic"""
    with pytest.raises(
        CompileError, match="Configuration for gate 'TestGate' has unknown attributes"
    ):
        get_test_device_capabilities(
            ProgramFeatures(False),
            dedent(
                r"""
                    schema = 2
                    [operators.gates.native]
                    TestGate = { unknown_attribute = 33 }
                """
            ),
        )


def test_config_invalid_condition_unknown():
    """Check the gate condition handling logic"""
    with pytest.raises(
        CompileError, match="Configuration for gate 'TestGate' has unknown conditions"
    ):
        get_test_device_capabilities(
            ProgramFeatures(True),
            dedent(
                r"""
                    schema = 2
                    [operators.gates.native]
                    TestGate = { condition = ["unknown", "analytic"] }
                """
            ),
        )


def test_config_invalid_property_unknown():
    """Check the gate condition handling logic"""
    with pytest.raises(
        CompileError, match="Configuration for gate 'TestGate' has unknown properties"
    ):
        get_test_device_capabilities(
            ProgramFeatures(True),
            dedent(
                r"""
                    schema = 2
                    [operators.gates.native]
                    TestGate = { properties = ["unknown", "invertible"] }
                """
            ),
        )


@pytest.mark.parametrize("shots", [True, False])
def test_config_invalid_condition_duplicate(shots):
    """Check the gate condition handling logic"""
    with pytest.raises(CompileError, match="Configuration for gate 'TestGate'"):
        get_test_device_capabilities(
            ProgramFeatures(shots),
            dedent(
                r"""
                    schema = 2
                    [operators.gates.native]
                    TestGate = { condition = ["finiteshots", "analytic"] }
                """
            ),
        )


def test_config_qjit_device_operations():
    """Check the gate condition handling logic"""
    capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            r"""
                schema = 2
                [operators.gates.native]
                PauliX = {}
                [operators.observables]
                PauliY = {}
            """
        ),
    )
    qjit_device = QJITDevice(capabilities, shots=1000, wires=2)
    assert "PauliX" in qjit_device.operations
    assert "PauliY" in qjit_device.observables


def test_config_unsupported_schema():
    """Test native matrix gates are properly obtained from the toml."""
    program_features = ProgramFeatures(False)
    config_text = dedent(
        r"""
            schema = 999
        """
    )
    config = get_test_config(config_text)

    with pytest.raises(CompileError):
        get_test_device_capabilities(program_features, config_text)

    with pytest.raises(CompileError):
        get_matrix_decomposable_gates(config, program_features)

    with pytest.raises(CompileError):
        get_decomposable_gates(config, program_features)

    with pytest.raises(CompileError):
        get_native_ops(config, program_features)

    with pytest.raises(CompileError):
        check_quantum_control_flag(config)


if __name__ == "__main__":
    pytest.main([__file__])
