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

import pytest

from catalyst.utils.exceptions import CompileError
from catalyst.utils.toml import (
    ALL_SUPPORTED_SCHEMAS,
    DeviceCapabilities,
    ProgramFeatures,
    TOMLDocument,
    load_device_capabilities,
    read_toml_file,
)


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
    device_capabilities = load_device_capabilities(config, program_features)
    return device_capabilities


@pytest.mark.parametrize("schema", ALL_SUPPORTED_SCHEMAS)
def test_get_observables(schema):
    """Test observables are properly obtained."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            f"""
                schema = {schema}
                [operators.observables]
                PauliX = {{ }}
            """
        ),
    )
    assert len(device_capabilities.native_obs) == 1
    assert "PauliX" in device_capabilities.native_obs


@pytest.mark.parametrize("schema", ALL_SUPPORTED_SCHEMAS)
def test_get_native_ops(schema):
    """Test native gates are properly obtained from the toml."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            f"""
                schema = {schema}
                [operators.gates.native]
                PauliX = {{ properties = [ 'controllable' ] }}
                PauliY = {{ }}
            """
        ),
    )

    assert len(device_capabilities.native_ops) == 2
    assert {"PauliX", "PauliY"}.issubset(device_capabilities.native_ops)
    assert device_capabilities.native_ops["PauliX"].controllable


@pytest.mark.parametrize("schema", ALL_SUPPORTED_SCHEMAS)
def test_get_native_ops_optional_shots(schema):
    """Test native gates are properly obtained from the toml."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(True),
        dedent(
            f"""
                schema = {schema}
                [operators.gates.native]
                PauliX = {{ condition = ['finiteshots'] }}
                PauliY = {{ condition = ['analytic'] }}
            """
        ),
    )
    assert "PauliX" in device_capabilities.native_ops
    assert "PauliY" not in device_capabilities.native_ops


@pytest.mark.parametrize("schema", ALL_SUPPORTED_SCHEMAS)
def test_get_native_ops_optional_noshots(schema):
    """Test native gates are properly obtained from the toml."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            f"""
                schema = {schema}
                [operators.gates.native]
                PauliX = {{ condition = ['finiteshots'] }}
                PauliY = {{ condition = ['analytic'] }}
            """
        ),
    )
    assert "PauliX" not in device_capabilities.native_ops
    assert "PauliY" in device_capabilities.native_ops


@pytest.mark.parametrize("schema", ALL_SUPPORTED_SCHEMAS)
def test_get_decomp_gates(schema):
    """Test native decomposition gates are properly obtained from the toml."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            f"""
            schema = {schema}
            [operators.gates]
            decomp = ["PauliX", "PauliY"]
        """
        ),
    )

    assert "PauliX" in device_capabilities.to_decomp_ops
    assert "PauliY" in device_capabilities.to_decomp_ops


@pytest.mark.parametrize("schema", ALL_SUPPORTED_SCHEMAS)
def test_get_matrix_decomposable_gates(schema):
    """Test native matrix gates are properly obtained from the toml."""
    device_capabilities = get_test_device_capabilities(
        ProgramFeatures(False),
        dedent(
            f"""
            schema = {schema}
            [operators.gates.matrix]
            PauliZ = {{}}
        """
        ),
    )

    assert "PauliZ" in device_capabilities.to_matrix_ops


@pytest.mark.parametrize("schema", ALL_SUPPORTED_SCHEMAS)
def test_config_invalid_attr(schema):
    """Check the gate condition handling logic"""
    with pytest.raises(
        CompileError, match="Configuration for gate 'TestGate' has unknown attributes"
    ):
        get_test_device_capabilities(
            ProgramFeatures(False),
            dedent(
                f"""
                    schema = {schema}
                    [operators.gates.native]
                    TestGate = {{ unknown_attribute = 33 }}
                """
            ),
        )


@pytest.mark.parametrize("schema", ALL_SUPPORTED_SCHEMAS)
def test_config_invalid_condition_unknown(schema):
    """Check the gate condition handling logic"""
    with pytest.raises(
        CompileError, match="Configuration for gate 'TestGate' has unknown conditions"
    ):
        get_test_device_capabilities(
            ProgramFeatures(True),
            dedent(
                f"""
                    schema = {schema}
                    [operators.gates.native]
                    TestGate = {{ condition = ["unknown", "analytic"] }}
                """
            ),
        )


@pytest.mark.parametrize("schema", ALL_SUPPORTED_SCHEMAS)
def test_config_invalid_property_unknown(schema):
    """Check the gate condition handling logic"""
    with pytest.raises(
        CompileError, match="Configuration for gate 'TestGate' has unknown properties"
    ):
        get_test_device_capabilities(
            ProgramFeatures(True),
            dedent(
                f"""
                    schema = {schema}
                    [operators.gates.native]
                    TestGate = {{ properties = ["unknown", "invertible"] }}
                """
            ),
        )


@pytest.mark.parametrize("schema", ALL_SUPPORTED_SCHEMAS)
def test_config_invalid_condition_duplicate_true(schema):
    """Check the gate condition handling logic for True"""
    with pytest.raises(CompileError, match="Configuration for gate 'TestGate'"):
        get_test_device_capabilities(
            ProgramFeatures(True),
            dedent(
                f"""
                    schema = {schema}
                    [operators.gates.native]
                    TestGate = {{ condition = ["finiteshots", "analytic"] }}
                """
            ),
        )


@pytest.mark.parametrize("schema", ALL_SUPPORTED_SCHEMAS)
def test_config_invalid_condition_duplicate_false(schema):
    """Check the gate condition handling logic for False"""
    with pytest.raises(CompileError, match="Configuration for gate 'TestGate'"):
        get_test_device_capabilities(
            ProgramFeatures(False),
            dedent(
                f"""
                    schema = {schema}
                    [operators.gates.native]
                    TestGate = {{ condition = ["finiteshots", "analytic"] }}
                """
            ),
        )


@pytest.mark.parametrize("schema", [1, 999])
def test_config_unsupported_schema(schema):
    """Test unsupported schema version."""
    program_features = ProgramFeatures(False)
    config_text = dedent(
        f"""
            schema = {schema}
        """
    )

    with pytest.raises(AssertionError, match="Unsupported config schema"):
        get_test_device_capabilities(program_features, config_text)


if __name__ == "__main__":
    pytest.main([__file__])
