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
    get_device_config,
    validate_config_with_device,
)
from catalyst.utils.toml import (
    DeviceConfig,
    ProgramFeatures,
    pennylane_operation_set,
    read_toml_file,
)


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


def parse_test_config(program_features: ProgramFeatures, config_text: str) -> DeviceConfig:
    """Parse test config into the DeviceConfig structure"""
    with TemporaryDirectory() as d:
        toml_file = join(d, "test.toml")
        with open(toml_file, "w", encoding="utf-8") as f:
            f.write(config_text)
        config = read_toml_file(toml_file)
        device_config = get_device_config(config, program_features, "dummy")

    return device_config


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

        config = read_toml_file(toml_file)

        device = DummyDevice()
        with pytest.raises(
            CompileError,
            match=f"Attempting to compile program for incompatible device '{device.name}'",
        ):
            validate_config_with_device(device, config)


def test_get_observables_schema1():
    """Test observables are properly obtained from the toml schema 1."""
    device_config = parse_test_config(
        ProgramFeatures(False),
        dedent(
            r"""
                schema = 1
                [operators]
                observables = [ "PauliX" ]
            """
        ),
    )
    assert {"PauliX"} == pennylane_operation_set(device_config.observables)


def test_get_observables_schema2():
    """Test observables are properly obtained from the toml schema 2."""
    device_config = parse_test_config(
        ProgramFeatures(False),
        dedent(
            r"""
                schema = 2
                [operators.observables]
                PauliX = { }
            """
        ),
    )
    assert {"PauliX"} == pennylane_operation_set(device_config.observables)


def test_get_native_gates_schema1_no_qcontrol():
    """Test native gates are properly obtained from the toml."""
    device_config = parse_test_config(
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
    assert {"PauliX"} == pennylane_operation_set(device_config.native_gates)


def test_get_native_gates_schema1_qcontrol():
    """Test native gates are properly obtained from the toml."""
    device_config = parse_test_config(
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
    assert {"PauliZ", "C(PauliZ)"} == pennylane_operation_set(device_config.native_gates)


@pytest.mark.parametrize("qadjoint", [True, False])
def test_get_native_gates_schema1_qadjoint(qadjoint):
    """Test native gates are properly obtained from the toml."""
    device_config = parse_test_config(
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
    assert device_config.native_gates["PauliZ"].invertible is qadjoint


def test_get_native_gates_schema2():
    """Test native gates are properly obtained from the toml."""
    device_config = parse_test_config(
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

    assert {"PauliX", "C(PauliX)", "PauliY"} == pennylane_operation_set(device_config.native_gates)


def test_get_native_gates_schema2_optional_shots():
    """Test native gates are properly obtained from the toml."""
    device_config = parse_test_config(
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
    assert "PauliX" in device_config.native_gates
    assert "PauliY" not in device_config.native_gates


def test_get_native_gates_schema2_optional_noshots():
    """Test native gates are properly obtained from the toml."""
    device_config = parse_test_config(
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
    assert "PauliX" not in device_config.native_gates
    assert "PauliY" in device_config.native_gates


def test_get_decomp_gates_schema1():
    """Test native decomposition gates are properly obtained from the toml."""
    device_config = parse_test_config(
        ProgramFeatures(False),
        dedent(
            f"""
                    schema = 1
                    [[operators.gates]]
                    decomp = ["PauliX", "PauliY"]
                """
        ),
    )

    assert "PauliX" in device_config.decomp
    assert "PauliY" in device_config.decomp
    assert "PauliZ" not in device_config.decomp


def test_get_decomp_gates_schema2():
    """Test native decomposition gates are properly obtained from the toml."""
    device_config = parse_test_config(
        ProgramFeatures(False),
        dedent(
            f"""
            schema = 2
            [operators.gates]
            decomp = ["PauliX", "PauliY"]
        """
        ),
    )

    assert "PauliX" in device_config.decomp
    assert "PauliY" in device_config.decomp


def test_get_matrix_decomposable_gates_schema1():
    """Test native matrix gates are properly obtained from the toml."""
    device_config = parse_test_config(
        ProgramFeatures(False),
        dedent(
            f"""
            schema = 1
            [[operators.gates]]
            matrix = ["PauliX", "PauliY"]
        """
        ),
    )

    assert "PauliX" in device_config.matrix
    assert "PauliY" in device_config.matrix


def test_get_matrix_decomposable_gates_schema2():
    """Test native matrix gates are properly obtained from the toml."""
    device_config = parse_test_config(
        ProgramFeatures(False),
        dedent(
            r"""
            schema = 2
            [operators.gates.matrix]
            PauliZ = {}
        """
        ),
    )

    assert "PauliZ" in device_config.matrix


def test_check_overlap_msg():
    """Test error is raised if there is an overlap in sets."""
    msg = "Device 'test' has overlapping gates."
    with pytest.raises(CompileError, match=msg):
        check_no_overlap(["A"], ["A"], ["A"], device_name="test")


def test_check_full_overlap():
    """Test that if there is no full overlap of operations, then an error is raised."""

    msg = f"Gates in qml.device.operations and specification file do not match"
    with pytest.raises(CompileError, match=msg):
        check_full_overlap({"A", "B", "C", "C(X)"}, {"A", "B", "Adjoint(Y)"})


def test_config_invalid_attr():
    """Check the gate condition handling logic"""
    with pytest.raises(
        CompileError, match="Configuration for gate 'TestGate' has unknown attributes"
    ):
        parse_test_config(
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
        parse_test_config(
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
        parse_test_config(
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
        parse_test_config(
            ProgramFeatures(shots),
            dedent(
                r"""
                        schema = 2
                        [operators.gates.native]
                        TestGate = { condition = ["finiteshots", "analytic"] }
                    """
            ),
        )


def test_config_unsupported_schema():
    """Test native matrix gates are properly obtained from the toml."""

    with pytest.raises(CompileError):
        parse_test_config(
            ProgramFeatures(False),
            dedent(
                r"""
                        schema = 999
                    """
            ),
        )


if __name__ == "__main__":
    pytest.main([__file__])
