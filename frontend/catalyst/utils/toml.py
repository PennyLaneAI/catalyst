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
"""
Module for abstracting which toml_load to use.
"""

import importlib.util
from dataclasses import dataclass, field
from functools import reduce
from itertools import repeat
from typing import Any, Dict, List, Set

from catalyst.utils.exceptions import CompileError

# TODO:
# Once Python version 3.11 is the oldest supported Python version, we can remove tomlkit
# and rely exclusively on tomllib.

# New in version 3.11
# https://docs.python.org/3/library/tomllib.html
tomllib = importlib.util.find_spec("tomllib")
tomlkit = importlib.util.find_spec("tomlkit")
# We need at least one of these to make sure we can read toml files.
if tomllib is None and tomlkit is None:  # pragma: nocover
    raise ImportError("Either tomllib or tomlkit need to be installed.")

# Give preference to tomllib
if tomllib:  # pragma: nocover
    from tomllib import load as toml_load

    TOMLDocument = Any
    TOMLException = Exception
else:  # pragma: nocover
    from tomlkit import TOMLDocument
    from tomlkit import load as toml_load
    from tomlkit.exceptions import TOMLKitError as TOMLException

ALL_SUPPORTED_SCHEMAS = [2]


def read_toml_file(toml_file: str) -> TOMLDocument:
    """Helper function opening toml file properly and reading it into a document"""
    with open(toml_file, "rb") as f:
        config = toml_load(f)
    return config


@dataclass
class OperationProperties:
    """Capabilities of a single operation"""

    invertible: bool = False
    controllable: bool = False
    differentiable: bool = False


def _intersect_properties(a: OperationProperties, b: OperationProperties) -> OperationProperties:
    """Calculate the intersection of OperationProperties"""
    return OperationProperties(
        invertible=a.invertible and b.invertible,
        controllable=a.controllable and b.controllable,
        differentiable=a.differentiable and b.differentiable,
    )


@dataclass
class DeviceCapabilities:  # pylint: disable=too-many-instance-attributes
    """Quantum device capabilities"""

    native_ops: Dict[str, OperationProperties] = field(default_factory=dict)
    to_decomp_ops: Dict[str, OperationProperties] = field(default_factory=dict)
    to_matrix_ops: Dict[str, OperationProperties] = field(default_factory=dict)
    native_obs: Dict[str, OperationProperties] = field(default_factory=dict)
    measurement_processes: Set[str] = field(default_factory=dict)
    qjit_compatible_flag: bool = False
    mid_circuit_measurement_flag: bool = False
    runtime_code_generation_flag: bool = False
    dynamic_qubit_management_flag: bool = False
    non_commuting_observables_flag: bool = False
    options: Dict[str, bool] = field(default_factory=dict)


def intersect_operations(
    a: Dict[str, OperationProperties], b: Dict[str, OperationProperties]
) -> Dict[str, OperationProperties]:
    """Intersects two sets of oepration properties"""
    return {k: _intersect_properties(a[k], b[k]) for k in (a.keys() & b.keys())}


def pennylane_operation_set(config_ops: Dict[str, OperationProperties]) -> Set[str]:
    """Returns a config section into a set of strings using PennyLane syntax"""
    ops = set()
    # Back-mapping from class names to string names
    for g, props in config_ops.items():
        ops.update({g})
        if props.controllable:
            ops.update({f"C({g})"})
        if props.invertible:
            ops.update({f"Adjoint({g})"})
        if props.controllable and props.invertible:
            ops.update({f"Adjoint(C({g}))"})
            ops.update({f"C(Adjoint({g}))"})
    return ops


@dataclass
class ProgramFeatures:
    """Program features, obtained from the user"""

    shots_present: bool


def _get_compilation_flag(config: TOMLDocument, flag_name: str) -> bool:
    """Get the flag in the toml document 'compilation' section."""
    return bool(config.get("compilation", {}).get(flag_name, False))


def _get_options(config: TOMLDocument) -> Dict[str, str]:
    """Get custom options sections"""
    return {str(k): str(v) for k, v in config.get("options", {}).items()}


def _parse_toml_section(
    config: TOMLDocument, path: List[str], program_features: ProgramFeatures
) -> Dict[str, dict]:
    """Parses the section of toml config file specified by `path`. Filters-out gates which don't
    match condition. For now the only condition we support is `shots_present`."""
    gates = {}
    analytic = "analytic"
    finiteshots = "finiteshots"
    try:
        iterable = reduce(lambda x, y: x[y], path, config)
    except TOMLException as _:  # pylint: disable=broad-exception-caught
        return {}
    gen = iterable.items() if hasattr(iterable, "items") else zip(iterable, repeat({}))
    for g, values in gen:
        unknown_attrs = set(values) - {"condition", "properties"}
        if len(unknown_attrs) > 0:
            raise CompileError(
                f"Configuration for gate '{str(g)}' has unknown attributes: {list(unknown_attrs)}"
            )
        properties = values.get("properties", {})
        unknown_props = set(properties) - {"invertible", "controllable", "differentiable"}
        if len(unknown_props) > 0:
            raise CompileError(
                f"Configuration for gate '{str(g)}' has unknown properties: {list(unknown_props)}"
            )
        if "condition" in values:
            # TODO: do not filter here. Parse the condition and then filter on demand instead.
            conditions = values["condition"]
            unknown_conditions = set(conditions) - {analytic, finiteshots}
            if len(unknown_conditions) > 0:
                raise CompileError(
                    f"Configuration for gate '{str(g)}' has unknown conditions: "
                    f"{list(unknown_conditions)}"
                )
            if all(c in conditions for c in [analytic, finiteshots]):
                raise CompileError(
                    f"Configuration for gate '{g}' can not contain both "
                    f"`{finiteshots}` and `{analytic}` conditions simultaniosly"
                )
            if analytic in conditions and not program_features.shots_present:
                gates[g] = values
            elif finiteshots in conditions and program_features.shots_present:
                gates[g] = values
        else:
            gates[g] = values
    return gates


def _get_observables(config: TOMLDocument, program_features: ProgramFeatures) -> Dict[str, dict]:
    """Override the set of supported observables."""
    return _parse_toml_section(config, ["operators", "observables"], program_features)


def _get_measurement_processes(
    config: TOMLDocument, program_features: ProgramFeatures
) -> Dict[str, dict]:
    """Get the measurements processes from the `native` section of the config."""
    return _parse_toml_section(config, ["measurement_processes"], program_features)


def _get_native_ops(config: TOMLDocument, program_features: ProgramFeatures) -> Dict[str, dict]:
    """Get the gates from the `native` section of the config."""
    return _parse_toml_section(config, ["operators", "gates", "native"], program_features)


def _get_decomposable_gates(
    config: TOMLDocument, program_features: ProgramFeatures
) -> Dict[str, dict]:
    """Get gates that will be decomposed according to PL's decomposition rules.

    Args:
        config (TOMLDocument): Configuration dictionary
    """
    return _parse_toml_section(config, ["operators", "gates", "decomp"], program_features)


def _get_matrix_decomposable_gates(
    config: TOMLDocument, program_features: ProgramFeatures
) -> Dict[str, dict]:
    """Get gates that will be decomposed to QubitUnitary.

    Args:
        config (TOMLDocument): Configuration dictionary
    """
    return _parse_toml_section(config, ["operators", "gates", "matrix"], program_features)


def _get_operation_properties(config_props: dict) -> OperationProperties:
    """Load operation properties from config"""
    properties = config_props.get("properties", {})
    return OperationProperties(
        invertible="invertible" in properties,
        controllable="controllable" in properties,
        differentiable="differentiable" in properties,
    )


def load_device_capabilities(
    config: TOMLDocument, program_features: ProgramFeatures
) -> DeviceCapabilities:
    """Load device capabilities from device config"""

    schema = int(config["schema"])
    assert schema in ALL_SUPPORTED_SCHEMAS, f"Unsupported config schema {schema}"

    native_gate_props = {}
    for g, props in _get_native_ops(config, program_features).items():
        native_gate_props[g] = _get_operation_properties(props)

    matrix_decomp_props = {}
    for g, props in _get_matrix_decomposable_gates(config, program_features).items():
        matrix_decomp_props[g] = _get_operation_properties(props)

    decomp_props = {}
    for g, props in _get_decomposable_gates(config, program_features).items():
        decomp_props[g] = _get_operation_properties(props)

    observable_props = {}
    for g, props in _get_observables(config, program_features).items():
        observable_props[g] = _get_operation_properties(props)

    measurements_props = set()
    for g, props in _get_measurement_processes(config, program_features).items():
        measurements_props.add(g)

    return DeviceCapabilities(
        native_ops=native_gate_props,
        to_decomp_ops=decomp_props,
        to_matrix_ops=matrix_decomp_props,
        native_obs=observable_props,
        measurement_processes=measurements_props,
        qjit_compatible_flag=_get_compilation_flag(config, "qjit_compatible"),
        mid_circuit_measurement_flag=_get_compilation_flag(config, "mid_circuit_measurement"),
        runtime_code_generation_flag=_get_compilation_flag(config, "runtime_code_generation"),
        dynamic_qubit_management_flag=_get_compilation_flag(config, "dynamic_qubit_management"),
        non_commuting_observables_flag=_get_compilation_flag(config, "non_commuting_observables"),
        options=_get_options(config),
    )
