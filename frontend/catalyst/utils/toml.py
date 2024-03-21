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
from dataclasses import dataclass
from functools import reduce
from itertools import repeat
from typing import Any, Dict, List, Set

import pennylane as qml
from pennylane.operation import Observable, Operation

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
    msg = "Either tomllib or tomlkit need to be installed."
    raise ImportError(msg)

# Give preference to tomllib
if tomllib:  # pragma: nocover
    from tomllib import load as toml_load

    TOMLDocument = Any
    TOMLException = Exception
else:  # pragma: nocover
    from tomlkit import TOMLDocument
    from tomlkit import load as toml_load
    from tomlkit.exceptions import TOMLKitError as TOMLException


def read_toml_file(toml_file: str) -> TOMLDocument:
    """Helper function opening toml file properly and reading it into a document"""
    with open(toml_file, "rb") as f:
        config = toml_load(f)
    return config


@dataclass(unsafe_hash=True)
class OperationProperties:
    invertible: bool
    controllable: bool
    differentiable: bool


def intersect_properties(a: OperationProperties, b: OperationProperties) -> OperationProperties:
    """Calculate the intersection of OperationProperties"""
    return OperationProperties(
        invertible=a.invertible and b.invertible,
        controllable=a.controllable and b.controllable,
        differentiable=a.differentiable and b.differentiable,
    )


@dataclass
class DeviceConfig:
    native_gates: Dict[Operation, OperationProperties]
    decomp: Dict[Operation, OperationProperties]
    matrix: Dict[Operation, OperationProperties]
    observables: Dict[Observable, OperationProperties]
    measurement: Dict[Operation, OperationProperties]
    mid_circuit_measurement_flag: bool
    runtime_code_generation_flag: bool
    dynamic_qubit_management_flag: bool


def intersect_operations(
    a: Dict[Operation, OperationProperties], b: Dict[Operation, OperationProperties]
) -> Dict[Operation, OperationProperties]:
    return {k: intersect_properties(a[k], b[k]) for k in (a.keys() & b.keys())}


def pennylane_operation_set(config_ops: Dict[Operation, OperationProperties]) -> Set[str]:
    ops = set()
    # Back-mapping from class names to string names
    supported_names = {v: k for k, v in map_supported_class_names().items()}
    for g, props in config_ops.items():
        ops.update({supported_names[g]})
        if props.controllable:
            ops.update({f"C({supported_names[g]})"})
    return ops


@dataclass
class ProgramFeatures:
    """Program features, obtained from the user"""

    shots_present: bool


def check_compilation_flag(config: TOMLDocument, flag_name: str) -> bool:
    return bool(config.get("compilation", {}).get(flag_name, False))


def check_adjoint_flag(config: TOMLDocument, program_features) -> bool:
    """Check the global adjoint flag for toml schema 1. For newer schemas the adjoint flag is
    defined to be set if all native gates are inverible"""
    schema = int(config["schema"])
    if schema == 1:
        return bool(config.get("compilation", {}).get("quantum_adjoint", False))

    elif schema == 2:
        return all(
            "invertible" in v.get("properties", {})
            for g, v in get_native_gates(config, program_features).items()
        )

    raise CompileError("quantum_adjoint flag is not supported in TOMLs schema >= 3")


def check_quantum_control_flag(config: TOMLDocument) -> bool:
    """Check the control flag. Only exists in toml config schema 1"""
    schema = int(config["schema"])
    if schema == 1:
        return bool(config.get("compilation", {}).get("quantum_control", False))

    raise CompileError("quantum_control flag is not supported in TOMLs schema >= 2")


def get_gates(
    config: TOMLDocument, path: List[str], program_features: ProgramFeatures
) -> Dict[str, dict]:
    """Read the toml config section specified by `path`. Filters-out gates which don't match
    condition. For now the only condition we support is `shots_present`."""
    gates = {}
    analytic = "analytic"
    finiteshots = "finiteshots"
    try:
        iterable = reduce(lambda x, y: x[y], path, config)
    except TOMLException as _:
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


def get_observables(config: TOMLDocument, program_features: ProgramFeatures) -> Dict[str, dict]:
    """Override the set of supported observables."""
    return get_gates(config, ["operators", "observables"], program_features)


def get_native_gates(config: TOMLDocument, program_features: ProgramFeatures) -> Dict[str, dict]:
    """Get the gates from the `native` section of the config."""

    schema = int(config["schema"])
    if schema == 1:
        return get_gates(config, ["operators", "gates", 0, "native"], program_features)
    elif schema == 2:
        return get_gates(config, ["operators", "gates", "native"], program_features)

    raise CompileError(f"Unsupported config schema {schema}")


def get_decomposable_gates(
    config: TOMLDocument, program_features: ProgramFeatures
) -> Dict[str, dict]:
    """Get gates that will be decomposed according to PL's decomposition rules.

    Args:
        config (TOMLDocument): Configuration dictionary
    """
    schema = int(config["schema"])
    if schema == 1:
        return get_gates(config, ["operators", "gates", 0, "decomp"], program_features)
    elif schema == 2:
        return get_gates(config, ["operators", "gates", "decomp"], program_features)

    raise CompileError(f"Unsupported config schema {schema}")


def get_matrix_decomposable_gates(
    config: TOMLDocument, program_features: ProgramFeatures
) -> Dict[str, dict]:
    """Get gates that will be decomposed to QubitUnitary.

    Args:
        config (TOMLDocument): Configuration dictionary
    """
    schema = int(config["schema"])
    if schema == 1:
        return get_gates(config, ["operators", "gates", 0, "matrix"], program_features)
    elif schema == 2:
        return get_gates(config, ["operators", "gates", "matrix"], program_features)

    raise CompileError(f"Unsupported config schema {schema}")


def get_operation_properties(config_props: dict) -> OperationProperties:
    return OperationProperties(
        invertible=config_props.get("invertible", False),
        controllable=config_props.get("controllable", False),
        differentiable=config_props.get("differentiable", False),
    )


def _build_supported_classes(qml_classes):
    acc = {}
    for cls in qml_classes:
        acc[cls.name] = cls
    return acc


def map_supported_class_names():
    from catalyst import pennylane_extensions as pe

    # fmt:off
    acc = {
        # PennyLane's gates
        'BlockEncode'                :qml.BlockEncode,
        'CNOT'                       :qml.CNOT,
        'ControlledPhaseShift'       :qml.ControlledPhaseShift,
        'ControlledQubitUnitary'     :qml.ControlledQubitUnitary,
        'CRot'                       :qml.CRot,
        'CRX'                        :qml.CRX,
        'CRY'                        :qml.CRY,
        'CRZ'                        :qml.CRZ,
        'CSWAP'                      :qml.CSWAP,
        'CY'                         :qml.CY,
        'CZ'                         :qml.CZ,
        'DoubleExcitationMinus'      :qml.DoubleExcitationMinus,
        'DoubleExcitationPlus'       :qml.DoubleExcitationPlus,
        'DoubleExcitation'           :qml.DoubleExcitation,
        'GlobalPhase'                :qml.GlobalPhase,
        'Hadamard'                   :qml.Hadamard,
        'Identity'                   :qml.Identity,
        'IsingXX'                    :qml.IsingXX,
        'IsingXY'                    :qml.IsingXY,
        'IsingYY'                    :qml.IsingYY,
        'IsingZZ'                    :qml.IsingZZ,
        'MultiRZ'                    :qml.MultiRZ,
        'PauliX'                     :qml.PauliX,
        'PauliY'                     :qml.PauliY,
        'PauliZ'                     :qml.PauliZ,
        'PhaseShift'                 :qml.PhaseShift,
        'QubitUnitary'               :qml.QubitUnitary,
        'Rot'                        :qml.Rot,
        'RX'                         :qml.RX,
        'RY'                         :qml.RY,
        'RZ'                         :qml.RZ,
        'SingleExcitationMinus'      :qml.SingleExcitationMinus,
        'SingleExcitationPlus'       :qml.SingleExcitationPlus,
        'SingleExcitation'           :qml.SingleExcitation,
        'S'                          :qml.S,
        'SWAP'                       :qml.SWAP,
        'Toffoli'                    :qml.Toffoli,
        'T'                          :qml.T,

        # Observables/measurements
        "BasisState"                 :qml.BasisState,
        "ControlledQubitUnitary"     :qml.ControlledQubitUnitary,
        "CPhase"                     :qml.CPhase,
        "DiagonalQubitUnitary"       :qml.DiagonalQubitUnitary,
        "ECR"                        :qml.ECR,
        "Exp"                        :qml.ops.Exp,
        "Hamiltonian"                :qml.Hamiltonian,
        "Hermitian"                  :qml.Hermitian,
        "ISWAP"                      :qml.ISWAP,
        "MultiControlledX"           :qml.MultiControlledX,
        "OrbitalRotation"            :qml.OrbitalRotation,
        "Prod"                       :qml.ops.Prod,
        "Projector"                  :qml.Projector,
        "PSWAP"                      :qml.PSWAP,
        "QFT"                        :qml.QFT,
        "QubitCarry"                 :qml.QubitCarry,
        "QubitStateVector"           :qml.QubitStateVector,
        "QubitSum"                   :qml.QubitSum,
        "SISWAP"                     :qml.SISWAP,
        "SparseHamiltonian"          :qml.SparseHamiltonian,
        "SProd"                      :qml.ops.SProd,
        "SQISW"                      :qml.SQISW,
        "StatePrep"                  :qml.StatePrep,
        "Sum"                        :qml.ops.Sum,
        "SX"                         :qml.SX,

        # Catalyst extension ops
        'Adjoint'                    :pe.Adjoint,
        'Cond'                       :pe.Cond,
        'ForLoop'                    :pe.ForLoop,
        'MidCircuitMeasure'          :pe.MidCircuitMeasure,
        'WhileLoop'                  :pe.WhileLoop,
    }
    # fmt:on
    return acc


def get_device_config(
    config: TOMLDocument, program_features: ProgramFeatures, device_name: str
) -> DeviceConfig:

    supported_classes = map_supported_class_names()

    native_gate_props = {}
    for g, props in get_native_gates(config, program_features).items():
        native_gate_props[supported_classes[g]] = get_operation_properties(props)

    matrix_decomp_props = {}
    for g, props in get_matrix_decomposable_gates(config, program_features).items():
        matrix_decomp_props[supported_classes[g]] = get_operation_properties(props)

    decomp_props = {}
    for g, props in get_decomposable_gates(config, program_features).items():
        decomp_props[supported_classes[g]] = get_operation_properties(props)

    observable_props = {}
    for g, props in get_observables(config, program_features).items():
        observable_props[supported_classes[g]] = get_operation_properties(props)

    measurement_props = {}
    for g, props in get_observables(config, program_features).items():
        measurement_props[supported_classes[g]] = get_operation_properties(props)

    schema = int(config["schema"])

    if schema == 1:
        supports_controlled = check_quantum_control_flag(config)

        if supports_controlled:
            for v in native_gate_props.values():
                v.controlled = True

            # TODO: remove after PR #642 is merged in lightning
            if device_name == "lightning.kokkos":  # pragma: nocover
                native_gate_props[qml.GlobalPhase] = OperationProperties(
                    invertible=True, controllable=True, differentiable=True
                )

        # TODO: remove after PR #642 is merged in lightning
        if device_name == "lightning.kokkos":  # pragma: nocover
            observable_props[qml.Projector] = OperationProperties(
                invertible=False, controllable=False, differentiable=False
            )

        # For toml schema 1 configs, the following condition is possible: (1) `QubitUnitary` gate is
        # supported, (2) native quantum control flag is enabled and (3) `ControlledQubitUnitary` is
        # listed in either matrix or decomposable sections. This is a contradiction, because
        # condition (1) means that `ControlledQubitUnitary` is also in the native set. We solve it
        # here by applying a fixup.
        # TODO: remove after PR #642 is merged in lightning
        if qml.ControlledQubitUnitary in native_gate_props:
            matrix.pop(qml.ControlledQubitUnitary)
            decomposable.pop(qml.ControlledQubitUnitary)

    return DeviceConfig(
        native_gates=native_gate_props,
        decomp=decomp_props,
        matrix=matrix_decomp_props,
        observables=observable_props,
        measurement=measurement_props,
        mid_circuit_measurement_flag=check_compilation_flag(config, "mid_circuit_measurement"),
        runtime_code_generation_flag=check_compilation_flag(config, "runtime_code_generation"),
        dynamic_qubit_management_flag=check_compilation_flag(config, "dynamic_qubit_management"),
    )
