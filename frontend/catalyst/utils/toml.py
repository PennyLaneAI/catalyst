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
from functools import reduce
from itertools import repeat
from typing import Any, Dict, List

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
else:  # pragma: nocover
    from tomlkit import TOMLDocument
    from tomlkit import load as toml_load


def read_toml_file(toml_file: str) -> TOMLDocument:
    """Helper function opening toml file properly and reading it into a document"""
    with open(toml_file, "rb") as f:
        config = toml_load(f)
    return config


def check_mid_circuit_measurement_flag(config: TOMLDocument) -> bool:
    """Check the global mid-circuit measurement flag"""
    return bool(config.get("compilation", {}).get("mid_circuit_measurement", False))


def check_adjoint_flag(config: TOMLDocument, shots_present: bool) -> bool:
    """Check the global adjoint flag for toml schema 1. For newer schemas the adjoint flag is
    defined to be set if all native gates are inverible"""
    schema = int(config["schema"])
    if schema == 1:
        return bool(config.get("compilation", {}).get("quantum_adjoint", False))

    elif schema == 2:
        return all(
            "invertible" in v.get("properties", {})
            for g, v in get_native_gates(config, shots_present).items()
        )

    raise CompileError("quantum_adjoint flag is not supported in TOMLs schema >= 3")


def check_quantum_control_flag(config: TOMLDocument) -> bool:
    """Check the control flag. Only exists in toml config schema 1"""
    schema = int(config["schema"])
    if schema == 1:
        return bool(config.get("compilation", {}).get("quantum_control", False))

    raise CompileError("quantum_control flag is not supported in TOMLs schema >= 2")


def get_gates(config: TOMLDocument, path: List[str], shots_present: bool) -> Dict[str, dict]:
    """Read the toml config section specified by `path`. Filters-out gates which don't match
    condition. For now the only condition we support is `shots_present`."""
    gates = {}
    analytic = "analytic"
    finiteshots = "finiteshots"
    iterable = reduce(lambda x, y: x[y], path, config)
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
            if analytic in conditions and not shots_present:
                gates[g] = values
            elif finiteshots in conditions and shots_present:
                gates[g] = values
        else:
            gates[g] = values
    return gates


def get_observables(config: TOMLDocument, shots_present: bool) -> Dict[str, dict]:
    """Override the set of supported observables."""
    return get_gates(config, ["operators", "observables"], shots_present)


def get_measurement_processes(config: TOMLDocument, shots_present: bool) -> Dict[str, dict]:
    """Get the measurements processes from the `native` section of the config."""

    schema = int(config["schema"])
    if schema == 1:
        shots_string = "finiteshots" if shots_present else "exactshots"
        return get_gates(config, ["measurement_processes", shots_string], shots_present)
    if schema == 2:
        return get_gates(config, ["measurement_processes"], shots_present)

    raise CompileError(f"Unsupported config schema {schema}")


def get_native_gates(config: TOMLDocument, shots_present: bool) -> Dict[str, dict]:
    """Get the gates from the `native` section of the config."""

    schema = int(config["schema"])
    if schema == 1:
        return get_gates(config, ["operators", "gates", 0, "native"], shots_present)
    elif schema == 2:
        return get_gates(config, ["operators", "gates", "native"], shots_present)

    raise CompileError(f"Unsupported config schema {schema}")


def get_decomposable_gates(config: TOMLDocument, shots_present: bool) -> Dict[str, dict]:
    """Get gates that will be decomposed according to PL's decomposition rules.

    Args:
        config (TOMLDocument): Configuration dictionary
    """
    schema = int(config["schema"])
    if schema == 1:
        return get_gates(config, ["operators", "gates", 0, "decomp"], shots_present)
    elif schema == 2:
        return get_gates(config, ["operators", "gates", "decomp"], shots_present)

    raise CompileError(f"Unsupported config schema {schema}")


def get_matrix_decomposable_gates(config: TOMLDocument, shots_present: bool) -> Dict[str, dict]:
    """Get gates that will be decomposed to QubitUnitary.

    Args:
        config (TOMLDocument): Configuration dictionary
    """
    schema = int(config["schema"])
    if schema == 1:
        return get_gates(config, ["operators", "gates", 0, "matrix"], shots_present)
    elif schema == 2:
        return get_gates(config, ["operators", "gates", "matrix"], shots_present)

    raise CompileError(f"Unsupported config schema {schema}")
