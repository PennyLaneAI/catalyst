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
from typing import Any, Set

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
if tomllib:
    from tomllib import load as toml_load  # pragma: nocover

    TOMLDocument = Any
else:
    from tomlkit import TOMLDocument
    from tomlkit import load as toml_load  # pragma: nocover

__all__ = ["toml_load", "TOMLDocument"]


def check_mid_circuit_measurement_flag(config: TOMLDocument) -> bool:
    return bool(config["compilation"]["mid_circuit_measurement"])


def check_adjoint_flag(config: TOMLDocument) -> bool:
    return bool(config["compilation"]["quantum_adjoint"])


def check_quantum_control_flag(config: TOMLDocument) -> bool:
    schema = int(config["schema"])
    if schema == 1:
        return bool(config["compilation"]["quantum_control"])

    raise NotImplementedError("quantum_control flag is deprecated in later TOMLs")


def get_observables(config: TOMLDocument) -> Set[str]:
    """Override the set of supported observables."""
    return set(config["operators"]["observables"])


def get_decomposable_gates(config: TOMLDocument) -> Set[str]:
    """Get gates that will be decomposed according to PL's decomposition rules.

    Args:
        config (TOMLDocument): Configuration dictionary
    """
    schema = int(config["schema"])
    if schema == 1:
        return set(config["operators"]["gates"][0]["decomp"])
    elif schema == 2:
        return set(config["operators"]["gates"]["decomp"])

    raise NotImplementedError(f"Unsupported config schema {schema}")


def get_matrix_decomposable_gates(config: TOMLDocument) -> Set[str]:
    """Get gates that will be decomposed to QubitUnitary.

    Args:
        config (TOMLDocument): Configuration dictionary
    """
    schema = int(config["schema"])
    if schema == 1:
        return set(config["operators"]["gates"][0]["matrix"])
    elif schema == 2:
        return set(config["operators"]["gates"]["matrix"])

    raise NotImplementedError(f"Unsupported config schema {schema}")
