# Copyright 2024 Xanadu Quantum Technologies Inc.

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
OQD Device Properties
~~~~~~~~~~~~~~~~~~~~~

This module defines the classes that represent the properties of an Open Quantum Design (OQD)
trapped-ion quantum computer device and the methods for loading them from their respective databases
and configuration files.
"""

import math  # pylint: disable=unused-import; required since eval() assumes math module is imported
import os
import sys
from collections.abc import Collection
from dataclasses import dataclass, field
from numbers import Number
from os import PathLike
from typing import Union, Collection

if sys.version_info >= (3, 11):
    import tomllib as toml  # pragma: no cover
else:
    import tomli as toml

SUPPORTED_SCHEMAS = ["v0.1"]


@dataclass
class OQDDeviceParameter:
    """A class to represent a device parameter for an OQD trapped-ion experiment workflow.

    Attributes:
        description: A short description of the device parameter.
        stage: The stage in the trapped-ion workflow that the parameter applies to. The ordered
            list of stages in the OQD experiment workflow is ['Loading', 'Trapping',
            'Initialization', 'Experiment', 'Detection'].
        process: The process within the given stage of the trapped-ion workflow that the parameter
            applies to.
        equation: The equation that describes how the parameter is computed, if applicable.
        value: The parameter value.
        unit: The unit associated with the parameter value, if applicable.
    """

    description: str = ""
    stage: str = ""
    process: str = ""
    equation: str = ""
    value: Union[int, float] = None
    unit: str = ""


@dataclass
class OQDDeviceProperties:
    """A class to represent the properties of an OQD device.

    The properties of the device include [TODO: hardware specification, experimental
    parameters, etc.]

    Attributes:
        parameters: A dictionary of device parameters.
    """

    parameters: dict[str, OQDDeviceParameter] = field(default_factory=dict)

    @classmethod
    def from_toml_file(cls, filepath: str):
        """Loads an OQDDeviceProperties object from a TOML file.

        Args:
            filepath (str): The path to the TOML file.
        """
        document = cls._load_toml_file(filepath)
        properties = cls._parse_toml_document(document)
        return properties


@dataclass
class OQDIonLevelParameters:
    """A class to represent ... for an OQD trapped-ion experiment workflow."""

    name: str
    principal: float
    spin: float
    orbital: float
    nuclear: float
    spin_orbital: float
    spin_orbital_nuclear: float
    spin_orbital_nuclear_magnetization: float
    energy: Union[float, str]

    @classmethod
    def from_dict(cls, name: str, params: dict) -> "OQDIonLevelParameters":
        return cls(
            name=name,
            principal=params["principal"],
            spin=params["spin"],
            orbital=params["orbital"],
            nuclear=params["nuclear"],
            spin_orbital=params["spin_orbital"],
            spin_orbital_nuclear=params["spin_orbital_nuclear"],
            spin_orbital_nuclear_magnetization=params["spin_orbital_nuclear_magnetization"],
            energy=_parse_value_or_expression_as_float(params["energy"]),
        )


@dataclass
class OQDIonTransitionParameters:
    """A class to represent ... for an OQD trapped-ion experiment workflow."""

    name: str
    level1: str
    level2: str
    einsteinA: float

    @classmethod
    def from_dict(cls, name: str, params: dict) -> "OQDIonTransitionParameters":
        return cls(
            name=name,
            level1=params["level1"],
            level2=params["level2"],
            einsteinA=params["einsteinA"],
        )


@dataclass
class OQDIonParameters:
    """A class to represent ... for an OQD trapped-ion experiment workflow."""

    mass: float
    charge: int
    position: list[int]
    levels: dict[str, OQDIonLevelParameters]
    transitions: dict[str, OQDIonTransitionParameters]

    @classmethod
    def from_dict(cls, params: dict) -> "OQDIonParameters":
        return cls(
            mass=params["mass"],
            charge=params["charge"],
            position=params["position"],
            levels={
                name: OQDIonLevelParameters.from_dict(name, level)
                for name, level in params["levels"].items()
            },
            transitions={
                name: OQDIonTransitionParameters.from_dict(name, transition)
                for name, transition in params["transitions"].items()
            },
        )


@dataclass
class OQDPhononParameters:
    """A class to represent ... for an OQD trapped-ion experiment workflow."""

    energy: Union[float, str]
    eigenvector: list[int]

    @classmethod
    def from_dict(cls, params: dict) -> "OQDPhononParameters":
        return cls(
            energy=_parse_value_or_expression_as_float(params["energy"]),
            eigenvector=params["eigenvector"],
        )


@dataclass
class OQDQubitParameters:
    """A class to represent ... for an OQD trapped-ion experiment workflow."""

    ion_parameters: dict[str, OQDIonParameters]
    phonon_parameters: dict[str, OQDPhononParameters]

    @classmethod
    def from_toml(
        cls,
        filepath_or_buffer: Union[str, PathLike],
        ion_species_filter: Union[str, Collection[str]] = None,
        phonon_mode_filter: Union[str, Collection[str]] = None,
    ) -> "OQDQubitParameters":
        """Loads an OQDQubitParameters object from a TOML file or string.

        Args:
            filepath_or_buffer: The path to the TOML file or a TOML document string.
            ion_species_filter (optional): A list of ion species to include in the
                OQDQubitParameters object. If None, all ion species are included.
            phonon_mode_filter (optional): A list of phonon modes to include in the
                OQDQubitParameters object. If None, all phonon modes are included.
        """
        if os.path.isfile(filepath_or_buffer):
            document = _load_toml_from_file(filepath_or_buffer)
        else:
            document = _load_toml_from_string(filepath_or_buffer)

        _check_oqd_config_schema(document)
        cls._check_required_keys(document)

        # Collect the ion properties
        apply_ion_species_filter = ion_species_filter is not None
        if apply_ion_species_filter:
            ion_species_filter = _string_or_collection_of_strings_to_set(ion_species_filter)

        _ion_properties = {}
        for ion_species in document["ions"]:
            if apply_ion_species_filter and ion_species not in ion_species_filter:
                continue
            _ion_properties[ion_species] = OQDIonParameters.from_dict(document["ions"][ion_species])

        # Collect the phonon properties
        apply_phonon_mode_filter = phonon_mode_filter is not None
        if apply_phonon_mode_filter:
            phonon_mode_filter = _string_or_collection_of_strings_to_set(phonon_mode_filter)

        _phonon_properties = {}
        for phonon_mode in document["phonons"]:
            if apply_phonon_mode_filter and phonon_mode not in phonon_mode_filter:
                continue
            _phonon_properties[phonon_mode] = OQDPhononParameters.from_dict(
                document["phonons"][phonon_mode]
            )

        return cls(
            ion_parameters=_ion_properties,
            phonon_parameters=_phonon_properties,
        )

    @staticmethod
    def _check_required_keys(document: dict):
        assert "ions" in document, "TOML document for OQD qubit parameters must contain key 'ions'"
        assert (
            "phonons" in document
        ), "TOML document for OQD qubit parameters must contain key 'phonons'"


def _load_toml_from_file(filepath: PathLike) -> dict:
    """Loads a TOML file and returns the parsed dict."""
    with open(filepath, "rb") as f:
        return toml.load(f)


def _load_toml_from_string(contents: str) -> dict:
    """Loads a TOML string and returns the parsed dict."""
    return toml.loads(contents)


# def _parse_toml_document(document: dict):
#     """Parses a TOML document into a OQDDeviceProperties object."""

#     schema = int(document["schema"])
#     assert schema in SUPPORTED_SCHEMAS, f"Unsupported TOML config schema {schema}"

#     return OQDDeviceProperties(parameters=document["parameters"])


def _parse_value_or_expression_as_float(input: Union[Number, str]):
    """Parses a numeric value, or an expression that can be evaluated to a numeric value, and return
    as a float.

    Args:
        x (Union[Number, str]): The numeric value or expression to be evaluated.

    Returns:
        float: The original value, or the evaluated expression, as a float.

    Raises:
        ValueError: If the expression is invalid.
        TypeError: If the input is not a number or string.
    """
    if isinstance(input, Number):
        return float(input)

    elif isinstance(input, str):
        try:
            result = float(eval(input))
        except Exception as e:
            raise ValueError(f"Invalid expression: '{input}'") from e

        return result

    else:
        raise TypeError(f"Expected a number or string, but got {type(input)}")


def _check_oqd_config_schema(document: dict):
    """Checks that the TOML document has the correct schema."""

    assert "oqd_config_schema" in document, "TOML document must contain key 'oqd_config_schema'"

    schema = document["oqd_config_schema"]
    assert schema in SUPPORTED_SCHEMAS, (
        f"Unsupported OQD TOML config schema '{schema}'; "
        f"supported schemas are {SUPPORTED_SCHEMAS}"
    )


def _string_or_collection_of_strings_to_set(input: Union[str, Collection[str]]) -> set[str]:
    """Converts a string or a collection of strings to a set of strings.

    Args:
        input (Union[str, Collection[str]]): The input string or collection of strings.

    Raises:
        TypeError: If the input is not a string or a collection of strings.

    Returns:
        set[str]: The set of strings.
    """
    if isinstance(input, str):
        return {input}
    elif isinstance(input, Collection):
        assert all(
            isinstance(item, str) for item in input
        ), "All items in collection must be strings"
        return set(input)
    else:
        raise TypeError("Input must be a string or a collection of strings.")
