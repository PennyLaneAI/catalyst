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

from collections.abc import Collection
from dataclasses import dataclass, field
from numbers import Number
from os import PathLike
from typing import Union, Collection

from catalyst.utils.toml_utils import load_toml, safe_eval


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

    # TODO: Some parameters, mainly laser frequencies, are expressed as the sum of a nominal
    # wavelength and a frequency offset, e.g. '493 nm - 7.506 GHz', which is equal to 493.0061 nm.
    # We need to be able to support this.
    # Furthermore, some parameters may either have one value or have two simultaneous values, e.g.
    # 'Doppler cooling laser frequency', which has values '493 nm - 7.506 GHz' and
    # '493 nm + 4.259 GHz'. We need to be able to support this as well.

    name: str
    description: str = ""
    stage: str = ""
    process: str = ""
    equation: str = ""
    value: Union[int, float] = None
    unit: str = ""

    @classmethod
    def from_dict(cls, name: str, params: dict) -> "OQDDeviceParameter":
        """Creates an OQDDeviceParameter object from a dictionary.

        Args:
            name: The name of the device parameter.
            params: A dictionary containing the device parameter parameters, typically as parsed
                from a TOML document.

        Returns:
            OQDDeviceParameter: The OQDDeviceParameter object.
        """
        return cls(
            name=name,
            description=params["description"],
            stage=params["stage"],
            process=params["process"],
            equation=params["equation"],
            value=params["value"],
            unit=params["unit"],
        )


@dataclass
class OQDDeviceDatabase:
    """A database class to represent the properties of an OQD device.

    The properties of the device include hardware specification, parameters relating to the
    experimental apparatus, and generally any other parameters needed by the compiler. Physical
    constants, such as the energy levels of the ion(s) being uses, are handled separately.

    Attributes:
        parameters: A dictionary of OQD device parameters.
    """

    parameters: dict[str, OQDDeviceParameter] = field(default_factory=dict)

    @classmethod
    def from_toml(cls, filepath_or_buffer: Union[str, PathLike]):
        """Loads an OQDDeviceProperties object from a TOML file or string.

        Args:
            filepath_or_buffer: The path to the TOML file or a TOML document string.
        """
        try:
            document = load_toml(filepath_or_buffer)

        except Exception as e:
            raise ValueError(
                "Failed to load TOML document when creating OQDDeviceProperties"
            ) from e

        properties = cls._parse_toml_document(document)
        return properties

    @classmethod
    def _parse_toml_document(cls, document: dict):
        """Parses a TOML document and returns an OQDDeviceProperties object."""
        _check_oqd_config_schema(document)
        cls._check_required_keys(document)
        return cls(
            parameters={
                name: OQDDeviceParameter.from_dict(name, level)
                for name, level in document["parameters"].items()
            }
        )

    @staticmethod
    def _check_required_keys(document: dict):
        assert (
            "parameters" in document
        ), "TOML document for OQD device parameters must contain key 'parameters'"


@dataclass
class OQDIonLevelParameters:
    """A class to represent ion energy levels for an OQD trapped-ion experiment workflow."""

    # pylint: disable=too-many-instance-attributes
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
        """Creates an OQDIonLevelParameters object from a dictionary.

        Args:
            name: The name of the level, e.g. 'downstate', 'upstate', 'estate', etc.
            params: A dictionary containing the level parameters, including the relevant quantum
                numbers and the level energy, typically as parsed from a TOML document.

        Returns:
            OQDIonLevelParameters: The OQDIonLevelParameters object.
        """
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
    """A class to represent a specific transition between ion energy levels for an OQD trapped-ion
    experiment workflow."""

    name: str
    level1: str
    level2: str
    einsteinA: float

    @classmethod
    def from_dict(cls, name: str, params: dict) -> "OQDIonTransitionParameters":
        """Creates an OQDIonTransitionParameters object from a dictionary.

        Args:
            name: The name of the transition as '<level1>_<level2>', e.g. 'downstate_upstate',
                'upstate_downstate', etc.
            params: A dictionary containing the transition parameters, typically as parsed from a
                TOML document.

        Returns:
            OQDIonTransitionParameters: The OQDIonTransitionParameters object.
        """
        return cls(
            name=name,
            level1=params["level1"],
            level2=params["level2"],
            einsteinA=params["einsteinA"],
        )


@dataclass
class OQDIonParameters:
    """A class to represent an ion used in an OQD trapped-ion experiment workflow."""

    mass: float
    charge: int
    position: list[int]
    levels: dict[str, OQDIonLevelParameters]
    transitions: dict[str, OQDIonTransitionParameters]

    @classmethod
    def from_dict(cls, params: dict) -> "OQDIonParameters":
        """Creates an OQDIonParameters object from a dictionary.

        Args:
            params: A dictionary containing the ion parameters, typically as parsed from a TOML
                document.

        Returns:
            OQDIonParameters: The OQDIonParameters object.
        """
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
    """A class to represent a phonon mode for an OQD trapped-ion experiment workflow."""

    energy: Union[float, str]
    eigenvector: list[int]

    @classmethod
    def from_dict(cls, params: dict) -> "OQDPhononParameters":
        """Creates an OQDPhononParameters object from a dictionary.

        Args:
            params: A dictionary containing the phonon mode parameters, typically as parsed from a
                TOML document.

        Returns:
            OQDPhononParameters: The OQDPhononParameters object.
        """
        return cls(
            energy=_parse_value_or_expression_as_float(params["energy"]),
            eigenvector=params["eigenvector"],
        )


@dataclass
class OQDQubitDatabase:
    """A database class to represent the qubit parameters for an OQD trapped-ion experiment workflow."""  # pylint: disable=line-too-long

    ion_parameters: dict[str, OQDIonParameters]
    phonon_parameters: dict[str, OQDPhononParameters]

    @classmethod
    def from_toml(
        cls,
        filepath_or_buffer: Union[str, PathLike],
        ion_species_filter: Union[str, Collection[str]] = None,
        phonon_mode_filter: Union[str, Collection[str]] = None,
    ) -> "OQDQubitDatabase":
        """Loads an OQDQubitDatabase object from a TOML file or string.

        Args:
            filepath_or_buffer: The path to the TOML file or a TOML document string.
            ion_species_filter (optional): A list of ion species to include in the
                OQDQubitDatabase object. If None, all ion species are included.
            phonon_mode_filter (optional): A list of phonon modes to include in the
                OQDQubitDatabase object. If None, all phonon modes are included.
        """
        try:
            document = load_toml(filepath_or_buffer)

        except Exception as e:
            raise ValueError("Failed to load TOML document when creating OQDQubitDatabase") from e

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


@dataclass
class OQDBeamParameters:
    """A class to represent the beam parameters for an OQD trapped-ion experiment workflow."""

    transition: str
    rabi: Union[float, str]
    detuning: float
    phase: float
    polarization: float
    wavevector: float

    @classmethod
    def from_dict(cls, params: dict) -> "OQDBeamParameters":
        """Creates an OQDBeamParameters object from a dictionary.

        Args:
            params: A dictionary containing the beam parameters, typically as parsed from a TOML
                document.

        Returns:
            OQDBeamParameters: The OQDBeamParameters object.
        """
        return cls(
            transition=params["transition"],
            rabi=_parse_value_or_expression_as_float(params["rabi"]),
            detuning=_parse_value_or_expression_as_float(params["detuning"]),
            phase=_parse_value_or_expression_as_float(params["phase"]),
            polarization=_parse_value_or_expression_as_float(params["polarization"]),
            wavevector=params["wavevector"],
        )


@dataclass
class OQDBeamDatabase:
    """A database class to represent the beam parameters for an OQD trapped-ion experiment workflow."""  # pylint: disable=line-too-long

    beam_parameters: dict[str, OQDBeamParameters]

    @classmethod
    def from_toml(cls, filepath_or_buffer: Union[str, PathLike]):
        """Loads an OQDBeamDatabase object from a TOML file or string.

        Args:
            filepath_or_buffer: The path to the TOML file or a TOML document string.
        """
        try:
            document = load_toml(filepath_or_buffer)

        except Exception as e:
            raise ValueError("Failed to load TOML document when creating OQDBeamDatabase") from e

        _check_oqd_config_schema(document)
        cls._check_required_keys(document)

        return cls(
            beam_parameters={
                beam: OQDBeamParameters.from_dict(document["beams"][beam])
                for beam in document["beams"]
            }
        )

    @staticmethod
    def _check_required_keys(document: dict):
        assert "beams" in document, "TOML document for OQD beam parameters must contain key 'beams'"


def _parse_value_or_expression_as_float(input_: Union[Number, str]):
    """Parses a numeric value, or an expression that can be evaluated to a numeric value, and return
    as a float.

    Args:
        input_: The numeric value or expression to be evaluated.

    Returns:
        float: The original value, or the evaluated expression, as a float.

    Raises:
        ValueError: If the expression is invalid.
        TypeError: If the input is not a number or string.
    """
    if isinstance(input_, Number):
        return float(input_)

    elif isinstance(input_, str):
        try:
            result = float(safe_eval(input_))
        except Exception as e:
            raise ValueError(f"Invalid expression: '{input_}'") from e

        return result

    else:
        raise TypeError(f"Expected a number or string, but got {type(input_)}")


def _check_oqd_config_schema(document: dict):
    """Checks that the TOML document has the correct schema."""

    assert "oqd_config_schema" in document, "TOML document must contain key 'oqd_config_schema'"

    schema = document["oqd_config_schema"]
    assert schema in SUPPORTED_SCHEMAS, (
        f"Unsupported OQD TOML config schema '{schema}'; "
        f"supported schemas are {SUPPORTED_SCHEMAS}"
    )


def _string_or_collection_of_strings_to_set(input_: Union[str, Collection[str]]) -> set[str]:
    """Converts a string or a collection of strings to a set of strings.

    Args:
        input (Union[str, Collection[str]]): The input string or collection of strings.

    Raises:
        TypeError: If the input is not a string or a collection of strings.

    Returns:
        set[str]: The set of strings.
    """
    if isinstance(input_, str):
        return {input_}

    elif isinstance(input_, Collection):
        assert all(
            isinstance(item, str) for item in input_
        ), "All items in collection must be strings"
        return set(input_)

    else:
        raise TypeError(f"Expected a string or a collection of strings, but got {type(input_)}")
