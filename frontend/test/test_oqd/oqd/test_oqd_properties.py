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

"""Tests for parsing and loading the configuration files for an Open Quantum Design (OQD)
trapped-ion quantum computer device.
"""

# pylint: disable=import-outside-toplevel

import math
import pathlib
import textwrap

import pytest

FRONTEND_ROOT_PATH = pathlib.Path(__file__).parent.parent.parent.parent
OQD_SRC_DIR = FRONTEND_ROOT_PATH / "catalyst/third_party/oqd/src/"
OQD_TEST_DIR = FRONTEND_ROOT_PATH / "test/test_oqd/oqd/"


class TestOQDDeviceDatabase:
    """Test suite for the OQDDeviceDatabase class."""

    def test_from_toml_production(self):
        """
        Tests that the OQDDeviceDatabase.from_toml method can load the device properties from the
        oqd_device_parameters.toml file used in production.
        """
        from catalyst.third_party.oqd import OQDDeviceDatabase

        device_database = OQDDeviceDatabase.from_toml(OQD_SRC_DIR / "oqd_device_parameters.toml")
        assert device_database.parameters

    def test_from_toml_string(self):
        """
        Tests that the OQDDeviceDatabase.from_toml method can load the device properties from a
        TOML document string.
        """
        from catalyst.third_party.oqd import OQDDeviceDatabase

        toml_document = textwrap.dedent(
            """\
            oqd_config_schema = "v0.1"

            #  Parameters
            [parameters.N_load]
            description = "Number of ions"
            stage = "Loading"
            process = "Ablation"
            equation = ""
            value = 1
            unit = ""

            [parameters.w_ablation]
            description = "Ablation laser frequency"
            stage = "Loading"
            process = "Ablation"
            equation = ""
            value = 532
            unit = "nm"
            """
        )

        properties = OQDDeviceDatabase.from_toml(toml_document)
        assert properties.parameters

        assert properties.parameters["N_load"]
        assert properties.parameters["N_load"].name == "N_load"
        assert properties.parameters["N_load"].description == "Number of ions"
        assert properties.parameters["N_load"].stage == "Loading"
        assert properties.parameters["N_load"].process == "Ablation"
        assert properties.parameters["N_load"].equation == ""
        assert properties.parameters["N_load"].value == 1
        assert properties.parameters["N_load"].unit == ""

        assert properties.parameters["w_ablation"]
        assert properties.parameters["w_ablation"].name == "w_ablation"
        assert properties.parameters["w_ablation"].description == "Ablation laser frequency"
        assert properties.parameters["w_ablation"].stage == "Loading"
        assert properties.parameters["w_ablation"].process == "Ablation"
        assert properties.parameters["w_ablation"].equation == ""
        assert properties.parameters["w_ablation"].value == 532
        assert properties.parameters["w_ablation"].unit == "nm"


class TestOQDQubitDatabase:
    """Test suite for the OQDQubitDatabase class."""

    def test_from_toml_production(self):
        """
        Tests that the OQDQubitDatabase.from_toml method can load the qubit parameters from the
        oqd_qubit_parameters.toml file used in production.
        """
        from catalyst.third_party.oqd import OQDQubitDatabase

        qubit_database = OQDQubitDatabase.from_toml(OQD_SRC_DIR / "oqd_qubit_parameters.toml")
        assert qubit_database
        assert qubit_database.ion_parameters
        assert qubit_database.phonon_parameters

    def test_from_toml(self):
        """
        Tests that the OQDQubitDatabase.from_toml method can load the qubit parameters from a
        sample oqd_qubit_parameters.toml file.
        """
        from catalyst.third_party.oqd import OQDQubitDatabase

        qubit_database = OQDQubitDatabase.from_toml(OQD_TEST_DIR / "oqd_qubit_parameters.toml")
        assert qubit_database

        # Check ion parameters
        assert qubit_database.ion_parameters
        assert qubit_database.ion_parameters["Yb171"]

        ion_params_yb171 = qubit_database.ion_parameters["Yb171"]
        assert ion_params_yb171.mass == 171
        assert ion_params_yb171.charge == +1
        assert ion_params_yb171.position == [0, 0, 0]
        assert set(ion_params_yb171.levels.keys()) == {"estate", "upstate", "downstate"}

        # Check ion level parameters
        assert math.isclose(ion_params_yb171.levels["downstate"].energy, 0.0)
        assert math.isclose(ion_params_yb171.levels["upstate"].energy, 2 * math.pi * 12.643e9)
        assert math.isclose(ion_params_yb171.levels["estate"].energy, 2 * math.pi * 811.52e12)

        assert set(ion_params_yb171.transitions.keys()) == {
            "downstate_upstate",
            "downstate_estate",
            "estate_upstate",
        }

        # Check ion transition parameters
        assert ion_params_yb171.transitions["downstate_upstate"].level1 == "downstate"
        assert ion_params_yb171.transitions["downstate_upstate"].level2 == "upstate"

        assert ion_params_yb171.transitions["downstate_estate"].level1 == "downstate"
        assert ion_params_yb171.transitions["downstate_estate"].level2 == "estate"

        assert ion_params_yb171.transitions["estate_upstate"].level1 == "estate"
        assert ion_params_yb171.transitions["estate_upstate"].level2 == "upstate"

        # Check phonon parameters
        assert qubit_database.phonon_parameters
        assert math.isclose(qubit_database.phonon_parameters["COM_x"].energy, 2 * math.pi * 5e6)
        assert math.isclose(qubit_database.phonon_parameters["COM_y"].energy, 2 * math.pi * 5e6)
        assert math.isclose(qubit_database.phonon_parameters["COM_z"].energy, 2 * math.pi * 1e6)

        assert qubit_database.phonon_parameters["COM_x"].eigenvector == [1, 0, 0]
        assert qubit_database.phonon_parameters["COM_y"].eigenvector == [0, 1, 0]
        assert qubit_database.phonon_parameters["COM_z"].eigenvector == [0, 0, 1]


class TestOQDQubitDatabaseInvalid:
    """Test suite for the OQDQubitDatabase class given invalid TOML input."""

    def test_from_toml_invalid_missing_schema(self):
        """
        Tests that the OQDQubitDatabase.from_toml method raises an error if the TOML file is
        missing the schema key.
        """
        from catalyst.third_party.oqd import OQDQubitDatabase

        toml_document = textwrap.dedent(
            """\
            # Empty TOML document
            """
        )

        with pytest.raises(
            AssertionError, match="TOML document must contain key 'oqd_config_schema'"
        ):
            OQDQubitDatabase.from_toml(toml_document)

    def test_from_toml_invalid_schema(self):
        """
        Tests that the OQDQubitDatabase.from_toml method raises an error if the TOML file contains
        an invalid schema.
        """
        from catalyst.third_party.oqd import OQDQubitDatabase

        toml_document = textwrap.dedent(
            """\
            oqd_config_schema = "v0.0"  # This is an invalid schema
            """
        )

        with pytest.raises(AssertionError, match="Unsupported OQD TOML config schema"):
            OQDQubitDatabase.from_toml(toml_document)

    def test_from_toml_invalid_missing_ions(self):
        """
        Tests that the OQDQubitDatabase.from_toml method raises an error if the TOML file is
        missing the 'ions' key.
        """
        from catalyst.third_party.oqd import OQDQubitDatabase

        toml_document = textwrap.dedent(
            """\
            oqd_config_schema = "v0.1"

            # --- Ions --- #
            # This section intentionally left blank

            # --- Phonons --- #
            [phonons.COM_x]
            energy = "2 * math.pi * 5e6"
            eigenvector = [1, 0, 0]
            """
        )

        with pytest.raises(
            AssertionError, match="TOML document for OQD qubit parameters must contain key 'ions'"
        ):
            OQDQubitDatabase.from_toml(toml_document)

    def test_from_toml_invalid_missing_phonons(self):
        """
        Tests that the OQDQubitDatabase.from_toml method raises an error if the TOML file is
        missing the 'phonons' key.
        """
        from catalyst.third_party.oqd import OQDQubitDatabase

        toml_document = textwrap.dedent(
            """\
            oqd_config_schema = "v0.1"

            # --- Ions --- #
            [ions.Yb171]
            mass = 171
            charge = +1
            position = [0, 0, 0]

            # --- Phonons --- #
            # This section intentionally left blank
            """
        )

        with pytest.raises(
            AssertionError,
            match="TOML document for OQD qubit parameters must contain key 'phonons'",
        ):
            OQDQubitDatabase.from_toml(toml_document)

    def test_from_toml_invalid_expr(self):
        """
        Tests that the OQDQubitDatabase.from_toml method raises an error if the TOML file contains
        an invalid arithmetic expression.
        """
        from catalyst.third_party.oqd import OQDQubitDatabase

        toml_document = textwrap.dedent(
            """\
            oqd_config_schema = "v0.1"

            # --- Ions --- #
            [ions.Yb171]
            mass = 171
            charge = +1
            position = [0, 0, 0]

            levels.downstate.principal = 6
            levels.downstate.spin = 0.5
            levels.downstate.orbital = 0
            levels.downstate.nuclear = 0.5
            levels.downstate.spin_orbital = 0.5
            levels.downstate.spin_orbital_nuclear = 0
            levels.downstate.spin_orbital_nuclear_magnetization = 0
            levels.downstate.energy = 0

            levels.upstate.principal = 6
            levels.upstate.spin = 0.5
            levels.upstate.orbital = 0
            levels.upstate.nuclear = 0.5
            levels.upstate.spin_orbital = 0.5
            levels.upstate.spin_orbital_nuclear = 1
            levels.upstate.spin_orbital_nuclear_magnetization = 0
            levels.upstate.energy = "1 + math.sin('a') + math.cos"  # Invalid arithmetic expression

            levels.estate.principal = 5
            levels.estate.spin = 0.5
            levels.estate.orbital = 1
            levels.estate.nuclear = 0.5
            levels.estate.spin_orbital = 0.5
            levels.estate.spin_orbital_nuclear = 0
            levels.estate.spin_orbital_nuclear_magnetization = 0
            levels.estate.energy = "1 + math.sin('a') + math.cos"  # Invalid arithmetic expression

            # --- Phonons --- #
            [phonons.COM_x]
            energy = "2 * math.pi * 5e6"
            eigenvector = [1, 0, 0]
            """
        )

        with pytest.raises(ValueError, match="Invalid expression:"):
            OQDQubitDatabase.from_toml(toml_document)


class TestOQDBeamDatabase:
    """Tests for the OQDBeamDatabase class."""

    def test_from_toml_production(self):
        """
        Tests that the OQDBeamDatabase.from_toml method can load the beam parameters from the
        oqd_beam_parameters.toml file used in production.
        """
        from catalyst.third_party.oqd import OQDBeamDatabase

        beam_database = OQDBeamDatabase.from_toml(OQD_SRC_DIR / "oqd_beam_parameters.toml")
        assert beam_database.beam_parameters

    def test_from_toml(self):
        """
        Tests that the OQDBeamDatabase.from_toml method can load the beam parameters from a
        sample oqd_beam_parameters.toml file.
        """
        from catalyst.third_party.oqd import OQDBeamDatabase

        beam_database = OQDBeamDatabase.from_toml(OQD_TEST_DIR / "oqd_beam_parameters.toml")
        assert beam_database.beam_parameters

        assert beam_database.beam_parameters["downstate_upstate"]
        assert beam_database.beam_parameters["downstate_upstate"].transition == "downstate_upstate"
        assert math.isclose(
            beam_database.beam_parameters["downstate_upstate"].rabi, 2 * math.pi * 1e6
        )
        assert beam_database.beam_parameters["downstate_upstate"].detuning == 0
        assert beam_database.beam_parameters["downstate_upstate"].phase == 0
        assert math.isnan(beam_database.beam_parameters["downstate_upstate"].polarization)
        assert math.isnan(beam_database.beam_parameters["downstate_upstate"].wavevector)

        assert beam_database.beam_parameters["downstate_estate"]
        assert beam_database.beam_parameters["downstate_estate"].transition == "downstate_estate"
        assert math.isclose(
            beam_database.beam_parameters["downstate_estate"].rabi, 2 * math.pi * 1e6
        )
        assert beam_database.beam_parameters["downstate_estate"].detuning == 0
        assert beam_database.beam_parameters["downstate_estate"].phase == 0
        assert math.isnan(beam_database.beam_parameters["downstate_estate"].polarization)
        assert math.isnan(beam_database.beam_parameters["downstate_estate"].wavevector)

    def test_from_toml_invalid_missing_schema(self):
        """
        Tests that the OQDQubitDatabase.from_toml method raises an error if the TOML file is
        missing the schema key.
        """
        from catalyst.third_party.oqd import OQDBeamDatabase

        toml_document = textwrap.dedent(
            """\
            # Empty TOML document
            """
        )

        with pytest.raises(
            AssertionError, match="TOML document must contain key 'oqd_config_schema'"
        ):
            OQDBeamDatabase.from_toml(toml_document)

    def test_from_toml_invalid_schema(self):
        """
        Tests that the OQDBeamDatabase.from_toml method raises an error if the TOML file contains
        an invalid schema.
        """
        from catalyst.third_party.oqd import OQDBeamDatabase

        toml_document = textwrap.dedent(
            """\
            oqd_config_schema = "v0.0"  # This is an invalid schema
            """
        )

        with pytest.raises(AssertionError, match="Unsupported OQD TOML config schema"):
            OQDBeamDatabase.from_toml(toml_document)

    def test_from_toml_invalid_beams(self):
        """
        Tests that the OQDBeamDatabase.from_toml method raises an error if the TOML file is
        missing the 'beams' key.
        """
        from catalyst.third_party.oqd import OQDBeamDatabase

        toml_document = textwrap.dedent(
            """\
            oqd_config_schema = "v0.1"

            # --- Beams --- #
            # This section intentionally left blank
            """
        )

        with pytest.raises(
            AssertionError, match="TOML document for OQD beam parameters must contain key 'beams'"
        ):
            OQDBeamDatabase.from_toml(toml_document)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
