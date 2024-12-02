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

# pylint: disable=import-outside-toplevel

import math
import pathlib

import pytest

FRONTEND_ROOT_PATH = pathlib.Path(__file__).parent.parent.parent.parent
OQD_SRC_DIR = FRONTEND_ROOT_PATH / "catalyst/third_party/oqd/src/"
OQD_TEST_DIR = FRONTEND_ROOT_PATH / "test/test_oqd/oqd/"


@pytest.mark.xfail(reason="OQDDeviceProperties is not yet implemented")
class TestOQDDeviceProperties:
    def test_from_toml_file(self):
        """
        Tests that the OQDDeviceProperties.from_toml_file method can load the device properties from
        a valid TOML file.
        """
        from catalyst.third_party.oqd import OQDDeviceProperties

        properties = OQDDeviceProperties.from_toml_file(OQD_SRC_DIR / "oqd_device_properties.toml")
        assert properties.parameters


class TestOQDQubitParameters:
    def test_from_toml_production(self):
        """
        Tests that the OQDQubitParameters.from_toml method can load the qubit parameters from the
        oqd_qubit_parameters.toml file used in production.
        """
        from catalyst.third_party.oqd import OQDQubitParameters

        qubit_parameters = OQDQubitParameters.from_toml(OQD_SRC_DIR / "oqd_qubit_parameters.toml")
        assert qubit_parameters
        assert qubit_parameters.ion_parameters
        assert qubit_parameters.phonon_parameters

    def test_from_toml(self):
        """
        Tests that the OQDQubitParameters.from_toml method can load the qubit parameters from a
        sample oqd_qubit_parameters.toml file.
        """
        from catalyst.third_party.oqd import OQDQubitParameters

        qubit_parameters = OQDQubitParameters.from_toml(OQD_TEST_DIR / "oqd_qubit_parameters.toml")
        assert qubit_parameters

        # Check ion parameters
        assert qubit_parameters.ion_parameters
        assert qubit_parameters.ion_parameters["Yb171"]

        ion_params_yb171 = qubit_parameters.ion_parameters["Yb171"]
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
        assert qubit_parameters.phonon_parameters
        assert math.isclose(qubit_parameters.phonon_parameters["COM_x"].energy, 2 * math.pi * 5e6)
        assert math.isclose(qubit_parameters.phonon_parameters["COM_y"].energy, 2 * math.pi * 5e6)
        assert math.isclose(qubit_parameters.phonon_parameters["COM_z"].energy, 2 * math.pi * 1e6)

        assert qubit_parameters.phonon_parameters["COM_x"].eigenvector == [1, 0, 0]
        assert qubit_parameters.phonon_parameters["COM_y"].eigenvector == [0, 1, 0]
        assert qubit_parameters.phonon_parameters["COM_z"].eigenvector == [0, 0, 1]


class TestOQDQubitParametersInvalid:
    def test_from_toml_invalid_missing_schema(self):
        """
        Tests that the OQDQubitParameters.from_toml method raises an error if the TOML file is
        missing the schema key.
        """
        from catalyst.third_party.oqd import OQDQubitParameters

        toml_document = """
        # Empty TOML document
        """

        with pytest.raises(
            AssertionError, match="TOML document must contain key 'oqd_config_schema'"
        ):
            OQDQubitParameters.from_toml(toml_document)

    def test_from_toml_invalid_schema(self):
        """
        Tests that the OQDQubitParameters.from_toml method raises an error if the TOML file contains
        an invalid schema.
        """
        from catalyst.third_party.oqd import OQDQubitParameters

        toml_document = """
        oqd_config_schema = "v0.0"  # This is an invalid schema
        """

        with pytest.raises(AssertionError, match="Unsupported OQD TOML config schema"):
            OQDQubitParameters.from_toml(toml_document)

    def test_from_toml_invalid_missing_ions(self):
        """
        Tests that the OQDQubitParameters.from_toml method raises an error if the TOML file is
        missing the 'ions' key.
        """
        from catalyst.third_party.oqd import OQDQubitParameters

        toml_document = """
        oqd_config_schema = "v0.1"

        # --- Ions --- #
        # This section intentionally left blank

        # --- Phonons --- #
        [phonons.COM_x]
        energy = "2 * math.pi * 5e6"
        eigenvector = [1, 0, 0]
        """

        with pytest.raises(
            AssertionError, match="TOML document for OQD qubit parameters must contain key 'ions'"
        ):
            OQDQubitParameters.from_toml(toml_document)

    def test_from_toml_invalid_missing_phonons(self):
        """
        Tests that the OQDQubitParameters.from_toml method raises an error if the TOML file is
        missing the 'phonons' key.
        """
        from catalyst.third_party.oqd import OQDQubitParameters

        toml_document = """
        oqd_config_schema = "v0.1"

        # --- Ions --- #
        [ions.Yb171]
        mass = 171
        charge = +1
        position = [0, 0, 0]

        # --- Phonons --- #
        # This section intentionally left blank
        """

        with pytest.raises(
            AssertionError,
            match="TOML document for OQD qubit parameters must contain key 'phonons'",
        ):
            OQDQubitParameters.from_toml(toml_document)

    def test_from_toml_invalid_expr(self):
        """
        Tests that the OQDQubitParameters.from_toml method raises an error if the TOML file contains
        an invalid arithmetic expression.
        """
        from catalyst.third_party.oqd import OQDQubitParameters

        toml_document = """
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

        with pytest.raises(ValueError, match="Invalid expression:"):
            OQDQubitParameters.from_toml(toml_document)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
