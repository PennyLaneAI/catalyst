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


@pytest.mark.xfail(reason="OQDDeviceProperties is not yet implemented")
class TestOQDDeviceProperties:
    def test_from_toml_file(self):
        """
        Tests that the OQDDeviceProperties.from_toml_file method can
        load the device properties from a valid TOML file.
        """
        from catalyst.third_party.oqd import OQDDeviceProperties

        properties = OQDDeviceProperties.from_toml_file(OQD_SRC_DIR / "oqd_device_properties.toml")
        assert properties.parameters


class TestOQDQubitParameters:
    def test_from_toml_file_production(self):
        """
        Tests that the OQDQubitParameters.from_toml_file method can load the qubit parameters from
        the oqd_qubit_parameters.toml file used in production.
        """
        from catalyst.third_party.oqd import OQDQubitParameters

        qubit_parameters = OQDQubitParameters.from_toml_file(
            OQD_SRC_DIR / "oqd_qubit_parameters.toml"
        )
        assert qubit_parameters
        assert qubit_parameters.ion_parameters
        assert qubit_parameters.phonon_parameters
    
    def test_from_toml_file(self):
        """
        Tests that the OQDQubitParameters.from_toml_file method can load the qubit parameters from
        a sample oqd_qubit_parameters.toml file.
        """
        from catalyst.third_party.oqd import OQDQubitParameters

        qubit_parameters = OQDQubitParameters.from_toml_file("oqd_qubit_parameters.toml")
        assert qubit_parameters

        # Check ion parameters
        assert qubit_parameters.ion_parameters
        assert qubit_parameters.ion_parameters["Yb171"]

        ion_params_yb171 = qubit_parameters.ion_parameters["Yb171"]
        assert ion_params_yb171.mass == 171
        assert ion_params_yb171.charge == +1
        assert ion_params_yb171.position == [0, 0, 0]
        assert set(ion_params_yb171.levels.keys()) == {'estate', 'upstate', 'downstate'}

        # Check ion level parameters
        downstate_energy_expected = 0
        upstate_energy_expected = 2 * math.pi * 12.643e9
        estate_energy_expected = 2 * math.pi * 811.52e12
        assert math.isclose(ion_params_yb171.levels["downstate"].energy, downstate_energy_expected)
        assert math.isclose(ion_params_yb171.levels["upstate"].energy, upstate_energy_expected)
        assert math.isclose(ion_params_yb171.levels["estate"].energy, estate_energy_expected)

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
        # TODO: Complete this section


class TestOQDQubitParametersInvalid:
    def test_from_toml_file_invalid_schema(self):
        """
        Tests that the OQDQubitParameters.from_toml_file method raises an error if the TOML file
        contains an invalid schema.
        """
        from catalyst.third_party.oqd import OQDQubitParameters

        with pytest.raises(AssertionError, match="Unsupported OQD TOML config schema"):
            OQDQubitParameters.from_toml_file("oqd_qubit_parameters_invalid_schema.toml")

    def test_from_toml_file_invalid_missing_schema(self):
        """
        Tests that the OQDQubitParameters.from_toml_file method raises an error if the TOML file
        is missing the schema key.
        """
        from catalyst.third_party.oqd import OQDQubitParameters

        with pytest.raises(
            AssertionError, match="TOML document must contain key 'oqd_config_schema'"
        ):
            OQDQubitParameters.from_toml_file("oqd_qubit_parameters_invalid_missing_schema.toml")

    def test_from_toml_file_invalid_missing_ions(self):
        """
        Tests that the OQDQubitParameters.from_toml_file method raises an error if the TOML file
        is missing the 'ions' key.
        """
        from catalyst.third_party.oqd import OQDQubitParameters

        with pytest.raises(
            AssertionError, match="TOML document for OQD qubit parameters must contain key 'ions'"
        ):
            OQDQubitParameters.from_toml_file("oqd_qubit_parameters_invalid_missing_ions.toml")

    def test_from_toml_file_invalid_missing_phonons(self):
        """
        Tests that the OQDQubitParameters.from_toml_file method raises an error if the TOML file
        is missing the 'phonons' key.
        """
        from catalyst.third_party.oqd import OQDQubitParameters

        with pytest.raises(
            AssertionError,
            match="TOML document for OQD qubit parameters must contain key 'phonons'",
        ):
            OQDQubitParameters.from_toml_file("oqd_qubit_parameters_invalid_missing_phonons.toml")

    def test_from_toml_file_invalid_expr(self):
        """
        Tests that the OQDQubitParameters.from_toml_file method raises an error if the TOML file
        contains an invalid arithmetic expression.
        """
        from catalyst.third_party.oqd import OQDQubitParameters

        with pytest.raises(ValueError, match="Invalid expression:"):
            OQDQubitParameters.from_toml_file("oqd_qubit_parameters_invalid_expr.toml")


if __name__ == "__main__":
    pytest.main(["-x", __file__])
