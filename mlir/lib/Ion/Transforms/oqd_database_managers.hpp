// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <iostream>

#include <cassert>
#include <regex>

#include <toml++/toml.hpp>

#include "oqd_database_types.hpp"
#include "oqd_database_utils.hpp"

namespace {

// TODO: we should decide where the calibration tomls will live.
// Very likely, they will be dumped into some local locations that are not part of the catalyst
// tree. It would be best to store the toml locations as pass options on the --quantum-to-ion pass
// This way, they can be directly specified in the frontend python layer, and passed down
// to the pass as pass options (via the standard transform dialect lowering pipeline we have)
// But for now this hard coded prototype is good enough.

static const std::string catalyst_root_path =
    std::regex_replace(__FILE__, std::regex("mlir/(.)*/oqd_database_managers.hpp"), "");

static const std::string oqd_device_parameters_toml_file_path =
    catalyst_root_path + "frontend/catalyst/third_party/oqd/src/oqd_device_parameters.toml";
static const std::string oqd_qubit_parameters_toml_file_path =
    catalyst_root_path + "frontend/catalyst/third_party/oqd/src/oqd_qubit_parameters.toml";
static const std::string oqd_gate_decomposition_parameters_toml_file_path =
    catalyst_root_path +
    "frontend/catalyst/third_party/oqd/src/oqd_gate_decomposition_parameters.toml";

class OQDDatabaseManager {

  public:
    OQDDatabaseManager()
    {
        sourceTomlDevice = toml::parse_file(oqd_device_parameters_toml_file_path);
        sourceTomlQubit = toml::parse_file(oqd_qubit_parameters_toml_file_path);
        sourceTomlGateDecomposition =
            toml::parse_file(oqd_gate_decomposition_parameters_toml_file_path);

        assert(sourceTomlDevice && "Parsing of device toml failed!");
        assert(sourceTomlQubit && "Parsing of qubit toml failed!");
        assert(sourceTomlGateDecomposition && "Parsing of gate decomposition toml failed!");

        loadBeams1Params();
        loadBeams2Params();

        loadPhononParams();
    }

    const std::vector<Beam> &getBeams1Params() const { return beams1; }
    const std::vector<Beam> &getBeams2Params() const { return beams2; }

    const std::vector<PhononMode> &getPhononParams() const { return phonons; }

  private:
    toml::parse_result sourceTomlDevice;
    toml::parse_result sourceTomlQubit;
    toml::parse_result sourceTomlGateDecomposition;

    std::vector<Beam> beams1;
    std::vector<Beam> beams2;

    std::vector<PhononMode> phonons;

    void loadBeams1Params() { loadBeamsParamsImpl("beams1"); }
    void loadBeams2Params() { loadBeamsParamsImpl("beams2"); }

    void loadBeamsParamsImpl(const std::string &mode)
    {
        // Read in the gate decomposition beam parameters from toml file.
        // The toml contains a list of beams, where each beam has the following fields:
        //   rabi = 4.4
        //   detuning = 5.5
        //   polarization = [6,7]
        //   wavevector = [8,9]

        toml::node_view<toml::node> beamsToml = sourceTomlGateDecomposition[mode];
        size_t numBeams = beamsToml.as_array()->size();

        std::vector<Beam> *collector;
        if (mode == "beams1") {
            collector = &beams1;
        }
        else if (mode == "beams2") {
            collector = &beams2;
        }
        else {
            assert(false && "Invalid beam mode. Only single-qubit gates and 2-qubit gates are "
                            "supported for decomposition onto beams.");
        }

        for (size_t i = 0; i < numBeams; i++) {
            auto beam = beamsToml[i];
            double rabi = beam["rabi"].as_floating_point()->get();
            double detuning = beam["detuning"].as_floating_point()->get();
            std::vector<int64_t> polarization =
                tomlArray2StdVector<int64_t>(*(beam["polarization"].as_array()));
            std::vector<int64_t> wavevector =
                tomlArray2StdVector<int64_t>(*(beam["wavevector"].as_array()));

            collector->push_back(Beam(rabi, detuning, polarization, wavevector));
        }
    }

    void loadPhononParams()
    {
        toml::node_view<toml::node> phononsToml = sourceTomlGateDecomposition["phonons"];
        size_t numPhononModes = phononsToml.as_array()->size();

        auto parseSingleDirection = [](auto direction) {
            double energy = direction["energy"].as_floating_point()->get();
            std::vector<int64_t> eigenvector =
                tomlArray2StdVector<int64_t>(*(direction["eigenvector"].as_array()));
            return Phonon(energy, eigenvector);
        };

        for (size_t i = 0; i < numPhononModes; i++) {
            auto phononMode = phononsToml[i];

            Phonon COM_x = parseSingleDirection(phononMode["COM_x"]);
            Phonon COM_y = parseSingleDirection(phononMode["COM_y"]);
            Phonon COM_z = parseSingleDirection(phononMode["COM_z"]);

            phonons.push_back(PhononMode(COM_x, COM_y, COM_z));
        }
    }
};

} // namespace
