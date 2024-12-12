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

#include <cassert>
#include <regex>
#include <type_traits>

#include <toml++/toml.hpp>

namespace {

static const std::string catalyst_root_path =
    std::regex_replace(__FILE__, std::regex("mlir/(.)*/oqd_database_managers.hpp"), "");

static const std::string oqd_device_parameters_toml_file_path =
    catalyst_root_path + "frontend/catalyst/third_party/oqd/src/oqd_device_parameters.toml";
static const std::string oqd_qubit_parameters_toml_file_path =
    catalyst_root_path + "frontend/catalyst/third_party/oqd/src/oqd_qubit_parameters.toml";
static const std::string oqd_gate_decomposition_parameters_toml_file_path =
    catalyst_root_path +
    "frontend/catalyst/third_party/oqd/src/oqd_gate_decomposition_parameters.toml";

template <typename T> std::vector<T> tomlArray2StdVector(const toml::array &arr)
{
    // A toml node can contain toml objects of arbitrary types, even other toml nodes
    // i.e. toml nodes are similar to pytrees
    // Therefore, toml++ does not provide a simple "toml array to std vector" converter
    //
    // For a "leaf" array node, whose contents are now simple values,
    // such a utility would come in handy.

    std::vector<T> vec;

    if constexpr (std::is_same_v<T, int64_t>) {
        for (const auto &elem : arr) {
            vec.push_back(elem.as_integer()->get());
        }
    }
    else if constexpr (std::is_same_v<T, double>) {
        for (const auto &elem : arr) {
            vec.push_back(elem.as_floating_point()->get());
        }
    }

    return vec;
}

struct Beam {
    // This struct contains the calibrated beam parameters.
    double rabi, detuning;
    std::vector<int64_t> polarization, wavevector;

    Beam(double _rabi, double _detuning, std::vector<int64_t> _polarization,
         std::vector<int64_t> _wavevector)
        : rabi(_rabi), detuning(_detuning), polarization(_polarization), wavevector(_wavevector)
    {
    }
};

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

        loadBeamParams();
    }

    std::vector<Beam> getBeamParams() { return beams; }

  private:
    toml::parse_result sourceTomlDevice;
    toml::parse_result sourceTomlQubit;
    toml::parse_result sourceTomlGateDecomposition;

    std::vector<Beam> beams;

    void loadBeamParams()
    {
        // Read in the gate decomposition beam parameters from toml file.
        // The toml contains a list of beams, where each beam has the following fields:
        //   rabi = 4.4
        //   detuning = 5.5
        //   polarization = [6,7]
        //   wavevector = [8,9]
        //
        // The i-th beam must be used by gates on the i-th qubit.

        toml::node_view<toml::node> beamsToml = sourceTomlGateDecomposition["beams"];
        size_t numBeams = beamsToml.as_array()->size();

        for (size_t i = 0; i < numBeams; i++) {
            auto beam = beamsToml[i];
            double rabi = beam["rabi"].as_floating_point()->get();
            double detuning = beam["detuning"].as_floating_point()->get();
            std::vector<int64_t> polarization =
                tomlArray2StdVector<int64_t>(*(beam["polarization"].as_array()));
            std::vector<int64_t> wavevector =
                tomlArray2StdVector<int64_t>(*(beam["wavevector"].as_array()));

            beams.push_back(Beam(rabi, detuning, polarization, wavevector));
        }
    }
};

} // namespace
