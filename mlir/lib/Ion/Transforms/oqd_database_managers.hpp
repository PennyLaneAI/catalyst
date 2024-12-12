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

static const std::string catalyst_root_path = std::regex_replace(
    __FILE__, std::regex("mlir/(.)*/oqd_database_managers.hpp"), "");
static const std::string oqd_device_parameters_toml_file_path =
    catalyst_root_path + "frontend/catalyst/third_party/oqd/src/oqd_device_parameters.toml";
static const std::string oqd_qubit_parameters_toml_file_path =
    catalyst_root_path + "frontend/catalyst/third_party/oqd/src/oqd_qubit_parameters.toml";

static const std::string oqd_gate_decomposition_parameters_toml_file_path =
    catalyst_root_path +
    "frontend/catalyst/third_party/oqd/src/oqd_gate_decomposition_parameters.toml";

toml::parse_result load_toml_file(const std::string &path) { return toml::parse_file(path); }

template <typename T> std::vector<T> TomlArray2StdVector(toml::array arr)
{
    std::vector<T> vec;
    for (size_t i = 0; i < arr.size(); i++) {
        if constexpr (std::is_same_v<T, int64_t>) {
            vec.push_back(arr[i].as_integer()->get());
        }
        else if constexpr (std::is_same_v<T, double>) {
            vec.push_back(arr[i].as_floating_point()->get());
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

std::vector<Beam> getBeams1Params()
{
    // Read in the 1-qubit gate decomposition beam parameters from toml file.
    // The toml contains a list of beams, where each beam has the following fields:
    //   rabi = 4.4
    //   detuning = 5.5
    //   polarization = [6,7]
    //   wavevector = [8,9]
    //
    // The i-th beam must be used by gates on the i-th qubit.

    toml::parse_result SourceToml =
        load_toml_file(oqd_gate_decomposition_parameters_toml_file_path);

    assert(SourceToml && "Parsing of gate decomposition beam toml failed!");

    toml::node_view<toml::node> TomlBeams = SourceToml["beams"];
    size_t NumOfBeams = TomlBeams.as_array()->size();
    std::vector<Beam> beams;

    for (size_t i = 0; i < NumOfBeams; i++) {
        auto beam = TomlBeams[i];
        double rabi = beam["rabi"].as_floating_point()->get();
        double detuning = beam["detuning"].as_floating_point()->get();
        std::vector<int64_t> polarization =
            TomlArray2StdVector<int64_t>(*(beam["polarization"].as_array()));
        std::vector<int64_t> wavevector =
            TomlArray2StdVector<int64_t>(*(beam["wavevector"].as_array()));

        beams.push_back(Beam(rabi, detuning, polarization, wavevector));
    }

    return beams;
}
