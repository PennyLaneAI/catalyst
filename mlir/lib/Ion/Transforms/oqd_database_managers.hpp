#include <cassert>
#include <iostream>
#include <regex>
#include <toml++/toml.hpp>

static const std::string catalyst_root_path = std::regex_replace(
    __FILE__, std::regex("mlir/lib/Ion/Transforms/./oqd_database_managers.hpp"), "");
static const std::string oqd_device_parameters_toml_file_path =
    catalyst_root_path + "frontend/catalyst/third_party/oqd/src/oqd_device_parameters.toml";
static const std::string oqd_qubit_parameters_toml_file_path =
    catalyst_root_path + "frontend/catalyst/third_party/oqd/src/oqd_qubit_parameters.toml";

static const std::string oqd_gate_decomposition_parameters_toml_file_path =
    catalyst_root_path +
    "frontend/catalyst/third_party/oqd/src/oqd_gate_decomposition_parameters.toml";

toml::parse_result load_toml_file(const std::string &path) { return toml::parse_file(path); }

std::vector<size_t> TomlArray2StdVector(toml::array arr)
{
    std::vector<size_t> vec;
    for (size_t i = 0; i < arr.size(); i++) {
        vec.push_back(arr[i].as_integer()->get());
    }
    return vec;
}

struct Beam {
    // This struct contains the calibrated beam parameters.
    double rabi, detuning;
    std::vector<size_t> polarization, wavevector;

    Beam(double rabi, double detuning, std::vector<size_t> polarization,
         std::vector<size_t> wavevector)
        : rabi(rabi), detuning(detuning), polarization(polarization), wavevector(wavevector)
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
        std::vector<size_t> polarization = TomlArray2StdVector(*(beam["polarization"].as_array()));
        std::vector<size_t> wavevector = TomlArray2StdVector(*(beam["wavevector"].as_array()));

        beams.push_back(Beam(rabi, detuning, polarization, wavevector));
    }

    return beams;
}
