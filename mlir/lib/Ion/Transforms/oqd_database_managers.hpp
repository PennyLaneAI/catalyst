#include <iostream>
#include <regex>
#include <toml++/toml.hpp>

static const std::string catalyst_root_path
 = std::regex_replace(__FILE__, std::regex("mlir/lib/Ion/Transforms/./oqd_database_managers.hpp") ,"");
static const std::string oqd_device_parameters_toml_file_path =
catalyst_root_path + "frontend/catalyst/third_party/oqd/src/oqd_device_parameters.toml";
static const std::string oqd_qubit_parameters_toml_file_path =
catalyst_root_path + "frontend/catalyst/third_party/oqd/src/oqd_qubit_parameters.toml";

static const std::string oqd_gate_decomposition_parameters_toml_file_path =
catalyst_root_path + "frontend/catalyst/third_party/oqd/src/oqd_gate_decomposition_parameters.toml";

toml::parse_result load_toml_file(const std::string& path){
    return toml::parse_file(path);
}


void getGateDecompositionParams(){
    toml::parse_result SourceToml = load_toml_file(oqd_gate_decomposition_parameters_toml_file_path);
    std::cout << SourceToml << "\n";
}
