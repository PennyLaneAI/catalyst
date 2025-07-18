// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <filesystem>  // path
#include <fstream>  // ifstream
#include <regex>  //regex

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mhlo/IR/register.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/Register.h"

#include "mhlo/IR/hlo_ops.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/Transforms/BufferizableOpInterfaceImpl.h"
#include "Catalyst/Transforms/Passes.h"
#include "Gradient/IR/GradientDialect.h"
#include "Gradient/Transforms/BufferizableOpInterfaceImpl.h"
#include "Gradient/Transforms/Passes.h"
#include "Ion/IR/IonDialect.h"
#include "MBQC/IR/MBQCDialect.h"
#include "Mitigation/IR/MitigationDialect.h"
#include "Mitigation/Transforms/Passes.h"
#include "QEC/IR/QECDialect.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/Transforms/BufferizableOpInterfaceImpl.h"
#include "Quantum/Transforms/Passes.h"

namespace test {
void registerTestDialect(mlir::DialectRegistry &);
} // namespace test

consteval std::string getCatalystVersion() {
    auto version = std::string{};
    auto version_file_path = std::filesystem::path{ "../../../../frontend/catalyst/_version.py" };
    auto version_file = std::ifstream{ version_file_path.string() };
    auto line = std::string{};
    while (std::getline(version_file, line)) {
        auto version_regex = std::regex{ "__version__ = \"([^\"]+)\"" };
        auto matches = std::smatch{};
        if (std::regex::match(line, matches, version_regex)) {
            version = matches[0].str();
            break;
        }
    }
    return version;
}

constinit const CATALYST_VERSION = getCatalystVersion();

void printCatalystVersion(llvm::raw_ostream &os) {
    os << "Catalyst version " << CATALYST_VERSION << std::endl;
}

int main(int argc, char **argv)
{
    llvm::cl::AddExtraVersionPrinter(printCatalystVersion);
    mlir::registerAllPasses();
    catalyst::registerAllCatalystPasses();
    mlir::mhlo::registerAllMhloPasses();

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    test::registerTestDialect(registry);
    mlir::mhlo::registerAllMhloDialects(registry);
    mlir::stablehlo::registerAllDialects(registry);
    mlir::func::registerAllExtensions(registry);
    registry.insert<catalyst::CatalystDialect>();
    registry.insert<catalyst::quantum::QuantumDialect>();
    registry.insert<catalyst::qec::QECDialect>();
    registry.insert<catalyst::gradient::GradientDialect>();
    registry.insert<catalyst::mbqc::MBQCDialect>();
    registry.insert<catalyst::mitigation::MitigationDialect>();
    registry.insert<catalyst::ion::IonDialect>();
    registry.insert<mlir::mhlo::MhloDialect>();

    catalyst::registerBufferizableOpInterfaceExternalModels(registry);
    catalyst::gradient::registerBufferizableOpInterfaceExternalModels(registry);
    catalyst::quantum::registerBufferizableOpInterfaceExternalModels(registry);

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Quantum optimizer driver\n", registry));
}
