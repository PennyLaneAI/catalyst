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

#include <filesystem> // path
#include <fstream>    // ifstream
#include <regex>      //regex

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/c/StablehloPasses.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/Transforms/BufferizableOpInterfaceImpl.h"
#include "Catalyst/Utils/PrintVersion.h"
#include "Gradient/IR/GradientDialect.h"
#include "Gradient/Transforms/BufferizableOpInterfaceImpl.h"
#include "Ion/IR/IonDialect.h"
#include "MBQC/IR/MBQCDialect.h"
#include "Mitigation/IR/MitigationDialect.h"
#include "PauliFrame/IR/PauliFrameDialect.h"
#include "QEC/IR/QECDialect.h"
#include "QRef/IR/QRefDialect.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/Transforms/BufferizableOpInterfaceImpl.h"
#include "RTIO/IR/RTIODialect.h"
#include "RegisterAllPasses.h"

namespace test {
void registerTestDialect(mlir::DialectRegistry &);
} // namespace test

int main(int argc, char **argv)
{
    llvm::cl::AddExtraVersionPrinter(catalyst::printVersion);
    mlir::registerAllPasses();
    catalyst::registerAllPasses();
    mlirRegisterAllStablehloPasses();
    mlir::stablehlo::registerOptimizationPasses();

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    test::registerTestDialect(registry);
    mlir::stablehlo::registerAllDialects(registry);
    mlir::func::registerAllExtensions(registry);
    registry.insert<catalyst::CatalystDialect>();
    registry.insert<catalyst::quantum::QuantumDialect>();
    registry.insert<catalyst::qref::QRefDialect>();
    registry.insert<catalyst::qec::QECDialect>();
    registry.insert<catalyst::gradient::GradientDialect>();
    registry.insert<catalyst::mbqc::MBQCDialect>();
    registry.insert<catalyst::mitigation::MitigationDialect>();
    registry.insert<catalyst::pauli_frame::PauliFrameDialect>();
    registry.insert<catalyst::ion::IonDialect>();
    registry.insert<catalyst::rtio::RTIODialect>();
    registry.insert<mlir::stablehlo::StablehloDialect>();

    catalyst::registerBufferizableOpInterfaceExternalModels(registry);
    catalyst::gradient::registerBufferizableOpInterfaceExternalModels(registry);
    catalyst::quantum::registerBufferizableOpInterfaceExternalModels(registry);

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Quantum optimizer driver\n", registry));
}
