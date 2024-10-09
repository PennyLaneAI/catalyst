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
#include "Catalyst/Transforms/Passes.h"
#include "Gradient/IR/GradientDialect.h"
#include "Gradient/Transforms/Passes.h"
#include "Mitigation/IR/MitigationDialect.h"
#include "Mitigation/Transforms/Passes.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/Transforms/Passes.h"

namespace test {
void registerTestDialect(mlir::DialectRegistry &);
} // namespace test

int main(int argc, char **argv)
{
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
    registry.insert<catalyst::gradient::GradientDialect>();
    registry.insert<catalyst::mitigation::MitigationDialect>();
    registry.insert<mlir::mhlo::MhloDialect>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Quantum optimizer driver\n", registry));
}
