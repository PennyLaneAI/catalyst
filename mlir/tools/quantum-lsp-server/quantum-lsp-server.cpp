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

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Gradient/IR/GradientDialect.h"
#include "Ion/IR/IonDialect.h"
#include "MBQC/IR/MBQCDialect.h"
#include "Mitigation/IR/MitigationDialect.h"
#include "QEC/IR/QECDialect.h"
#include "Quantum/IR/QuantumDialect.h"

#include "mhlo/IR/register.h"
#include "stablehlo/dialect/Register.h"

int main(int argc, char **argv)
{
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert<catalyst::CatalystDialect>();
    registry.insert<catalyst::quantum::QuantumDialect>();
    registry.insert<catalyst::qec::QECDialect>();
    registry.insert<catalyst::gradient::GradientDialect>();
    registry.insert<catalyst::mbqc::MBQCDialect>();
    registry.insert<catalyst::mitigation::MitigationDialect>();
    registry.insert<catalyst::ion::IonDialect>();

    mlir::mhlo::registerAllMhloDialects(registry);
    mlir::stablehlo::registerAllDialects(registry);

    return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
