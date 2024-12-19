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

#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace catalyst {

std::unique_ptr<mlir::Pass> createQuantumBufferizationPass();
std::unique_ptr<mlir::Pass> createQuantumConversionPass();
std::unique_ptr<mlir::Pass> createEmitCatalystPyInterfacePass();
std::unique_ptr<mlir::Pass> createCopyGlobalMemRefPass();
std::unique_ptr<mlir::Pass> createAdjointLoweringPass();
std::unique_ptr<mlir::Pass> createRemoveChainedSelfInversePass();
std::unique_ptr<mlir::Pass> createAnnotateFunctionPass();
std::unique_ptr<mlir::Pass> createSplitMultipleTapesPass();
std::unique_ptr<mlir::Pass> createMergeRotationsPass();
std::unique_ptr<mlir::Pass> createPropagateSimpleStatesTesterPass();
std::unique_ptr<mlir::Pass> createDisentangleCNOTPass();
std::unique_ptr<mlir::Pass> createIonsDecompositionPass();
std::unique_ptr<mlir::Pass> createStaticCustomLoweringPass();

} // namespace catalyst
