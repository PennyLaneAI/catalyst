// Copyright 2023 Xanadu Quantum Technologies Inc.

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

std::unique_ptr<mlir::Pass> createCatalystBufferizationPass();
std::unique_ptr<mlir::Pass> createArrayListToMemRefPass();
std::unique_ptr<mlir::Pass> createCatalystConversionPass();
std::unique_ptr<mlir::Pass> createScatterLoweringPass();
std::unique_ptr<mlir::Pass> createHloCustomCallLoweringPass();
std::unique_ptr<mlir::Pass> createQnodeToAsyncLoweringPass();
std::unique_ptr<mlir::Pass> createDisableAssertionPass();
std::unique_ptr<mlir::Pass> createAddExceptionHandlingPass();
std::unique_ptr<mlir::Pass> createGEPInboundsPass();
std::unique_ptr<mlir::Pass> createRegisterInactiveCallbackPass();
std::unique_ptr<mlir::Pass> createMemrefCopyToLinalgCopyPass();
std::unique_ptr<mlir::Pass> createApplyTransformSequencePass();
std::unique_ptr<mlir::Pass> createDetensorizeSCFPass();

void registerAllCatalystPasses();

} // namespace catalyst
