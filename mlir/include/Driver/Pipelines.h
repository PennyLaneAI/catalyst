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

#pragma once

#include "mlir/Pass/Pass.h"

namespace catalyst {
namespace driver {

void createEnforceRuntimeInvariantsPipeline(mlir::OpPassManager &pm);
void createHloLoweringPipeline(mlir::OpPassManager &pm);
void createQuantumCompilationPipeline(mlir::OpPassManager &pm);
void createBufferizationPipeline(mlir::OpPassManager &pm);
void createLLVMDialectLoweringPipeline(mlir::OpPassManager &pm);
void createDefaultCatalystPipeline(mlir::OpPassManager &pm);

void registerEnforceRuntimeInvariantsPipeline();
void registerHloLoweringPipeline();
void registerQuantumCompilationPipeline();
void registerBufferizationPipeline();
void registerLLVMDialectLoweringPipeline();
void registerDefaultCatalystPipeline();
void registerAllCatalystPipelines();

} // namespace driver
} // namespace catalyst
