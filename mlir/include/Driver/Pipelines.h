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
class Pipeline {
  public:
    using PipelineFunc = void (*)(mlir::OpPassManager &);

    Pipeline() {}

    void addPass(std::string &pass) { passes.push_back(pass); }

    mlir::LogicalResult addPipeline(mlir::OpPassManager &pm)
    {
        if (registerFunc) {
            registerFunc(pm);
            return mlir::success();
        }
        return mlir::failure();
    }

    const std::string &getName() const { return name; }

    const llvm::SmallVector<std::string> &getPasses() const { return passes; }

    PipelineFunc getRegisterFunc() const { return registerFunc; }

    void setRegisterFunc(PipelineFunc func) { registerFunc = func; }

    void setName(const std::string &pipelineName) { name = pipelineName; }

    void setPasses(const llvm::SmallVector<std::string> &newPasses) { passes = newPasses; }

  private:
    std::string name;
    llvm::SmallVector<std::string> passes;
    PipelineFunc registerFunc;
};

std::vector<Pipeline> getDefaultPipeline();

} // namespace driver
} // namespace catalyst
