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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Passes.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace quantum {

struct QuantumToOpenQasm3Pass
    : public PassWrapper<QuantumToOpenQasm3Pass, OperationPass<ModuleOp>> {
    QuantumToOpenQasm3Pass() {}

    StringRef getArgument() const override { return "convert-quantum-to-openqasm3"; }

    StringRef getDescription() const override { return "Convert quantum dialect to openqasm3."; }

    void getDependentDialects(DialectRegistry &registry) const override
    {
    }

    void runOnOperation() final
    {
    }
};

} // namespace quantum

std::unique_ptr<Pass> createQuantumToOpenQasm3Pass()
{
    return std::make_unique<quantum::QuantumToOpenQasm3Pass>();
}

} // namespace catalyst
