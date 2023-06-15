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

#define DEBUG_TYPE "adjoint"

#include <memory>
#include <vector>

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;

namespace catalyst {
namespace quantum {

struct AdjointDistributionPass : public OperationPass<ModuleOp> {
    AdjointDistributionPass() : OperationPass<ModuleOp>(TypeID::get<AdjointDistributionPass>()) {}
    AdjointDistributionPass(const AdjointDistributionPass &other) : OperationPass<ModuleOp>(other)
    {
    }

    StringRef getName() const override { return "AdjointLoweringPass"; }

    StringRef getArgument() const override { return "adjoint-distribution"; }

    StringRef getDescription() const override
    {
        return "Distribute adjoint over adjoint MLIR regions.";
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<arith::ArithDialect>();
        registry.insert<linalg::LinalgDialect>();
        registry.insert<index::IndexDialect>();
        registry.insert<tensor::TensorDialect>();
        registry.insert<memref::MemRefDialect>();
        registry.insert<bufferization::BufferizationDialect>();
    }

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "adjoint distribution pass"
                          << "\n");
    }

    std::unique_ptr<Pass> clonePass() const override
    {
        return std::make_unique<AdjointDistributionPass>(*this);
    }
};

struct AdjointLoweringPass : public OperationPass<ModuleOp> {
    AdjointLoweringPass() : OperationPass<ModuleOp>(TypeID::get<AdjointLoweringPass>()) {}
    AdjointLoweringPass(const AdjointLoweringPass &other) : OperationPass<ModuleOp>(other) {}

    StringRef getName() const override { return "AdjointLoweringPass"; }

    StringRef getArgument() const override { return "adjoint-lowering"; }

    StringRef getDescription() const override
    {
        return "Lower adjoint regions containing a single quantum operations.";
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<arith::ArithDialect>();
        registry.insert<linalg::LinalgDialect>();
        registry.insert<index::IndexDialect>();
        registry.insert<tensor::TensorDialect>();
        registry.insert<memref::MemRefDialect>();
        registry.insert<bufferization::BufferizationDialect>();
    }

    void runOnOperation() final
    {
        LLVM_DEBUG(dbgs() << "adjoint lowering pass"
                          << "\n");
        ModuleOp op = getOperation();

        RewritePatternSet patterns(&getContext());
        populateAdjointPatterns(patterns);

        if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
            return signalPassFailure();
        }
    }

    std::unique_ptr<Pass> clonePass() const override
    {
        return std::make_unique<AdjointLoweringPass>(*this);
    }
};

} // namespace quantum

std::unique_ptr<Pass> createAdjointDistributionPass()
{
    return std::make_unique<quantum::AdjointDistributionPass>();
}

std::unique_ptr<Pass> createAdjointLoweringPass()
{
    return std::make_unique<quantum::AdjointLoweringPass>();
}

} // namespace catalyst
