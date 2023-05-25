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

#include <memory>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace catalyst {
namespace gradient {

struct GradientLoweringPass : public OperationPass<ModuleOp> {
    GradientLoweringPass() : OperationPass<ModuleOp>(TypeID::get<GradientLoweringPass>()) {}
    GradientLoweringPass(const GradientLoweringPass &other) : OperationPass<ModuleOp>(other) {}

    StringRef getName() const override { return "GradientLoweringPass"; }

    StringRef getArgument() const override { return "lower-gradients"; }

    StringRef getDescription() const override
    {
        return "Lower gradient operation to MLIR operation.";
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<arith::ArithDialect>();
        registry.insert<linalg::LinalgDialect>();
        registry.insert<index::IndexDialect>();
        registry.insert<tensor::TensorDialect>();
        registry.insert<scf::SCFDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<memref::MemRefDialect>();
        registry.insert<bufferization::BufferizationDialect>();
    }

    void runOnOperation() final
    {
        ModuleOp op = getOperation();
        TypeConverter vectorTypeConverter;
        vectorTypeConverter.addConversion([](Type type) -> llvm::Optional<Type> {
            if (MemRefType::isValidElementType(type)) {
                return type;
            }
            return llvm::None;
        });
        vectorTypeConverter.addConversion(
            [](ParameterVectorType type, SmallVectorImpl<Type> &resultTypes) {
                // Data
                resultTypes.push_back(MemRefType::get(
                    {}, MemRefType::get({ShapedType::kDynamic}, type.getElementType())));
                auto indexMemRef = MemRefType::get({}, IndexType::get(type.getContext()));
                // Size
                resultTypes.push_back(indexMemRef);
                // Capacity
                resultTypes.push_back(indexMemRef);
                return success();
            });

        RewritePatternSet gradientPatterns(&getContext());
        populateLoweringPatterns(gradientPatterns, lowerOnly);

        // This is required to remove qubit values returned by if/for ops in the
        // quantum gradient function of the parameter-shift pattern.
        scf::IfOp::getCanonicalizationPatterns(gradientPatterns, &getContext());
        scf::ForOp::getCanonicalizationPatterns(gradientPatterns, &getContext());
        catalyst::quantum::InsertOp::getCanonicalizationPatterns(gradientPatterns, &getContext());
        catalyst::quantum::DeallocOp::getCanonicalizationPatterns(gradientPatterns, &getContext());

        if (failed(applyPatternsAndFoldGreedily(op, std::move(gradientPatterns)))) {
            return signalPassFailure();
        }

        if (lowerVector) {
            RewritePatternSet gradientVectorPatterns(&getContext());
            ConversionTarget target(getContext());
            target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect, func::FuncDialect,
                                   scf::SCFDialect>();
            target.addLegalOp<UnrealizedConversionCastOp>();
            target.addIllegalOp<VectorInitOp, VectorPushOp, VectorSizeOp, VectorLoadDataOp>();

            populateVectorLoweringPatterns(vectorTypeConverter, gradientVectorPatterns);

            if (failed(applyPartialConversion(op, target, std::move(gradientVectorPatterns)))) {
                return signalPassFailure();
            }
        }
    }

    std::unique_ptr<Pass> clonePass() const override
    {
        return std::make_unique<GradientLoweringPass>(*this);
    }

  protected:
    Option<std::string> lowerOnly{
        *this, "only", llvm::cl::desc("Restrict lowering to a specific type of gradient.")};

    Option<bool> lowerVector{*this, "lower-vectors",
                             llvm::cl::desc("Lower gradient vector operations."),
                             llvm::cl::init(true)};
};

} // namespace gradient

std::unique_ptr<Pass> createGradientLoweringPass()
{
    return std::make_unique<gradient::GradientLoweringPass>();
}

} // namespace catalyst
