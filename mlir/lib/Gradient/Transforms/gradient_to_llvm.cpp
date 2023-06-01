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

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#include "Gradient/IR/GradientDialect.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Transforms/Patterns.h"
#include "Quantum/IR/QuantumDialect.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace catalyst {
namespace gradient {

struct GradientConversionPass
    : public PassWrapper<GradientConversionPass, OperationPass<ModuleOp>> {
    GradientConversionPass() {}

    StringRef getArgument() const override { return "convert-gradient-to-llvm"; }

    StringRef getDescription() const override
    {
        return "Perform a dialect conversion from Gradient to LLVM.";
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<LLVM::LLVMDialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<catalyst::quantum::QuantumDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<scf::SCFDialect>();
        registry.insert<memref::MemRefDialect>();
    }

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        LLVMTypeConverter typeConverter(context);

        RewritePatternSet patterns(context);
        populateConversionPatterns(typeConverter, patterns);
        mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        populateSCFToControlFlowConversionPatterns(patterns);
        // memref::populateExpandStridedMetadataPatterns(patterns);
        // populateMemRefToLLVMConversionPatterns(typeConverter, patterns);

        LLVMConversionTarget target(*context);
        target.addIllegalDialect<GradientDialect>();
        target.addIllegalDialect<arith::ArithDialect>();
        target.addIllegalOp<scf::ForOp>();

        target.addLegalDialect<catalyst::quantum::QuantumDialect>();
        target.addLegalDialect<func::FuncDialect>();
        target.addLegalDialect<memref::MemRefDialect>();
        // target.addIllegalDialect<memref::MemRefDialect>();

        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace gradient

std::unique_ptr<Pass> createGradientConversionPass()
{
    return std::make_unique<gradient::GradientConversionPass>();
}

} // namespace catalyst
