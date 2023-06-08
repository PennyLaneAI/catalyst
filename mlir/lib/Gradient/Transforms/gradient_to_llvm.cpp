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

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
    }

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        LLVMTypeConverter typeConverter(context);

        RewritePatternSet patterns(context);
        populateConversionPatterns(typeConverter, patterns);

        LLVMConversionTarget target(*context);
        target.addIllegalDialect<GradientDialect>();
        target.addLegalDialect<catalyst::quantum::QuantumDialect>();
        target.addLegalDialect<func::FuncDialect>();

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