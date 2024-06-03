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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

#include "Gradient/IR/GradientDialect.h"
#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Transforms/Patterns.h"
#include "Quantum/IR/QuantumDialect.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace catalyst {
namespace gradient {

#define GEN_PASS_DECL_GRADIENTCONVERSIONPASS
#define GEN_PASS_DEF_GRADIENTCONVERSIONPASS
#include "Gradient/Transforms/Passes.h.inc"

struct GradientConversionPass : impl::GradientConversionPassBase<GradientConversionPass> {
    using GradientConversionPassBase::GradientConversionPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        LowerToLLVMOptions options(context);
        options.useGenericFunctions = useGenericFunctions;

        LLVMTypeConverter typeConverter(context, options);

        RewritePatternSet patterns(context);
        populateConversionPatterns(typeConverter, patterns);

        LLVMConversionTarget target(*context);
        target.addIllegalDialect<GradientDialect>();
        target.addLegalDialect<catalyst::quantum::QuantumDialect>();
        target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect, func::FuncDialect,
                               index::IndexDialect, memref::MemRefDialect>();

        // This is a bit unfortunate.
        // We need custom grad to have the three functions in terms of llvm pointers.
        // but that can't be achieved until CatalystLowering.
        // Because the original function is a callback, which is on the Catalyst dialect.
        // And won't be changed to llvm pointers until the catalyst lowering process.
        // A potential solution would be to have a custom grad op in catalyst
        // and just change dialects. But this seems a bit redundant.
        // Or yet, add another wrapper...
        // For now this seems good enough.
        target.addLegalOp<CustomGradOp>();

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
