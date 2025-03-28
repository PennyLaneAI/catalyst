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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Gradient/IR/GradientOps.h"
#include "Gradient/Transforms/Passes.h"
#include "Gradient/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace catalyst {
namespace gradient {

#define GEN_PASS_DEF_GRADIENTBUFFERIZATIONPASS
#include "Gradient/Transforms/Passes.h.inc"

struct GradientBufferizationPass : impl::GradientBufferizationPassBase<GradientBufferizationPass> {
    using GradientBufferizationPassBase::GradientBufferizationPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        bufferization::BufferizeTypeConverter typeConverter;

        RewritePatternSet patterns(context);
        populateBufferizationPatterns(typeConverter, patterns);
        populateFunctionOpInterfaceTypeConversionPattern<ForwardOp>(patterns, typeConverter);
        populateFunctionOpInterfaceTypeConversionPattern<ReverseOp>(patterns, typeConverter);

        ConversionTarget target(*context);
        bufferization::populateBufferizeMaterializationLegality(target);
        // Default to operations being legal with the exception of the ones below.
        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
        // Gradient ops which return arrays need to be marked illegal when the type is a tensor.
        target.addDynamicallyLegalOp<AdjointOp>(
            [&](AdjointOp op) { return typeConverter.isLegal(op); });

        target.addDynamicallyLegalOp<ForwardOp>([&](ForwardOp op) {
            return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                   typeConverter.isLegal(&op.getBody()) && !op.empty();
        });

        target.addDynamicallyLegalOp<ReverseOp>([&](ReverseOp op) {
            return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                   typeConverter.isLegal(&op.getBody()) && !op.empty();
        });

        target.addDynamicallyLegalOp<BackpropOp>(
            [&](BackpropOp op) { return typeConverter.isLegal(op); });

        target.addDynamicallyLegalOp<catalyst::gradient::ReturnOp>(
            [&](catalyst::gradient::ReturnOp op) { return typeConverter.isLegal(op); });

        target.addLegalDialect<func::FuncDialect, memref::MemRefDialect>();

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace gradient

std::unique_ptr<Pass> createGradientBufferizationPass()
{
    return std::make_unique<gradient::GradientBufferizationPass>();
}

} // namespace catalyst
