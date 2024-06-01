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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Catalyst/Transforms/Passes.h"
#include "Catalyst/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst;

namespace catalyst {

#define GEN_PASS_DEF_CATALYSTBUFFERIZATIONPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct CatalystBufferizationPass : impl::CatalystBufferizationPassBase<CatalystBufferizationPass> {
    using CatalystBufferizationPassBase::CatalystBufferizationPassBase;

    void runOnOperation() final
    {
        MLIRContext *context = &getContext();
        bufferization::BufferizeTypeConverter typeConverter;

        RewritePatternSet patterns(context);
        populateBufferizationPatterns(typeConverter, patterns);
        populateFunctionOpInterfaceTypeConversionPattern<CallbackOp>(patterns, typeConverter);

        ConversionTarget target(*context);
        bufferization::populateBufferizeMaterializationLegality(target);
        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
        target.addDynamicallyLegalOp<PrintOp>(
            [&](PrintOp op) { return typeConverter.isLegal(op); });
        target.addDynamicallyLegalOp<CustomCallOp>(
            [&](CustomCallOp op) { return typeConverter.isLegal(op); });
        target.addDynamicallyLegalOp<CallbackOp>([&](CallbackOp op) {
            return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                   typeConverter.isLegal(&op.getBody()) && op.getResultTypes().empty();
        });

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createCatalystBufferizationPass()
{
    return std::make_unique<CatalystBufferizationPass>();
}

} // namespace catalyst
