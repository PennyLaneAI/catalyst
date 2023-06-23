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

#define GEN_PASS_DEF_QUANTUMBUFFERIZATIONPASS
#include "Quantum/Transforms/Passes.h.inc"

struct QuantumBufferizationPass
    : impl::QuantumBufferizationPassBase<QuantumBufferizationPass> {
    using QuantumBufferizationPassBase::QuantumBufferizationPassBase;

    void runOnOperation() final {
        MLIRContext *context = &getContext();
        bufferization::BufferizeTypeConverter typeConverter;

        RewritePatternSet patterns(context);
        populateBufferizationPatterns(typeConverter, patterns);

        ConversionTarget target(*context);
        bufferization::populateBufferizeMaterializationLegality(target);
        populateBufferizationLegality(typeConverter, target);

        if (failed(applyPartialConversion(getOperation(), target,
                                          std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace quantum

std::unique_ptr<Pass> createQuantumBufferizationPass() {
    return std::make_unique<quantum::QuantumBufferizationPass>();
}

} // namespace catalyst
