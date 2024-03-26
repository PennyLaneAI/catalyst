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
#include "Catalyst/IR/CatalystDialect.h"
#include "Catalyst/IR/CatalystOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace catalyst;

namespace {
struct LowerQuantumModuleToPayloadRewritePattern : public mlir::OpRewritePattern<func::CallOp> {
    using mlir::OpRewritePattern<func::CallOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(func::CallOp callOp,
                                        mlir::PatternRewriter &rewriter) const override
    {

        return failure();
    }
};

} // namespace

namespace catalyst {

void populateLowerQuantumModuleToPayloadPatterns(RewritePatternSet &patterns)
{
    patterns.add<LowerQuantumModuleToPayloadRewritePattern>(patterns.getContext());
}

} // namespace catalyst
