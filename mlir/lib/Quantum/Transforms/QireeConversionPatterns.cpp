// Copyright 2022-2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace {

constexpr int64_t UNKNOWN = ShapedType::kDynamic;
constexpr int32_t NO_POSTSELECT = -1;


////////////////////////
// Runtime Management //
////////////////////////

template <typename T> struct RTBasedPattern : public OpRewritePattern<T> {
    using OpRewritePattern<T>::OpRewritePattern;

    LogicalResult matchAndRewrite(T op,
                                  PatternRewriter &rewriter) const override
    {
        rewriter.eraseOp(op);

        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateQIREEConversionPatterns(RewritePatternSet &patterns)
{
    patterns.add<RTBasedPattern<InitializeOp>>( patterns.getContext());
    patterns.add<RTBasedPattern<FinalizeOp>>(patterns.getContext());
}

} // namespace quantum
} // namespace catalyst
