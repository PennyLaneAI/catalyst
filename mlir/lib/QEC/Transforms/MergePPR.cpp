// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "merge-ppr"

#include "mlir/Analysis/TopologicalSortUtils.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/IR/QECOpInterfaces.h"
#include "QEC/Transforms/Patterns.h"
#include "QEC/Utils/PauliStringWrapper.h"

using namespace mlir;
using namespace catalyst::qec;

namespace {

struct MergePPR : public OpRewritePattern<PPRotationOp> {
    using OpRewritePattern::OpRewritePattern;

    size_t MAX_PAULI_SIZE;

    MergePPR(mlir::MLIRContext *context, size_t maxPauliSize, PatternBenefit benefit)
        : OpRewritePattern(context), MAX_PAULI_SIZE(maxPauliSize)
    {
    }

    LogicalResult matchAndRewrite(PPRotationOp op, PatternRewriter &rewriter) const override
    {
        // TODO
        // check that next op is PPR

        // check same pauli strings

        // merge angles
        // delete ops
        return failure();
    }
};
} // namespace

namespace catalyst {
namespace qec {

void populateMergePPRPatterns(mlir::RewritePatternSet &patterns, unsigned int maxPauliSize)
{
    patterns.add<MergePPR>(patterns.getContext(), maxPauliSize, 1);
}
} // namespace qec

} // namespace catalyst
