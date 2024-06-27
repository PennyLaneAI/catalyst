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

#define DEBUG_TYPE "chained-hadamards"

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace {

struct ChainedHadamardsOpRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    /// We check if the operation and it's parent are hadamard operations. If so, replace op
    /// with it's "grandparent".
    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following hadamard operation:\n" << op << "\n");
        if (op.getGateName().str() != "Hadamard")
            return failure();

        ValueRange qbs = op.getInQubits();
        auto parentHadamard = dyn_cast<CustomOp>(qbs[0].getDefiningOp());

        if (parentHadamard == nullptr)
            return failure();

        if (parentHadamard.getGateName().str() != "Hadamard")
            return failure();

        Value simplifiedVal = parentHadamard.getInQubits()[0];
        rewriter.replaceOp(op, simplifiedVal);
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateHadamardsPatterns(RewritePatternSet &patterns)
{
    patterns.add<ChainedHadamardsOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
