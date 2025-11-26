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
        // NOTE: a bit unorthodox, but we find the *second* PPR in a pair, since it's easier to
        // look backwards by checking inQubits.getDefiningOp
        ValueRange inQubits = op.getInQubits();
        auto definingOp = inQubits[0].getDefiningOp();

        if (!definingOp) {
            return failure();
        }

        auto prevOp = dyn_cast<PPRotationOp>(definingOp);

        if (!prevOp) {
            return failure();
        }

        // if pauli string too long
        if (exceedPauliSizeLimit(op.getPauliProduct().size(), MAX_PAULI_SIZE)) {
            return failure();
        }

        // check same pauli strings
        if (op.getPauliProduct() != prevOp.getPauliProduct()) {
            return failure();
        }

        int16_t opRotation = static_cast<int16_t>(op.getRotationKind());
        int16_t prevOpRotation = static_cast<int16_t>(prevOp.getRotationKind());

        // cancel inverse operations
        if (opRotation == -prevOpRotation) {
            // erase in reverse to avoid use issues
            ValueRange originalQubits = prevOp.getInQubits();
            rewriter.replaceOp(op, originalQubits);
            rewriter.replaceOp(prevOp, originalQubits);

            return success();
        }

        if (opRotation != prevOpRotation) {
            return failure();
        }

        int16_t newAngle = opRotation / 2;

        // newAngle of 1 indicates denominator of 1
        if (newAngle != 1 and newAngle != -1) {
            // "replace" the operation by changing the rotationKind
            prevOp.setRotationKind(newAngle);

            // replace references to current op with prevOp
            rewriter.replaceOp(op, prevOp);
        }
        else {
            ValueRange originalQubits = prevOp.getInQubits();
            rewriter.replaceOp(op, originalQubits);
            rewriter.replaceOp(prevOp, originalQubits);
        }

        return success();
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
