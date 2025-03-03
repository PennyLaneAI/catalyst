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

#include "mlir/IR/Operation.h"
#include "llvm/Support/Casting.h"
#define DEBUG_TYPE "commute-clifford-t-ppr"


#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::qec;

namespace {

    struct CommuteCliffordTPPR : public OpRewritePattern<PPRotationOp> {
        using OpRewritePattern::OpRewritePattern;

        LogicalResult matchAndRewrite(PPRotationOp op, PatternRewriter &rewriter) const override {
            // check if next gate is commuteed or not
            
             // TESTING
            auto name = op.getPauliProduct();

            for (Operation *nextOp: op->getUsers()){
                if (PPRotationOp nextPPROp = dyn_cast_or_null<PPRotationOp>(nextOp)){
                    if (name == nextPPROp.getPauliProductAttrName()){
                        llvm::errs() << "name work here";
                    }
                    if (op.isCommuted(nextPPROp)){
                        llvm::errs() << "IT work";
                    }
                }
            }

            return failure();
        }
    };
}

namespace catalyst {
namespace qec {

void populateCommuteCliffordTPPRPatterns(mlir::RewritePatternSet &patterns)
{
    patterns.add<CommuteCliffordTPPR>(patterns.getContext(), 1);
}
}

}
