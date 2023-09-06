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

#define DEBUG_TYPE "scatter"

#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mhlo/IR/hlo_ops.h"

using llvm::dbgs;

using namespace mlir;

namespace {

struct ScatterOpRewritePattern : public mlir::OpRewritePattern<mlir::mhlo::ScatterOp> {
    using mlir::OpRewritePattern<mlir::mhlo::ScatterOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(mlir::mhlo::ScatterOp scatter,
                                        mlir::PatternRewriter &rewriter) const override
    {
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateScatternPatterns(RewritePatternSet &patterns)
{
    patterns.add<ScatterOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
