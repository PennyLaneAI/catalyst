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

#define DEBUG_TYPE "chained-hadamard"

#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"
#include "Quantum/Utils/QuantumSplitting.h"

using llvm::dbgs;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace {

struct ChainedHadamardOpRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    /// We check if the operation and it's parent are hadamard operations. If so, replace op
    /// with it's "grandparent".
    mlir::LogicalResult matchAndRewrite(CustomOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following hadamard operation:\n" << op << "\n");
        if (op.getGateName().str() != "Hadamard")
            return failure();

        ValueRange qbs = op.getInQubits();
        Operation *parent = qbs[0].getDefiningOp();
        CustomOp *parentHadamard = dyn_cast<CustomOp>(parent);

        if (parentHadamard == nullptr)
            return failure();

        if (parentHadamard->getGateName().str() != "Hadamard")
            return failure();

        Value simplifiedVal = parentHadamard->getInQubits()[0];
        rewriter.replaceOp(op, simplifiedVal);
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateChainedHadamardPatterns(RewritePatternSet &patterns)
{
    patterns.add<ChainedHadamardOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
